#!/usr/bin/env python3
"""Monthly backtest for the Russell 3000 log-area/momentum method."""
from __future__ import annotations
import argparse, json, math
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import pandas as pd
import yfinance as yf
from rank_russell3000_areas import RANKING_COLUMNS, select_low_area_high_return
from six_month_price_area import _absolute_linear_area

DEFAULT_DATA_ROOT = Path("data/russell3000")
DEFAULT_OUTPUT_DIR = Path("output/log_area_momentum_backtest")

@dataclass
class BacktestResult:
    average_yahoo_beta: float
    yahoo_beta_observation_count: int
    yahoo_beta_holding_count: int
    russell_3000_beta: float
    strategy_total_return: float
    russell_3000_total_return: float
    monthly: pd.DataFrame
    holdings: pd.DataFrame
    diagnostics: pd.DataFrame
    figure: Figure

def _raw_area(series: pd.Series) -> float:
    values = series.astype(float).map(math.log).tolist()
    if len(values) < 2: return math.nan
    n, start = len(values)-1, values[0]
    change = values[-1]-start
    gaps = [v-(start+change*i/n) for i,v in enumerate(values)]
    return float(sum(_absolute_linear_area(a,b) for a,b in zip(gaps,gaps[1:])))

def build_monthly_summary(prices: pd.DataFrame) -> pd.DataFrame:
    """Collapse daily closes into ticker/month statistics using vectorized geometry."""
    frame = prices[["date", "ticker", "adjusted_close_usd"]].copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    frame = frame.sort_values(["ticker", "date"]).reset_index(drop=True)
    frame["month"] = frame["date"].dt.to_period("M")
    keys = ["ticker", "month"]
    grouped = frame.groupby(keys, sort=False, observed=True)
    frame["_count"] = grouped["adjusted_close_usd"].transform("size")
    frame["_index"] = grouped.cumcount()
    frame["_log_price"] = frame["adjusted_close_usd"].map(math.log)
    frame["_start"] = grouped["_log_price"].transform("first")
    frame["_end"] = grouped["_log_price"].transform("last")
    denominator = (frame["_count"] - 1).where(frame["_count"].gt(1), 1)
    baseline = frame["_start"] + (frame["_end"] - frame["_start"]) * frame["_index"] / denominator
    left = frame["_log_price"] - baseline
    right = left.groupby([frame["ticker"], frame["month"]], observed=True).shift(-1)
    left_abs, right_abs = left.abs(), right.abs()
    same_side = left.eq(0) | right.eq(0) | left.mul(right).gt(0)
    trapezoid = (left_abs + right_abs) / 2
    split = (left_abs.pow(2) + right_abs.pow(2)) / (2 * (left_abs + right_abs))
    frame["_interval_area"] = trapezoid.where(same_side, split)
    summary = grouped.agg(
        observation_count=("adjusted_close_usd", "size"),
        first_date=("date", "first"), last_date=("date", "last"),
        first_price=("adjusted_close_usd", "first"),
        last_price=("adjusted_close_usd", "last"),
        raw_area=("_interval_area", "sum"),
    ).reset_index()
    summary.loc[summary["observation_count"].lt(2), "raw_area"] = math.nan
    return summary

def _load_prices(root: Path) -> pd.DataFrame:
    files=sorted((root/"daily_prices").glob("year=*/*.parquet"))
    if not files: raise FileNotFoundError(f"No prices found under {root}")
    return pd.concat([pd.read_parquet(p,columns=["date","ticker","adjusted_close_usd"])
                      for p in files],ignore_index=True)

def _load_monthly_summaries(root: Path) -> pd.DataFrame:
    """Summarize one year at a time to keep peak memory bounded."""
    partitions = sorted((root / "daily_prices").glob("year=*"))
    if not partitions:
        raise FileNotFoundError(f"No price partitions found under {root}")
    summaries = []
    for partition in partitions:
        files = sorted(partition.glob("*.parquet"))
        yearly = pd.concat([
            pd.read_parquet(path, columns=["date", "ticker", "adjusted_close_usd"])
            for path in files
        ], ignore_index=True)
        summaries.append(build_monthly_summary(yearly))
    return pd.concat(summaries, ignore_index=True)


def _ranking(window: pd.DataFrame, constituents: pd.DataFrame) -> pd.DataFrame:
    ordered = window.sort_values(["ticker", "month"])
    stats = ordered.groupby("ticker", sort=False, observed=True).agg(
        month_count=("month", "nunique"),
        minimum_observations=("observation_count", "min"),
        raw_area=("raw_area", "sum"),
        start_price=("first_price", "first"),
        end_price=("last_price", "last"),
    )
    stats = stats.loc[stats["month_count"].eq(6) & stats["minimum_observations"].ge(2)].copy()
    if stats.empty:
        return pd.DataFrame(columns=RANKING_COLUMNS)
    stats["area"] = stats["raw_area"]
    stats["six_month_return"] = stats["end_price"] / stats["start_price"] - 1
    metadata = constituents[["ticker", "company_name"]].rename(columns={"ticker": "native_ticker"})
    stats = stats.join(metadata, how="left")
    stats["native_ticker"] = stats["native_ticker"].fillna(stats.index.to_series())
    stats["company_name"] = stats["company_name"].fillna(stats.index.to_series())
    out = stats.reset_index().rename(columns={"ticker": "yahoo_ticker", "native_ticker": "ticker"})
    out = out[["ticker", "yahoo_ticker", "company_name", "area", "start_price", "end_price", "six_month_return"]]
    out = out.sort_values(["area", "ticker"], kind="stable").reset_index(drop=True)
    out.insert(0, "rank", range(1, len(out) + 1))
    return out

def _market_prices(start: pd.Timestamp,end: pd.Timestamp,cache: Path,refresh: bool)->pd.DataFrame:
    cache.mkdir(parents=True,exist_ok=True); path=cache/"rua_gspc_daily.parquet"
    if path.exists() and not refresh: frame=pd.read_parquet(path)
    else:
        data=yf.download(["^RUA","^GSPC"],start=start.date().isoformat(),
            end=(end+pd.offsets.Day(1)).date().isoformat(),auto_adjust=False,progress=False)
        frame=data["Close"].rename(columns={"^RUA":"russell_3000","^GSPC":"sp500"}).reset_index().rename(columns={"Date":"date"})
        frame.to_parquet(path,index=False)
    frame["date"]=pd.to_datetime(frame["date"]).dt.tz_localize(None); return frame.sort_values("date")

def _betas(tickers:set[str],cache:Path,refresh:bool)->dict[str,float|None]:
    cache.mkdir(parents=True,exist_ok=True); path=cache/"yahoo_beta_snapshot.json"; old={}
    if path.exists() and not refresh: old=json.loads(path.read_text()).get("betas",{})
    def fetch(t):
        try:
            v=float(yf.Ticker(t).info.get("beta")); return t,v if math.isfinite(v) else None
        except Exception:return t,None
    missing=sorted(tickers if refresh else tickers-set(old))
    with ThreadPoolExecutor(max_workers=12) as pool:
        for t,v in pool.map(fetch,missing): old[t]=v
    path.write_text(json.dumps({"created_at_utc":pd.Timestamp.now(tz="UTC").isoformat(),"betas":old},indent=2))
    return {t:old.get(t) for t in tickers}

def _index_return(frame:pd.DataFrame,column:str,start:pd.Timestamp,end:pd.Timestamp,final:pd.Timestamp)->float:
    a=frame.loc[frame.date.ge(start)&frame[column].notna()]
    if a.empty:return math.nan
    b=(frame.loc[frame.date.ge(end)&frame[column].notna()] if end<=final
       else frame.loc[frame.date.le(final)&frame[column].notna()])
    if b.empty:return math.nan
    return float((b.iloc[0] if end<=final else b.iloc[-1])[column]/a.iloc[0][column]-1)

def backtest_area_momentum(data_root:Path=DEFAULT_DATA_ROOT,output_dir:Path|None=None,
    market_prices:pd.DataFrame|None=None,beta_values:Mapping[str,float|None]|None=None,
    refresh_market_data:bool=False,local_prices:pd.DataFrame|None=None,
    start_rebalance:str|pd.Timestamp|None=None)->BacktestResult:
    root=Path(data_root); report=json.loads((root/"validation_report.json").read_text())
    dataset_end=pd.Timestamp(report["maximum_date"]); last=dataset_end.to_period("M").start_time
    first=(pd.Timestamp(start_rebalance).to_period("M").start_time if start_rebalance is not None else last-pd.DateOffset(months=294)); dates=pd.date_range(first,last,freq="MS")
    daily=local_prices.copy() if local_prices is not None else None
    summary=(build_monthly_summary(daily) if daily is not None else _load_monthly_summaries(root)); by_month={m:g for m,g in summary.groupby("month",sort=False)}
    constituents=pd.read_csv(root/"current_constituents.csv").set_index("yahoo_ticker")
    selections={}
    for date in dates:
        periods=pd.period_range(date-pd.DateOffset(months=6),periods=6,freq="M")
        parts=[by_month[p] for p in periods if p in by_month]
        rank=_ranking(pd.concat(parts,ignore_index=True),constituents) if parts else pd.DataFrame(columns=RANKING_COLUMNS)
        selections[date]=select_low_area_high_return(rank)
    selected=set().union(*(set(x.yahoo_ticker) for x in selections.values()))
    cache=(Path(output_dir) if output_dir else DEFAULT_OUTPUT_DIR)/"cache"
    betas=dict(beta_values) if beta_values is not None else _betas(selected,cache,refresh_market_data)
    market=_market_prices(first,dataset_end,cache,refresh_market_data) if market_prices is None else market_prices.copy()
    market["date"]=pd.to_datetime(market["date"]).dt.tz_localize(None)
    holdings=[]; months=[]; diagnostics=[]
    for i,date in enumerate(dates):
        nxt=date+pd.DateOffset(months=1); cf=by_month.get(date.to_period("M")); nf=by_month.get(nxt.to_period("M"))
        current=cf.set_index("ticker") if cf is not None else pd.DataFrame(); following=nf.set_index("ticker") if nf is not None else pd.DataFrame()
        chosen=selections[date]; weight=1/len(chosen) if len(chosen) else 0; returns=[]; month_betas=[]
        for h in chosen.itertuples(index=False):
            ticker=h.yahoo_ticker
            if ticker not in current.index: entry=exit_price=math.nan; ret=0.; status="cash: missing entry"
            else:
                entry=float(current.loc[ticker,"first_price"])
                if i==len(dates)-1: exit_price=float(current.loc[ticker,"last_price"]); status="final partial month"
                elif ticker in following.index: exit_price=float(following.loc[ticker,"first_price"]); status="complete"
                else: exit_price=float(current.loc[ticker,"last_price"]); status="used last quote"
                ret=exit_price/entry-1
            beta=betas.get(ticker)
            if beta is not None and math.isfinite(float(beta)):month_betas.append(float(beta))
            returns.append(ret); holdings.append({"rebalance_date":date,"ticker":h.ticker,"yahoo_ticker":ticker,
                "weight":weight,"area":h.area,"signal_return":h.six_month_return,"entry_price":entry,
                "exit_price":exit_price,"holding_return":ret,"yahoo_beta":beta,"status":status})
        months.append({"rebalance_date":date,"holding_end":min(nxt,dataset_end),"selected_count":len(chosen),
            "strategy_return":sum(r*weight for r in returns),
            "russell_3000_return":_index_return(market,"russell_3000",date,nxt,dataset_end),
            "sp500_return":_index_return(market,"sp500",date,nxt,dataset_end),
            "average_yahoo_beta":sum(month_betas)/len(month_betas) if month_betas else math.nan,
            "beta_count":len(month_betas)})
        diagnostics.append({"rebalance_date":date,"selected_count":len(chosen),"missing_beta_count":len(chosen)-len(month_betas)})
    monthly=pd.DataFrame(months); monthly["strategy_growth"]=(1+monthly.strategy_return).cumprod(); monthly["russell_3000_growth"]=(1+monthly.russell_3000_return).cumprod()
    aligned=monthly[["russell_3000_return","sp500_return"]].dropna(); rb=float(aligned.iloc[:,0].cov(aligned.iloc[:,1])/aligned.iloc[:,1].var())
    beta_months=monthly.average_yahoo_beta.dropna(); strategy_beta=float(beta_months.mean()) if len(beta_months) else math.nan; fig,ax=plt.subplots(figsize=(12,7))
    ax.plot(monthly.holding_end,monthly.strategy_growth,label="Log-area/momentum strategy"); ax.plot(monthly.holding_end,monthly.russell_3000_growth,label="Russell 3000 (^RUA)")
    ax.set(title="Growth of $1: Log-Area/Momentum vs Russell 3000",xlabel="Date",ylabel="Portfolio value"); ax.grid(alpha=.25); ax.legend(); beta_label=(f"Average Yahoo beta (strategy): {strategy_beta:.3f}" if math.isfinite(strategy_beta) else "Average Yahoo beta (strategy): N/A") + f"\nRussell 3000 beta vs S&P 500: {rb:.3f}"; ax.text(0.02,0.97,beta_label,transform=ax.transAxes,va="top",ha="left",bbox={"boxstyle":"round","facecolor":"white","alpha":0.85}); fig.tight_layout()
    return BacktestResult(strategy_beta,int(monthly.beta_count.sum()),len(holdings),rb,float(monthly.strategy_growth.iloc[-1]-1),float(monthly.russell_3000_growth.iloc[-1]-1),monthly,pd.DataFrame(holdings),pd.DataFrame(diagnostics),fig)

def write_backtest_result(result:BacktestResult,output_dir:Path)->None:
    output_dir.mkdir(parents=True,exist_ok=True)
    summary={"area_y_axis":"natural log of adjusted price","average_yahoo_beta":result.average_yahoo_beta,"yahoo_beta_observation_count":result.yahoo_beta_observation_count,"yahoo_beta_holding_count":result.yahoo_beta_holding_count,"russell_3000_beta":result.russell_3000_beta,"strategy_total_return":result.strategy_total_return,"russell_3000_price_index_return":result.russell_3000_total_return,"survivorship_bias_warning":"Uses the November 2026 universe throughout history.","beta_warning":"Yahoo constituent betas are current, not historical."}
    (output_dir/"summary.json").write_text(json.dumps(summary,indent=2)); result.monthly.to_csv(output_dir/"monthly_performance.csv",index=False); result.holdings.to_csv(output_dir/"monthly_holdings.csv",index=False); result.diagnostics.to_csv(output_dir/"diagnostics.csv",index=False); result.figure.savefig(output_dir/"growth_comparison.png",dpi=160)

def main(argv:Sequence[str]|None=None)->None:
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--data-root",type=Path,default=DEFAULT_DATA_ROOT); p.add_argument("--output-dir",type=Path,default=DEFAULT_OUTPUT_DIR); p.add_argument("--refresh-market-data",action="store_true"); p.add_argument("--start-date"); a=p.parse_args(argv)
    result=backtest_area_momentum(a.data_root,a.output_dir,refresh_market_data=a.refresh_market_data,start_rebalance=a.start_date); write_backtest_result(result,a.output_dir)
    print(json.dumps({"average_yahoo_beta":result.average_yahoo_beta,"russell_3000_beta":result.russell_3000_beta,"strategy_total_return":result.strategy_total_return,"russell_3000_price_index_return":result.russell_3000_total_return},indent=2))
if __name__=="__main__":main()

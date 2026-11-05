"""Tests for the monthly area/momentum backtest."""
import json
import math
import tempfile
import unittest
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from backtest_area_momentum import (
    backtest_area_momentum, build_monthly_summary, write_backtest_result,
)
from six_month_price_area import calculate_six_month_area

class BacktestTests(unittest.TestCase):
    def test_precomputed_area_matches_original(self):
        dates=[]; prices=[]
        for month in pd.period_range("2026-01",periods=6,freq="M"):
            dates += [month.start_time+pd.offsets.Day(1),month.start_time+pd.offsets.Day(14),month.end_time.normalize()]
            prices += [10.,20.,10.]
        frame=pd.DataFrame({"date":dates,"ticker":"AAA","adjusted_close_usd":prices})
        summary=build_monthly_summary(frame)
        self.assertAlmostEqual(summary.raw_area.sum(),calculate_six_month_area(frame))

    def test_full_backtest_and_files(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)/"data"; out=Path(td)/"out"; root.mkdir()
            tickers=[f"T{i:02d}" for i in range(25)]
            pd.DataFrame({"ticker":tickers,"yahoo_ticker":tickers,
                "company_name":[f"Firm {i}" for i in range(25)]}).to_csv(root/"current_constituents.csv",index=False)
            (root/"validation_report.json").write_text(json.dumps({"minimum_date":"2001-11-05","maximum_date":"2026-11-05"}))
            rows=[]; periods=pd.period_range("2001-11", "2026-11",freq="M")
            for mi,month in enumerate(periods):
                days=[1,3,5] if month==periods[-1] else [1,14,28]
                for ti,ticker in enumerate(tickers):
                    base=20+ti
                    trend=1+mi*(0.0005+ti*0.00001)
                    values=[base*trend,base*trend*(1+0.01*(ti%5)),base*trend*(1+0.002*(ti%7))]
                    for day,value in zip(days,values):
                        rows.append({"date":month.start_time+pd.offsets.Day(day-1),
                            "ticker":ticker,"adjusted_close_usd":value})
            local=pd.DataFrame(rows)
            boundaries=pd.date_range("2002-05-01","2026-11-01",freq="MS")
            market=pd.DataFrame({"date":list(boundaries)+[pd.Timestamp("2026-11-05")],
                "russell_3000":[100*(1.01**i) for i in range(len(boundaries)+1)],
                "sp500":[200*(1.01**i) for i in range(len(boundaries)+1)]})
            result=backtest_area_momentum(root,out,market_prices=market,
                beta_values={ticker:1.0 for ticker in tickers},local_prices=local)
            self.assertEqual(len(result.monthly),295)
            self.assertEqual(result.monthly.rebalance_date.iloc[0],pd.Timestamp("2002-05-01"))
            self.assertEqual(result.monthly.rebalance_date.iloc[-1],pd.Timestamp("2026-11-01"))
            self.assertTrue(result.monthly.selected_count.eq(1).all())
            self.assertTrue(result.holdings.weight.eq(1.0).all())
            self.assertAlmostEqual(result.average_yahoo_beta,1.0)
            self.assertAlmostEqual(result.russell_3000_beta,1.0)
            self.assertIn("Average Yahoo beta (strategy): 1.000", result.figure.axes[0].texts[0].get_text())
            self.assertIn("Russell 3000 beta vs S&P 500: 1.000", result.figure.axes[0].texts[0].get_text())
            self.assertTrue(math.isfinite(result.strategy_total_return))
            self.assertTrue(math.isfinite(result.russell_3000_total_return))
            self.assertEqual(result.holdings.status.iloc[-1],"final partial month")
            write_backtest_result(result,out)
            for name in ["summary.json","monthly_performance.csv","monthly_holdings.csv","diagnostics.csv","growth_comparison.png"]:
                self.assertTrue((out/name).exists(),name)
            plt.close(result.figure)

if __name__=="__main__": unittest.main()

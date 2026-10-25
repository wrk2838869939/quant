#!/usr/bin/env python3
"""汇总经典策略工程回测产物，生成中文报告与逐日资金/换手诊断。"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    run = args.run
    manifest = json.loads((run / "manifest.json").read_text())
    if manifest["status"] not in ["complete", "completed_with_failures"]:
        raise ValueError("只分析已完成的运行")
    summary = pd.read_csv(run / "summary.csv")
    checks = pd.read_csv(run / "process_checks.csv")
    cfg = manifest["config"]
    diagnostics, event_counts, independent_checks = [], [], []
    market = pd.read_parquet(run / "inputs/market.parquet", columns=["date", "ts_code", "close"])
    tolerance = cfg["audit"]["money_atol"]
    for row in summary.itertuples():
        path = run / row.strategy
        daily = pd.read_parquet(path / "daily.parquet")
        trades = pd.read_parquet(path / "trades.parquet")
        holdings = pd.read_parquet(path / "holdings.parquet")
        events = pd.read_parquet(path / "events.parquet")
        trade_gross = trades.groupby("date").gross.sum().reindex(daily.index, fill_value=0.) if len(trades) else daily.cash * 0
        fees = trades.groupby("date").fee.sum().reindex(daily.index, fill_value=0.) if len(trades) else daily.cash * 0
        # 使用外部冻结收盘价重新核验持仓估值，独立重算回撤与逐日收益。
        valuation_error = 0.
        if len(holdings):
            marked = holdings.merge(market, on=["date", "ts_code"], how="left", validate="one_to_one")
            known = marked.close.notna()
            valuation_error = float((marked.loc[known, "shares"] * marked.loc[known, "close"] - marked.loc[known, "value"]).abs().max())
            assert valuation_error <= tolerance
        curve = daily.equity.to_numpy()
        expected_return = curve / np.r_[row.capital, curve[:-1]] - 1
        expected_mdd = float(np.min(curve / np.maximum.accumulate(np.r_[row.capital, curve])[1:] - 1))
        np.testing.assert_allclose(daily.ret, expected_return, atol=1e-12, rtol=1e-12)
        assert abs(expected_mdd - row.mdd) <= 1e-12
        assert daily.n_hold.max() <= cfg["top_n"]
        if len(trades):
            grid = {date: i for i, date in enumerate(daily.index)}
            for buy in trades.loc[trades.side.eq("buy")].itertuples():
                assert grid[buy.date] == grid[buy.signal_date] + 1
        independent_checks.append(dict(strategy=row.strategy, close_valuation_max_error=valuation_error,
                                       recomputed_mdd=expected_mdd, next_session_buys=True,
                                       returns_recomputed=True, position_count_pass=True))
        previous_equity = daily.equity.shift().fillna(row.capital)
        flow = daily.assign(traded_notional=trade_gross, fees=fees,
                            gross_turnover=trade_gross / previous_equity,
                            cash_fraction=daily.cash / daily.equity)
        flow.to_csv(path / "daily_diagnostics.csv")
        concentration = (holdings.value / holdings.date.map(daily.equity) if len(holdings) else pd.Series([0.]))
        diagnostics.append(dict(strategy=row.strategy, min_cash=float(daily.cash.min()),
                                max_positions=int(daily.n_hold.max()),
                                max_single_stock_weight=float(concentration.max()),
                                max_abs_daily_return=float(daily.ret.abs().max()),
                                annual_gross_turnover=float(flow.gross_turnover.sum() / len(daily) * cfg["annual_days"]),
                                cash_only_days=int(daily.n_hold.eq(0).sum()),
                                buy_trades=int(trades.side.eq("buy").sum()) if len(trades) else 0,
                                sell_trades=int(trades.side.eq("sell").sum()) if len(trades) else 0))
        if len(events):
            if "reason" not in events:
                events["reason"] = ""
            for (kind, reason), size in events.fillna({"reason": ""}).groupby(["kind", "reason"]).size().items():
                event_counts.append(dict(strategy=row.strategy, kind=kind, reason=reason, count=int(size)))
    pd.DataFrame(independent_checks).to_csv(run / "independent_valuation_checks.csv", index=False)
    diagnostics = pd.DataFrame(diagnostics)
    diagnostics.to_csv(run / "diagnostics.csv", index=False)
    pd.DataFrame(event_counts).to_csv(run / "event_counts.csv", index=False)
    nav = pd.read_csv(run / "nav_comparison.csv", index_col=0)
    benchmark = nav.CSI300_price_reference
    bm_total = benchmark.iloc[-1] - 1
    bm_mdd = (benchmark / np.maximum.accumulate(np.r_[1., benchmark])[1:] - 1).min()
    text = ["# 经典策略端到端工程验收", "",
            "本次验证冻结数据、经典信号、10万元现金股数账本和结果导出的整条链路。所有参数预先固定，未按收益寻优。",
            "**本报告属于工程测试，不是新的样本外策略有效性证明。**", "",
            "## 输入与模拟口径", "",
            f"- 冻结数据：`{manifest['data_snapshot']['snapshot_id']}`；内容标识 `{manifest['data_snapshot']['content_id']}`。",
            f"- 期间：{cfg['start']}～{cfg['end']}，实际 {manifest['preparation']['dates']} 个交易日；之前历史仅用于指标预热。",
            f"- 用 {cfg['formation_start']}～{cfg['formation_end']} 的成交额中位数选定100只沪深主板股票；不按当前在市名单过滤。",
            f"- 公司行动/价格预核验隔离 {len(manifest['preparation']['quarantine_codes'])} 只，所有策略共享剩余 {manifest['preparation']['kept_codes']} 只。隔离依赖全期数据质量，存在事后样本筛选偏差，不能外推全市场收益。",
            "- 初始资金10万元；最多5只；日频或月频调仓。t日收盘决策，下一交易日开盘模拟成交，现金预算内整手买入，T+1、官方涨跌停限制，现金不计利息。",
            "- 旧持仓仍在前5名时继续持有；卖出后把可用现金按空位分配，不是每日恢复等权。高价股可能因为买不起一手被拒单，不用碎股补齐。",
            "- 买入有效费用0.10%、卖出0.20%，最低5元，另加每侧0.05%滑点成本。成本是固定研究假设，未逐年模拟真实税费制度。",
            "- 分红现金/应收款、送转股及可卖日期独立记账；现金分红统一保守扣减20%。期末按收盘估值，不虚构清仓。",
            "- 回撤30%仅作结果验收；没有为了达标截断净值或追加止损。",
            "- ST用冻结改名史，生效/公告日期取较晚值并在下一交易日执行状态变化；缺公告日期时假设生效日收盘已可知，无时分级发布认证。",
            "- 开盘撮合、卖出款当日再买、日成交量是否为零判断停牌均为日线模拟假设，不是逐笔或集合竞价队列复现。",
            "", "## 策略定义", "",
            "- 动量：20日后复权收益>0，按收益降序选前5名。",
            "- MACD：标准12/26/9，DIF>DEA且DIF>0，按(DIF−DEA)/后复权收盘价排序。这里测试趋势状态，不是仅金叉当天入场。",
            "- 均线：MA20>MA60且价格>MA20，按MA20/MA60−1排序。",
            "- 随机基准：同股票池、同费用、同资金约束，固定随机种子；仅一个种子，不构成统计显著性检验。",
            "- 现金基准：不交易，净值必须严格保持1。月频在月末决策，窗口首日先初始化一次。",
            "", "## 结果", "",
            "| 策略 | 累计收益 | 年化收益 | 最大回撤 | 交易笔数 | 费用含滑点/元 | 平均投入比例 | 回撤≤30% |",
            "|---|---:|---:|---:|---:|---:|---:|:---:|"]
    for row in summary.itertuples():
        text.append(f"| {row.strategy} | {row.total:+.2%} | {row.ann:+.2%} | {row.mdd:.2%} | {row.n_trades} | {row.total_fees:,.2f} | {row.mean_invested:.1%} | {'是' if row.risk_pass else '否'} |")
    text += ["", f"同期沪深300价格指数参考：累计 {bm_total:+.2%}，最大回撤 {bm_mdd:.2%}。该指数不含分红、无交易费用，与可投资组合不是完全同口径。",
             "", "## 过程核验", "",
             f"- 已完成 {len(summary)} 个策略/频率组合，失败 {len(manifest.get('failed_strategies', []))} 个。",
             f"- 现金流水重建最大误差：{checks.cash_max_error.max():.3g} 元；股数重建最大误差：{checks.shares_max_error.max():.3g} 股。",
             f"- 逐笔现金误差：{checks.trade_cash_max_error.max():.3g} 元；费用公式最大误差：{checks.fees_max_error.max():.3g} 元。",
             "- 独立核验：净资产=现金+股票市值+分红应收款；持仓市值汇总一致；无负现金；T+1/红股锁定、整手买入、原始开盘成交价和官方涨跌停约束一致。",
             "- 另以归档行情的原始收盘价重算当日有报价持仓价值，独立重算日收益/最大回撤，验证买入恰在信号后下一交易日、持仓不超过5只。停牌日依赖账本中的上次估值/公司行动调整，未用虚构成交价填补。",
             "- 动量、均线、MACD与直接从冻结历史行情计算的pandas公式逐股对账。",
             "- 用2021年末截断窗口重跑信号和账本，历史净值必须与长窗口重叠区间一致；该检查不消除全期质量隔离所带来的样本选择偏差。",
             f"- 最大单日绝对组合收益：{checks.max_abs_daily_return.max():.2%}；超过35%的硬检查未触发。",
             f"- 全部持仓数量最大值：{diagnostics.max_positions.max()}；单股权重最大值：{diagnostics.max_single_stock_weight.max():.2%}，留仓上涨后不会自动恢复等权。",
             f"- 引擎执行资料充分标记：{int(summary.execution_data_sufficient.sum())}/{len(summary)}；若出现长期停牌估值警告，须与流水一起评估，不能只看收益。",
             "", "## 产物与复现", "",
             "- `summary.csv`、`yearly.csv`：总结果及逐年结果；`nav_drawdown.png`、`nav_comparison.csv`：净值与回撤。",
             "- 每个策略目录：`signals.parquet`、`targets.csv`、`daily.csv`、`trades.csv`、`holdings.csv`、`events.csv`、`daily_diagnostics.csv`、`audit.json`。",
             "- `inputs/`：行情、股本主档、状态区间、官方限价、原始/规范化公司行动、特征、资格掩码、隔离清单、源数据及准备产物哈希。",
             "- `sources/`：本次代码与配置快照；`manifest.json`：数据版本、代码版本和运行状态。",
             "", "```bash",
             "python datahub/scripts/frozen_research.py run --release datahub/releases/selection_v1_20260920 alpha_lab/scripts/run_classic_strategy_smoke.py",
             f"python alpha_lab/scripts/report_classic_strategy_smoke.py --run {run}",
             "```", "",
             "重新准备输入会校验并记录活动执行缓存；与本次精确比较时先对照 inputs/preparation.json 的源哈希。",
             "后续用 --prepared 指向已通过的运行目录，即可校验并直接复用执行输入，不再读取活动执行缓存：", "", "```bash",
             f"python datahub/scripts/frozen_research.py run --release datahub/releases/selection_v1_20260920 alpha_lab/scripts/run_classic_strategy_smoke.py --prepared {run}",
             "```"]
    (run / "REPORT.md").write_text("\n".join(text) + "\n", encoding="utf-8")
    print(summary[["strategy", "total", "ann", "mdd", "n_trades", "total_fees", "risk_pass"]].to_string(index=False))
    print(f"报告：{run / 'REPORT.md'}")


if __name__ == "__main__":
    main()

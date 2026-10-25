#!/usr/bin/env python3
"""学习失败后的基线逐股归因、风格描述与有限压力测试。"""
from pathlib import Path
from datetime import datetime, timezone
import json
import sys
import numpy as np
import pandas as pd
import yaml
ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
import run_policy_balance as B
from baseline_attribution import attribute_account
from run_classic_strategy_smoke import digest, save_json
from run_rule_strategy_search import write_result


def main():
    config = ROOT / 'configs/baseline_attribution_v1.yaml'
    cfg = yaml.safe_load(config.read_text())
    run = ROOT / 'outputs/baseline_attribution' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    run.mkdir(parents=True)
    print('RUN=' + str(run), flush=True)
    _, _, account = B.setup(ROOT / 'configs/policy_balance_v1.yaml', run)
    source = PROJECT / cfg['source_run']
    parent = json.loads((source / 'manifest.json').read_text())
    assert parent['status'] == 'complete'
    assert all(digest(PROJECT / p) == h for p, h in parent['source_hashes'].items())
    hashes = json.loads((source / 'artifact_hashes.json').read_text())
    balance = pd.read_csv(PROJECT / cfg['balance_run'] / 'outer_summary.csv').set_index('strategy')
    assert balance.loc['adaptive', 'total'] < balance.loc['baseline', 'total']
    sources = [Path(__file__), config, ROOT / 'src/baseline_attribution.py', ROOT / 'tests/test_baseline_attribution.py']
    manifest = dict(status='running', study=cfg['study'], dataset=account['dataset'],
                    source_hashes={**parent['source_hashes'], **{str(p.relative_to(PROJECT)): digest(p) for p in sources}},
                    source_manifest_sha256=digest(source / 'manifest.json'), unbiased_oos=False)
    save_json(run / 'manifest.json', manifest)
    capital = B.G['portfolio']['capital']
    wide, eligible = B.G['wide'], B.G['eligible']
    styles = dict(turnover=wide['turnover_rate'].rolling(cfg['style_window']).mean(),
                  volatility=wide['hfq_close'].pct_change(fill_method=None).rolling(cfg['style_window']).std(),
                  market_cap=wide['circ_mv_yuan'], earnings_yield=1 / wide['pe_ttm'].where(wide['pe_ttm'] > 0))
    styles = {k: v.where(eligible.reindex_like(v).fillna(False).astype(bool)).rank(axis=1, pct=True) for k, v in styles.items()}
    summaries, years, stresses, stock_tables, verified = [], [], [], {}, []
    for name in cfg['baselines']:
        old, folder = source / 'outer' / name, run / name
        folder.mkdir()
        for file in [*[f'{t}.parquet' for t in B.R.original.TABLES], 'signals.parquet', 'top_n_schedule.parquet']:
            relative = str((old / file).relative_to(source))
            assert digest(old / file) == hashes[relative]
        signals = pd.read_parquet(old / 'signals.parquet')
        schedule = pd.read_parquet(old / 'top_n_schedule.parquet').iloc[:, 0]
        result = B.R.original.execute(signals, schedule, cfg['start'], cfg['end'])
        for table in B.R.original.TABLES:
            pd.testing.assert_frame_equal(getattr(result, table), pd.read_parquet(old / f'{table}.parquet'), check_exact=True)
        B.R.original.audit(result)
        write_result(folder / 'original', result, signals)
        stock, pnl, values, error = attribute_account(result.daily, result.trades, result.events, result.holdings, capital)
        stock.to_csv(folder / 'stock_attribution.csv')
        pnl.to_parquet(folder / 'daily_stock_pnl.parquet')
        stock_tables[name] = stock
        total = float(stock.net_pnl.sum())
        weights = values.div(values.sum(axis=1).replace(0, np.nan), axis=0)
        exposures = {}
        for key, rank in styles.items():
            rank = rank.reindex(index=weights.index, columns=weights.columns)
            exposures[key + '_percentile'] = float((weights * rank).sum(axis=1).div(weights.where(rank.notna()).sum(axis=1).replace(0, np.nan)).mean())
        summaries.append(dict(strategy=name, total=result.summary['total'], mdd=result.summary['mdd'],
                              net_pnl=total, price_pnl=stock.price_pnl.sum(), dividends=stock.dividends.sum(),
                              fees=stock.fees.sum(), slippage_embedded=result.trades.slippage_cost.sum(),
                              best_stock=stock.index[0], top1_net_pnl=stock.net_pnl.iloc[0], top3_net_pnl=stock.net_pnl.head(3).sum(),
                              active_stocks=int((stock.net_pnl != 0).sum()), profitable_stocks=int((stock.net_pnl > 0).sum()),
                              max_reconciliation_error=error, **exposures))
        previous = capital
        for year, group in result.daily.groupby(result.daily.index.year):
            net = float(group.equity.iloc[-1] - previous)
            years.append(dict(strategy=name, year=int(year), net_pnl=net, total=float(group.equity.iloc[-1] / previous - 1)))
            previous = group.equity.iloc[-1]
        altered = signals.copy()
        altered.loc[:, stock.index[0]] = np.nan
        cases = [('remove_top_contributor', altered, schedule, cfg['start']),
                 ('top_n_5', signals, pd.Series(cfg['stress_top_n'], index=schedule.index), cfg['start']),
                 ('comparable_2022_2023', signals.loc[cfg['comparison_start']:], schedule.loc[cfg['comparison_start']:], cfg['comparison_start'])]
        for kind, score, numbers, start in cases:
            counter = B.R.original.execute(score, numbers, start, cfg['end'])
            B.R.original.audit(counter)
            write_result(folder / kind, counter, score)
            stresses.append(dict(strategy=name, case=kind, total=counter.summary['total'], mdd=counter.summary['mdd'], fees=counter.summary['total_fees']))
        verified.append(dict(strategy=name, exact_four_tables=True, attribution_error=error))
        print(name + ' ' + json.dumps(summaries[-1]), flush=True)
    summary = pd.DataFrame(summaries)
    summary.to_csv(run / 'summary.csv', index=False)
    pd.DataFrame(years).to_csv(run / 'annual.csv', index=False)
    pd.DataFrame(stresses).to_csv(run / 'stress.csv', index=False)
    save_json(run / 'verification.json', verified)
    lines = ['# 基线收益来源与外推边界', '',
             f'运行：`{run.relative_to(PROJECT)}`。重新用当前冻结数据回放四条登记基线，逐日账户、交易、事件、持仓与原归档完全一致。初始资金 10 万，期间 2021–2023。', '',
             '本轮联合学习连续外层收益 -3.86%、回撤 17.36%，固定基线收益 +31.81%、回撤 13.31%，因此停止追加学习网格，先回到基线机制分析。这里四条登记基线按原锁定年度参数运行，与联合实验中固定 2021 参数的基线不同。', '',
             '## 可精确加总的收益', '',
             '| 基线 | 净收益 | 回撤 | 价格及成交盈亏（元） | 应计分红（元） | 手续费（元） | 最大贡献股 | 该股净利润/组合净利润 |',
             '|---|---:|---:|---:|---:|---:|---|---:|']
    for row in summaries:
        lines.append(f'| {row["strategy"]} | {row["total"]:+.2%} | {abs(row["mdd"]):.2%} | {row["price_pnl"]:,.0f} | {row["dividends"]:,.0f} | {row["fees"]:,.0f} | {row["best_stock"]} | {row["top1_net_pnl"]/row["net_pnl"]:.1%} |')
    lines += ['', '净利润 = 价格及成交盈亏 + 应计分红 − 手续费。滑点已经嵌入成交价，不能再次扣减。价格项包含价格变化、成交时点和公司行动估值变化，不直接等于选股 alpha。个股贡献比可超过 100%，表示其他股票净拖累；分红也不能脱离除息价格变化解释为额外免费收益。', '',
              '## 时间与持仓集中度', '', '| 基线 | 2021 年 | 2022 年 | 2023 年 | 盈利股数 / 有盈亏股票数 |', '|---|---:|---:|---:|---:|']
    for row in summaries:
        returns = [next(r['total'] for r in years if r['strategy'] == row['strategy'] and r['year'] == year) for year in [2021, 2022, 2023]]
        lines.append(f'| {row["strategy"]} | {returns[0]:+.2%} | {returns[1]:+.2%} | {returns[2]:+.2%} | {row["profitable_stocks"]}/{row["active_stocks"]} |')
    lines += ['', '## 压力测试', '', '| 基线 | 原策略 | 剔除最大贡献股 | 统一持有 5 股 |', '|---|---:|---:|---:|']
    for row in summaries:
        case = {r['case']: r for r in stresses if r['strategy'] == row['strategy']}
        lines.append(f'| {row["strategy"]} | {row["total"]:+.2%} | {case["remove_top_contributor"]["total"]:+.2%} / 回撤 {abs(case["remove_top_contributor"]["mdd"]):.2%} | {case["top_n_5"]["total"]:+.2%} / 回撤 {abs(case["top_n_5"]["mdd"]):.2%} |')
    lines += ['', '剔除对象由全期贡献事后确定，只是依赖度压力测试，不是可部署的样本外选股规则。统一 5 股也是新提出的描述性消融，本轮不据此换用收益更高的配置。', '',
              '## 风格暴露', '', '| 基线 | 换手分位 | 波动分位 | 流通市值分位 | 正盈利收益率分位 |', '|---|---:|---:|---:|---:|']
    for row in summaries:
        values = [row[k + '_percentile'] for k in ['turnover', 'volatility', 'market_cap', 'earnings_yield']]
        lines.append('| ' + row['strategy'] + ' | ' + ' | '.join(f'{v:.1%}' for v in values) + ' |')
    lines += ['', '分位只相对于冻结 93 股的当日合格截面；表中为实际持仓市值加权后逐日平均，缺失观测重新归一。该表描述暴露，不能识别各风格的因果收益贡献，也不是全 A 风格基准。', '',
              '## 可检验的外推路线', '',
              '1. 先验证收益是否能跨股票复制：完整保留剔除第一贡献股的结果；后续预先划分不重叠股票组，使用同一规则，禁止按组重新挑冠军。',
              '2. 再验证时间稳定性：区分参数预先可得、未再调参的持有段；已观察的 2021–2023 只能用于提出假设，真正外推需新增未观察时段或前向模拟。',
              '3. 把低换手、低波动、盈利收益率视为待区分的风格解释，下一轮用保持市值/波动暴露相近的对照，比较组合净收益和费用差，不能用本表直接宣称选股 alpha。',
              '4. 先研究持仓分散与费用敏感性，保留 10 万、整手约束、相同费用与 30% 回撤验收；如果优势只能由少数个股与特定年份解释，就降低外推预期。', '',
              '本轮没有新增数据、训练模型、修改登记基线或更换生产策略。完整逐股净利润、年度利润、风格暴露与压力测试账户见运行目录。']
    report = ROOT / 'docs/BASELINE_ATTRIBUTION_RESULTS_20260921.md'
    report.write_text('\n'.join(lines) + '\n')
    assert all(digest(PROJECT / p) == h for p, h in manifest['source_hashes'].items())
    manifest.update(status='complete', report=str(report.relative_to(PROJECT)), report_sha256=digest(report),
                    baseline_replays=len(summaries), counterfactual_replays=len(stresses), max_reconciliation_error=max(r['attribution_error'] for r in verified))
    save_json(run / 'manifest.json', manifest)
    save_json(run / 'artifact_hashes.json', {str(p.relative_to(run)): digest(p) for p in run.rglob('*') if p.is_file() and p.name != 'artifact_hashes.json'})
    print('COMPLETE ' + str(report), flush=True)


if __name__ == '__main__':
    main()

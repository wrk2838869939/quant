#!/usr/bin/env python3
"""逐时点板块信号下的个股研究；仅报告价格观察，不冒充成交回测。"""
from pathlib import Path
import argparse
import json
import sys

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'alpha_lab/src'), str(ROOT / 'alpha_lab/scripts')]
from sector_dataset import digest
from report_sector_entry_window import official_limits


def metrics(values):
    """事件等权描述，空样本保持未知。"""
    return dict(mean=values.mean(), median=values.median(), q10=values.quantile(.1),
                positive=values.gt(0).mean() if len(values) else np.nan)


def cluster_interval(frame, calendar, cfg):
    """相邻20交易日内的事件归同一行情簇，避免多股票虚增样本。"""
    if frame.empty:
        return dict(clusters=0, ci_low=np.nan, ci_high=np.nan)
    frame = frame.sort_values('date').copy()
    positions = calendar.get_indexer(frame.date)
    frame['cluster'] = np.cumsum(np.r_[True, np.diff(positions) > cfg['cluster_gap_sessions']])
    sums = frame.groupby('cluster').excess.agg(['sum', 'count'])
    result = dict(clusters=len(sums), ci_low=np.nan, ci_high=np.nan)
    if len(sums) < 5:
        return result
    rng = np.random.default_rng(cfg['bootstrap_seed'])
    draws = rng.integers(0, len(sums), size=(cfg['bootstrap_repetitions'], len(sums)))
    boot = sums['sum'].to_numpy()[draws].sum(axis=1) / sums['count'].to_numpy()[draws].sum(axis=1)
    result['ci_low'], result['ci_high'] = np.quantile(boot, [.025, .975])
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    out = args.run
    cfg = yaml.safe_load((out / 'config.yaml').read_text())
    assert (out / 'stocks_complete.json').exists() and (out / 'event_prefix_audit.json').exists()
    pred = pd.read_parquet(out / 'predictions_BEFORE_EVALUATION.parquet')
    choices = pd.read_parquet(out / 'first_choices_BEFORE_EVALUATION.parquet')
    frame = pd.read_parquet(out / 'stock_decisions_BEFORE_EVALUATION.parquet')
    prices = pd.read_parquet(out / 'observation_hfq_open.parquet')
    raw_open = pd.read_parquet(out / 'observation_open.parquet')
    volume = pd.read_parquet(out / 'observation_vol_share.parquet')
    calendar = prices.index
    costs = cfg['observation']
    cost_ratio = (1 - costs['sell_cost'] - costs['slippage_per_side']) / (1 + costs['buy_cost'] + costs['slippage_per_side'])
    required = choices[['event_id', 'date']].drop_duplicates()
    universe = frame[frame.qualified].merge(required, on=['event_id', 'date'], validate='many_to_one')
    observations = []
    for r in universe.itertuples():
        index = calendar.get_loc(r.date)
        for horizon in costs['horizons']:
            entry_i, exit_i = index + 1, index + 1 + horizon
            complete = exit_i < len(calendar)
            entry, end = calendar[entry_i] if entry_i < len(calendar) else pd.NaT, calendar[exit_i] if complete else pd.NaT
            values = prices[r.ts_code].iloc[entry_i:exit_i + 1]
            vols = volume[r.ts_code].iloc[entry_i:exit_i + 1]
            complete = complete and values.gt(0).all() and vols.gt(0).all() and len(values) == horizon + 1
            ret = values.iloc[-1] / values.iloc[0] * cost_ratio - 1 if complete else np.nan
            observations.append(dict(event_id=r.event_id, date=r.date, ts_code=r.ts_code, horizon=horizon,
                                      entry_date=entry, exit_date=end, complete=bool(complete), price_return=ret))
    obs = pd.DataFrame(observations)
    obs.to_parquet(out / 'price_observations_NOT_ACCOUNT.parquet', index=False)
    baseline = obs.groupby(['event_id', 'date', 'horizon']).agg(base_n=('ts_code', 'size'), base_complete=('complete', 'all'),
                                                                            base_return=('price_return', 'mean')).reset_index()
    baseline.loc[~baseline.base_complete, 'base_return'] = np.nan
    selected = choices[['variant', 'event_id', 'date', 'ts_code']].merge(obs, on=['event_id', 'date', 'ts_code'], validate='many_to_many')
    event = selected.groupby(['variant', 'event_id', 'date', 'horizon']).agg(stocks=('ts_code', 'size'),
                     complete=('complete', 'all'), price_return=('price_return', 'mean')).reset_index()
    event.loc[~event.complete, 'price_return'] = np.nan
    event = event.merge(baseline, on=['event_id', 'date', 'horizon'], validate='many_to_one')
    event['excess'] = event.price_return - event.base_return
    event.to_csv(out / 'event_price_comparison_NOT_ACCOUNT.csv', index=False)
    summaries, annual = [], []
    for (variant, horizon), part in event.groupby(['variant', 'horizon']):
        valid = part[part.complete]
        paired = part[part.complete & part.base_complete]
        summaries.append(dict(variant=variant, horizon=horizon, events=len(part), complete_events=len(valid),
                              incomplete_events=len(part) - len(valid), paired_events=len(paired),
                              stocks=int(part.stocks.sum()), mean_excess=paired.excess.mean(),
                              **metrics(valid.price_return), **cluster_interval(paired, calendar, costs)))
        for year, group in part.groupby(part.date.dt.year):
            annual.append(dict(variant=variant, horizon=horizon, year=year, events=len(group),
                               complete_events=int(group.complete.sum()), mean_excess=group.excess.mean(),
                               **metrics(group.loc[group.complete, 'price_return'])))
    summary = pd.DataFrame(summaries)
    summary.to_csv(out / 'summary_NOT_ACCOUNT.csv', index=False)
    pd.DataFrame(annual).to_csv(out / 'annual_NOT_ACCOUNT.csv', index=False)
    # 执行资料只做覆盖审计：未知限价与冲突均不得宣称可成交。
    limits, conflicts, sources = official_limits(selected.ts_code.unique())
    checks = []
    for r in selected.drop_duplicates(['date', 'ts_code', 'horizon']).itertuples():
        statuses = []
        for side, date in [('buy', r.entry_date), ('sell', r.exit_date)]:
            if pd.isna(date):
                status = 'future_date_not_observed'
            elif not (raw_open.at[date, r.ts_code] > 0 and volume.at[date, r.ts_code] > 0):
                status = 'missing_quote_or_suspended'
            elif (date, r.ts_code) in conflicts:
                status = 'conflicting_official_limits'
            elif (date, r.ts_code) not in limits:
                status = 'missing_official_limits'
            else:
                bound = limits[(date, r.ts_code)]
                if not np.isfinite([bound['up_limit'], bound['down_limit']]).all():
                    status = 'missing_official_limits'
                elif (side == 'buy' and raw_open.at[date, r.ts_code] >= bound['up_limit']) or (side == 'sell' and raw_open.at[date, r.ts_code] <= bound['down_limit']):
                    status = 'opening_at_limit'
                else:
                    status = 'quote_and_limit_only_pass'
            statuses.append(status)
        checks.append(dict(date=r.date, ts_code=r.ts_code, horizon=r.horizon, entry_status=statuses[0], exit_status=statuses[1]))
    execution = selected.merge(pd.DataFrame(checks), on=['date', 'ts_code', 'horizon'], validate='many_to_one')
    execution.groupby(['variant', 'horizon', 'entry_status', 'exit_status']).size().rename('stock_paths').reset_index().to_csv(out / 'execution_coverage.csv', index=False)
    (out / 'execution_limit_sources.json').write_text(json.dumps(sources, indent=2))
    selected[~selected.complete].to_csv(out / 'incomplete_selected_paths.csv', index=False)
    # 条件计数使用所有同事件有效日期；与仅看首个机会的绩效统计分开。
    maxima = frame.groupby('event_id')[['qualified', 'trend', 'price', 'original', 'no_expansion', 'price_trend', 'trend_only', 'price_only']].any()
    funnel = pred[pred.selected].merge(maxima, on='event_id', validate='one_to_one')
    funnel.to_csv(out / 'model_pass_stock_funnel.csv', index=False)
    counts = []
    for variant in cfg['stock']['variants']:
        chosen = choices[choices.variant.eq(variant)]
        counts.append(dict(variant=variant, model_pass_events=int(pred.selected.sum()),
                           candidate_events=chosen.event_id.nunique(), first_date_stocks=len(chosen)))
    counts = pd.DataFrame(counts)
    counts.to_csv(out / 'opportunity_counts.csv', index=False)
    old = pd.read_parquet(ROOT / cfg['parent_run'] / 'event_scores.parquet')
    comparison = pred.merge(old[['event_id', 'chronological_selected']], on='event_id', validate='one_to_one')
    comparison.to_csv(out / 'replaces_previous_chronological.csv', index=False)
    originals = choices[choices.variant.eq('original')].sort_values(['date', 'ts_code'], ascending=[False, True])
    originals.to_csv(out / 'original_rule_historical_candidates.csv', index=False)
    folds = json.loads((out / 'fits.json').read_text())
    selected_years = pred.groupby(pred.date.dt.year).agg(scored=('event_id', 'size'), passed=('selected', 'sum'))
    lines = ['# 模拟历史决策时点：板块通过后的个股筛选', '',
             '本报告替代随机冻结模型历史评分口径。算法按历史时间推进；数据版本尚未达到原始时点可见性认证，不能称为完全重建当时实盘。', '',
             '## 时间边界', '',
             '- 年初训练、全年使用；规则数、学习率、轮数只在此前内层时间验证选择。训练标签必须在年初前20个交易日截止之前结束；内层同样隔离。',
             '- 行情与板块状态只用决策日及之前数据；名称公告必须严格早于决策日。预测和候选先落盘，后读取收益作评价。',
             '- 当前研究家族及候选网格并非当年真实预注册；不能把历史开发结果宣传为新的独立样本外业绩。',
             '- 行业成员仅有生效区间，没有公告时间和历史下载版本；只能按供应商今天记录的历史区间重建。', '',
             '## 板块模型', '', '```', selected_years.to_string(), '```', '',
             f'合计评分{len(pred)}个独立事件，通过{int(pred.selected.sum())}个。旧固定参数时间推进版与本轮选择不同的事件有{int(comparison.selected.ne(comparison.chronological_selected).sum())}个。', '',
             '## 个股候选数量', '', '```', counts.to_string(index=False), '```', '',
             'original=原完整规则；no_expansion=取消前期放量硬门槛；price_trend=价格形态加趋势；trend_only=仅趋势；price_only=仅价格形态；qualified_all=全部合格成员基准；末两项为price_trend内60日强度前5与20日低波前5。', '',
             '每方案每事件只取首次出现候选的日期，候选等权。不是实际资金账户；候选数量不能解释成独立成交笔数。', '',
             '## 20日价格观察', '',
             '次日开盘到20交易日后开盘，固定往返约40bp费用压力测试。事件等权，不将同板块多股当作独立事件。相对值配对该事件同日全部合格成员；只要任一成分路径缺失，该篮子整体不报告收益，缺失数量保留。', '',
             '```', summary[summary.horizon.eq(20)].to_string(index=False), '```', '',
             '区间按相隔20交易日的行情簇重采样；少于5簇不报区间。年度差异见annual_NOT_ACCOUNT.csv。置信区间未校正多个方案比较，不能用于从中挑选“显著赢家”。', '',
             '## 可交易性边界', '',
             '已有涨跌停资料的覆盖见execution_coverage.csv；缺少官方限价不得称可成交。没有整手现金、持仓重叠、真实成交排队、分红到账和停牌退出账本；不报告账户收益或最大回撤。', '',
             '## 最新的原规则历史候选', '', '```', originals[['date', 'sector_code', 'ts_code']].head(12).to_string(index=False), '```', '',
             '这些是采用今天规定的回放流程重建的历史候选，不是当时发出的实盘通知。', '',
             '## 复现与审计文件', '',
             '- config.yaml与sources/：本轮规则、预注册与源代码；fits.json、inner_search.csv、models/：训练ID、成熟截止、验证选参、权重。',
             '- predictions_BEFORE_EVALUATION.parquet与stock_decisions_BEFORE_EVALUATION.parquet：评价前预测与选股。',
             '- event_prefix_audit.json、stock_prefix_audit.json：截断未来数据后历史输出一致性。',
             '- event_price_comparison_NOT_ACCOUNT.csv：每个方案每个事件的候选篮子与同日基准。', '']
    (out / 'REPORT.md').write_text('\n'.join(lines))
    manifest = json.loads((out / 'manifest.json').read_text())
    manifest.update(status='complete', model_predictions=int(len(pred)), selected_events=int(pred.selected.sum()),
                    data_vintage_certified=False, account_backtest=False,
                    reporter_sha256=digest(__file__),
                    artifacts={str(p.relative_to(out)): digest(p) for p in out.rglob('*') if p.is_file() and p.name != 'manifest.json'})
    (out / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(counts.to_string(index=False))
    print(summary[summary.horizon.eq(20)].to_string(index=False))
    print('REPORT', out / 'REPORT.md')


if __name__ == '__main__':
    main()

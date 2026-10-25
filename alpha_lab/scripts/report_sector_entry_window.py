#!/usr/bin/env python3
"""汇总五日资格对照；价格观察与可执行账户结果明确分开。"""
from pathlib import Path
import argparse
import hashlib
import json
import shutil

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + '\n')


def official_limits(codes):
    """合并已有逐股缓存；冲突保留为拒绝核验，不默选某个版本。"""
    pieces, sources = [], {}
    for code in codes:
        for path in sorted((ROOT / 'alpha_lab/data/raw/execution_audit').glob(f'*/stk_limit/{code}.parquet')):
            frame = pd.read_parquet(path)
            if frame.empty:
                continue
            assert frame.ts_code.eq(code).all()
            frame['date'] = pd.to_datetime(frame.trade_date, format='%Y%m%d')
            pieces.append(frame[['date', 'ts_code', 'up_limit', 'down_limit']])
            sources[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    if not pieces:
        return {}, set(), sources
    combined = pd.concat(pieces, ignore_index=True).drop_duplicates()
    conflicts = combined[combined.duplicated(['date', 'ts_code'], keep=False)]
    conflict_keys = set(zip(conflicts.date, conflicts.ts_code))
    clean = combined[~combined.duplicated(['date', 'ts_code'], keep=False)]
    return clean.set_index(['date', 'ts_code']).to_dict('index'), conflict_keys, sources


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    out = args.run
    cfg = yaml.safe_load((out / 'config.yaml').read_text())
    for stage in ['events', 'models', 'stocks']:
        assert json.loads((out / f'{stage}_complete.json').read_text())['status'] == 'complete'
    stock = pd.read_parquet(out / 'stock_candidates_before_sector_gate.parquet')
    scores = pd.read_parquet(out / 'event_scores.parquet')
    days = pd.read_parquet(out / 'event_days.parquet')
    prices = pd.read_parquet(out / 'observation_hfq_open.parquet')
    closes = pd.read_parquet(out / 'observation_hfq_close.parquet')
    raw_open = pd.read_parquet(out / 'observation_open.parquet')
    volumes = pd.read_parquet(out / 'observation_vol_share.parquet')
    adjustments = pd.read_parquet(out / 'observation_adj_factor.parquet')
    calendar = prices.index
    stock = stock.merge(scores.drop(columns=['date', 'sector_code']), on='event_id', validate='many_to_one')
    days = days.merge(scores.drop(columns=['date', 'sector_code']), on='event_id', validate='many_to_one')
    scopes = ['all_events', 'random_test_selected', 'chronological_selected']
    keys = ['event_id', 'ts_code']
    counts, variants, selected = [], [], {}
    for scope in scopes:
        for variant in cfg['variants']:
            current = stock[stock[scope] & stock[variant]].sort_values(['date', *keys])
            first = current.drop_duplicates(keys).copy()
            selected[(scope, variant)] = first
            sdays = days[days[scope] & days[variant]]
            counts.append(dict(scope=scope, variant=variant,
                               model_selected_events=int(scores[scope].sum()),
                               eligible_sector_days=len(sdays), stock_days=len(current),
                               event_stocks=len(first), sector_events=first.event_id.nunique(),
                               signal_dates=current.date.nunique(), first_entry_dates=first.date.nunique()))
            first['scope'], first['variant'] = scope, variant
            variants.append(first)
    first_all = pd.concat(variants, ignore_index=True)
    first_all.to_parquet(out / 'first_opportunities.parquet', index=False)
    limits, conflicts, limit_sources = official_limits(first_all.ts_code.unique())
    save_json(out / 'execution_limit_sources.json', limit_sources)

    def opening_status(date, code, side):
        quote = raw_open.at[date, code]
        volume = volumes.at[date, code]
        if not np.isfinite(quote) or quote <= 0 or not np.isfinite(volume) or volume <= 0:
            return 'missing_quote_or_suspended'
        if (date, code) in conflicts:
            return 'conflicting_official_limits'
        bound = limits.get((date, code))
        if bound is None or not np.isfinite(list(bound.values())).all():
            return 'missing_official_limits'
        if (side == 'buy' and quote >= bound['up_limit']) or (side == 'sell' and quote <= bound['down_limit']):
            return 'blocked_at_price_limit'
        return 'open_quote_and_limit_pass_only'
    comparisons = []
    for scope in scopes:
        a = selected[(scope, 'same_event_today')].set_index(keys)
        b = selected[(scope, 'persistent_5d')].set_index(keys)
        old_keys, new_keys = set(a.index), set(b.index)
        for key in sorted(old_keys | new_keys):
            old_date = a.at[key, 'date'] if key in old_keys else pd.NaT
            new_date = b.at[key, 'date'] if key in new_keys else pd.NaT
            status = 'added' if key not in old_keys else 'removed' if key not in new_keys else 'earlier' if new_date < old_date else 'later' if new_date > old_date else 'unchanged'
            comparisons.append(dict(scope=scope, event_id=key[0], ts_code=key[1], status=status,
                                    same_event_date=old_date, persistent_date=new_date))
    compare = pd.DataFrame(comparisons)
    compare.to_csv(out / 'opportunity_changes.csv', index=False)
    pd.DataFrame(counts).to_csv(out / 'opportunity_counts.csv', index=False)
    yearly = first_all.assign(year=first_all.date.dt.year).groupby(['scope', 'variant', 'year']).agg(
        event_stocks=('ts_code', 'size'), sector_events=('event_id', 'nunique'), signal_dates=('date', 'nunique'))
    yearly.to_csv(out / 'opportunities_by_year.csv')
    # 只生成价格观察；缺失未来路径不自动当作零收益或可卖出。
    samples = []
    ocfg = cfg['observation']
    for row in first_all.itertuples():
        entry_i = calendar.get_loc(row.date) + 1
        for horizon in ocfg['horizons']:
            exit_i = entry_i + horizon
            rec = dict(scope=row.scope, variant=row.variant, event_id=row.event_id, ts_code=row.ts_code,
                       signal_date=row.date, horizon=horizon, complete=False, status='right_censored')
            if exit_i < len(calendar):
                entry = prices.iloc[entry_i][row.ts_code]
                exit_price = prices.iloc[exit_i][row.ts_code]
                path = closes[row.ts_code].iloc[entry_i:exit_i]
                rec.update(entry_date=calendar[entry_i], exit_date=calendar[exit_i], entry_hfq_open=entry,
                           exit_hfq_open=exit_price, status='missing_or_invalid_quote',
                           entry_execution_check=opening_status(calendar[entry_i], row.ts_code, 'buy'),
                           exit_execution_check=opening_status(calendar[exit_i], row.ts_code, 'sell'),
                           adjustment_changes=int(adjustments[row.ts_code].iloc[entry_i:exit_i + 1].pct_change(fill_method=None).abs().gt(1e-5).sum()))
                complete = bool(np.isfinite([entry, exit_price]).all() and min(entry, exit_price) > 0
                                and len(path) == horizon and np.isfinite(path).all() and path.gt(0).all())
                if complete:
                    gross = exit_price / entry - 1
                    net = exit_price * (1 - ocfg['sell_cost'] - ocfg['slippage_per_side']) / (
                        entry * (1 + ocfg['buy_cost'] + ocfg['slippage_per_side'])) - 1
                    rec.update(complete=True, status='price_observation_only', gross_return=gross,
                               cost_adjusted_price_return=net, minimum_close_relative_to_entry=float((path / entry - 1).min()))
            samples.append(rec)
    observed = pd.DataFrame(samples)
    observed.to_parquet(out / 'price_observations_NOT_ACCOUNT.parquet', index=False)
    if len(observed):
        observed.groupby(['scope', 'variant', 'horizon', 'entry_execution_check', 'exit_execution_check'], dropna=False).size().rename('observations').to_csv(out / 'execution_coverage.csv')
    metrics = []
    if len(observed):
        observed = observed.merge(compare[['scope', 'event_id', 'ts_code', 'status']].rename(columns={'status': 'opportunity_change'}),
                                  on=['scope', 'event_id', 'ts_code'], how='left', validate='many_to_one')
        for (scope, variant, horizon), group in observed.groupby(['scope', 'variant', 'horizon']):
            for subset in ['all', 'added']:
                part = group if subset == 'all' else group[group.opportunity_change.eq('added') & group.variant.eq('persistent_5d')]
                if part.empty:
                    continue
                valid = part[part.complete]
                x = valid.cost_adjusted_price_return
                metrics.append(dict(scope=scope, variant=variant, horizon=horizon, subset=subset,
                                    opportunities=len(part), complete=len(valid), incomplete=len(part) - len(valid),
                                    sector_events=valid.event_id.nunique(), mean=x.mean(), median=x.median(),
                                    q10=x.quantile(.1), positive=x.gt(0).mean(), minimum=x.min(),
                                    mean_minimum_close=valid.minimum_close_relative_to_entry.mean()))
    metrics = pd.DataFrame(metrics)
    metrics.to_csv(out / 'price_observation_summary_NOT_ACCOUNT.csv', index=False)
    stock_variant = cfg.get('stock_gate_ablation', 'original')
    lines = ['# 入场资格与换手门槛：历史回放结果', '',
             f'个股规则版本：{stock_variant}。original保留原规则；no_expansion_gate只取消放量硬门槛；no_turnover_gates取消两项换手硬门槛。其他价格、趋势及资格条件不变。数据截至2026-10-12；实时配置、模型及候选指针未修改。', '',
             '## 机会数量', '',
             '| 事件范围 | 版本 | 板块资格日 | 股票候选日 | 去重事件×股票 | 有股票的板块事件 | 候选日期 |',
             '|---|---|---:|---:|---:|---:|---:|']
    for row in counts:
        lines.append('| {scope} | {variant} | {eligible_sector_days} | {stock_days} | {event_stocks} | {sector_events} | {signal_dates} |'.format(**row))
    lines += ['', 'all_events是不经过模型的结构对照；random_test_selected是原随机留出且FNN通过事件，含跨时间训练问题；chronological_selected按年初已成熟标签重训，固定4/8规则两组模型取或。三个范围不能混作同一总体。', '',
              'legacy_today保留当日代表窗口确认；same_event_today要求与原确认同窗口、同事件、同峰值；persistent_5d保留确认日及后4个交易日资格，结构失效后永久取消。', '',
              '## 五日保留相对严格同事件当天确认的变化', '',
              compare.groupby(['scope', 'status']).size().to_string() if len(compare) else '无个股机会。', '',
              '## 次日开盘价格观察', '',
              '以下仅是复权开盘价格观察及固定成本压力测试：买入10bp、卖出20bp、双边各5bp滑点；没有整手最小佣金、正式涨跌停与停牌匹配、资金占用、公司行动现金账本。不能当作可实现收益或组合最大回撤。缺失路径与截尾数量单独披露。', '',
              '```', metrics.to_string(index=False) if len(metrics) else '无可评价价格观察。', '```', '',
              '## 有效性边界', '',
              '- 时间推进模型仅使用成熟且年初前20交易日已结束的标签；训练样本不足60的年份不评分。全部训练ID、截止日、门槛和权重保存。固定超参数来自已观察历史，仍是历史开发复核，不是新的独立测试。',
              '- 历史行业成员和名称区间使用当时生效区间，没有用当前在市状态筛选历史；名称缺失或多条冲突拒绝。供应商历史修订与发布时间PIT尚未完整认证。',
              '- 当前在线补充成员无法完全历史复原；本轮使用冻结历史成员，不能称原实时扫描逐股票完全一致。板块331个原事件已逐条精确复原。',
              '- 不能用同一板块多股票、多日重复信号扩大独立样本数；新增/消失/提前入场分别记录。',
              '- execution_coverage.csv单独核对计划买卖日的原始开盘、成交量及已有官方涨跌停缓存；即使这些通过，也没有完成整手现金账户与公司行动结算验证。价格观察汇总不会把缺少限价资料的样本冒充可交易样本。',
              '- 生产采用还须新增机会成交与账户风险验证，本次没有自动部署。', '',
              '## 复现', '', f'运行目录：`{out}`。先经frozen_research.py分别运行research_sector_entry_window.py的events、models、stocks阶段，再运行本汇总脚本。', '']
    (out / 'REPORT.md').write_text('\n'.join(lines))
    sources = [Path(__file__), ROOT / 'alpha_lab/scripts/research_sector_entry_window.py',
               ROOT / 'alpha_lab/src/sector_entry_window.py', ROOT / 'alpha_lab/tests/test_sector_entry_window.py',
               ROOT / 'alpha_lab/docs/current_strategy_20261013/ENTRY_WINDOW_PREREG.md',
               ROOT / 'alpha_lab/src/sector_shape_events.py', ROOT / 'alpha_lab/src/sector_shape_expansion.py',
               ROOT / 'alpha_lab/src/sector_multiscale_events.py', ROOT / 'alpha_lab/src/left_side_rules.py',
               ROOT / 'alpha_lab/src/left_side_trend.py', ROOT / 'alpha_lab/experiments/sector_fuzzy_v1/model.py']
    (out / 'sources').mkdir(exist_ok=True)
    for path in sources:
        shutil.copy2(path, out / 'sources' / path.name)
    manifest = json.loads((out / 'manifest.json').read_text())
    manifest.update(status='complete', account_backtest=False, production_modified=False,
                    sources={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
                    artifacts={str(p.relative_to(out)): hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in out.rglob('*') if p.is_file() and p.name != 'manifest.json'})
    save_json(out / 'manifest.json', manifest)
    print(pd.DataFrame(counts).to_string(index=False), flush=True)
    print(compare.groupby(['scope', 'status']).size().to_string() if len(compare) else 'NO_OPPORTUNITIES', flush=True)
    print('REPORT', out / 'REPORT.md', flush=True)


if __name__ == '__main__':
    main()

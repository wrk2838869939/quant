"""最近完整启动窗口的入选持续性、行业暴露与退出后价格诊断。"""
from pathlib import Path
import argparse
import json

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--run', type=Path, required=True)
    run = p.parse_args().run; cfg = yaml.safe_load((run / 'config.yaml').read_text()); out = run / 'analysis'
    summary = pd.read_csv(out / 'all_accounts.csv')
    cohort = sorted(summary.loc[summary.kind.eq('startup'), 'cohort'].unique())[-1]
    offset = max(cfg['calendar_days']); boot = pd.Timestamp(summary.loc[summary.cohort.eq(cohort) & summary.offset.eq(offset), 'boot_date'].iloc[0])
    basics = pd.read_parquet(run / 'inputs/basics.parquet').set_index('ts_code')
    ranks = pd.read_parquet(run / 'signals/daily_original.parquet')
    smooth = pd.read_parquet(run / 'signals/daily_smooth20.parquet')
    records = []; codes = set(); cash = []
    for policy in ['baseline', 'smooth20']:
        name = f'startup_{cohort}_{offset:02d}_{policy}'
        folder = (ROOT / cfg['startup_parent'] / 'accounts' / f'{cohort}_{offset:02d}_fresh_once'
                  if policy == 'baseline' else run / 'accounts' / name)
        trades = pd.read_parquet(folder / 'trades.parquet')
        first = trades.loc[trades.side.eq('buy') & trades.decision_date.eq(boot)].copy()
        first['stock_name'] = first.ts_code.map(basics.name)
        first['policy'] = policy
        records.extend(first.to_dict('records')); codes.update(first.ts_code)
        bank_gross = first.loc[first.stock_name.str.contains('银行', na=False), 'gross'].sum()
        row = summary.loc[summary.name.eq(name)].iloc[0]
        cash.append(dict(policy=policy, cohort=cohort, boot=str(boot.date()), start=row.start, end=row.end,
            total=row.total, mdd=row.mdd, initial_bank_buy_gross=float(bank_gross), initial_bank_gross_fraction=float(bank_gross/cfg['capital'])))
    pd.DataFrame(records).to_csv(out / 'recent_example_initial_fills.csv', index=False)
    current = ranks.loc[boot].rank(ascending=False, method='min')
    averaged = smooth.loc[boot].rank(ascending=False, method='min')
    qualified = ranks.loc[:boot].tail(cfg['primary_window']).notna().sum()
    features = pd.DataFrame(dict(stock_name=basics.name, current_rank=current, smooth_rank=averaged, qualified_days20=qualified)).loc[sorted(codes)]
    features.to_csv(out / 'recent_example_score_states.csv')
    (out / 'recent_example_summary.json').write_text(json.dumps(cash, ensure_ascii=False, indent=2))
    exits = pd.read_csv(out / 'rank_exit_forward_marks.csv')
    # 同一股票/成交日/政策的重复账户只保留一次；仍不是独立随机事件。
    unique = exits.loc[exits.kind.eq('long')].drop_duplicates(['policy', 'ts_code', 'sell_date'])
    records = []
    for policy, group in unique.groupby('policy'):
        for horizon in [20, 60]:
            values = group[f'after_{horizon}'].dropna()
            records.append(dict(policy=policy, horizon=horizon, n=len(values), positive=int(values.gt(0).sum()),
                median=float(values.median()), mean=float(values.mean()), missing=int(group[f'after_{horizon}'].isna().sum())))
    pd.DataFrame(records).to_csv(out / 'rank_exit_forward_summary.csv', index=False)
    print(pd.DataFrame(cash).to_string(index=False)); print(features.to_string()); print(pd.DataFrame(records).to_string(index=False))


if __name__ == '__main__': main()

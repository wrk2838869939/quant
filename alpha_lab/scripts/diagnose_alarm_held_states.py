#!/usr/bin/env python3
"""警报收盘的原持仓趋势状态：只作过程归因，不改变日/月复核规则。"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    run = parser.parse_args().run.resolve()
    frame = pd.read_csv(run / 'cash_held_baskets/initial_basket_components.csv', parse_dates=['signal_date'])
    with np.load(run / 'policy_market.npz', allow_pickle=False) as saved:
        dates, codes = pd.DatetimeIndex(saved['dates']), pd.Index(saved['codes'])
        trend, observable = saved['trend'], saved['observable']
    assert dates.max() < pd.Timestamp('2024-01-01')
    hold = pd.read_parquet(run / 'cash_execution/monthly_hold_allowed.parquet')
    records = []
    for row in frame.itertuples():
        i, j = dates.get_loc(row.signal_date), codes.get_loc(row.ts_code)
        previous = hold.index.searchsorted(row.signal_date, side='right')-1
        assert previous >= 0
        records.append(dict(case=row.case, alarm_id=row.alarm_id, signal_date=row.signal_date,
            ts_code=row.ts_code, weight=row.frozen_weight,
            observable=bool(observable[i,j]), raw_trend=bool(trend[i,j]) if observable[i,j] else np.nan,
            known_failure=bool(observable[i,j] and not trend[i,j]),
            last_review=hold.index[previous], last_review_hold_allowed=hold.iloc[previous][row.ts_code]))
    result = pd.DataFrame(records)
    result['known_failure_weight'] = result.weight * result.known_failure
    result.to_csv(run / 'cash_held_baskets/alarm_held_states.csv', index=False)
    result.groupby(['case','alarm_id','signal_date']).agg(
        stocks=('ts_code','size'), known=('observable','sum'), failed=('known_failure','sum'),
        failed_account_weight=('known_failure_weight','sum')).to_csv(run / 'cash_held_baskets/alarm_held_states_summary.csv')


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""按共同成熟可入场样本展示政策时钟与删失，不把未结束段当已平仓。"""
import argparse
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    run = parser.parse_args().run.resolve()
    labels = pd.read_parquet(run / 'policy_labels.parquet')
    records = []
    for window, lo, hi in [('fit_2014_2018','2014-01-01','2018-12-31'),
                          ('development_2019_2021','2019-01-01','2021-12-31'),
                          ('execution_2022_2023','2022-01-01','2023-12-31')]:
        all_rows = labels[labels.date.between(lo, hi) & labels.trend]
        mature = all_rows[all_rows.label_end_120.le(pd.Timestamp(hi))]
        cohort = mature[~mature.execution_unknown_120 & ~mature.entry_missing_120]
        for target in ['fixed60', 'fixed120', 'patient120']:
            prefix = target+'__'
            records.append(dict(window=window, target=target, mature_trend_opportunities=len(mature),
                common_entered_known=len(cohort),
                unknown_mature=int(mature.execution_unknown_120.sum()),
                no_entry_mature=int(mature.entry_missing_120.sum()),
                completed_sales=int(cohort[prefix+'actual_sell_open_step'].notna().sum()),
                delisted=int(cohort[prefix+'delist_writeoff_step'].notna().sum()),
                open_at_cutoff=int(cohort[prefix+'open_at_cutoff'].sum()),
                open_fraction=cohort[prefix+'open_at_cutoff'].mean(),
                pending_at_cutoff=int(cohort[prefix+'pending_exit'].sum()),
                delayed_sell_opportunities=int(cohort[prefix+'blocked_sell_days'].gt(0).sum()),
                unknown_review_opportunities=int(cohort[prefix+'unknown_reviews'].gt(0).sum()),
                first_failure_median=cohort[prefix+'first_failure_review_step'].median(),
                earliest_requested_sell_median=cohort[prefix+'earliest_sell_open_step'].median(),
                completed_sell_median=cohort[prefix+'actual_sell_open_step'].median(),
                request_to_fill_max=cohort[prefix+'request_to_fill_intervals'].max(),
                mean_cash_occupation=cohort[prefix+'cash_occupation_intervals'].mean(),
                date_equal_gross_return=cohort.groupby('date')[target+'_return'].mean().mean()))
    pd.DataFrame(records).to_csv(run / 'policy_label_process_summary.csv', index=False)


if __name__ == '__main__':
    main()

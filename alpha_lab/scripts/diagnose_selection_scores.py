"""相邻交易日名单诊断，仅衡量信号变化，不冒充日频交易回测。"""
from pathlib import Path
import argparse
import json

import pandas as pd


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--run', type=Path, required=True)
    run = p.parse_args().run; rows = []
    for policy in ['original', 'smooth10', 'smooth20', 'smooth40']:
        daily = pd.read_parquet(run / f'signals/daily_{policy}.parquet').loc['2022-01-01':]
        prior = None
        for date, values in daily.iterrows():
            current = set(values.dropna().nlargest(5).index)
            if prior is not None:
                rows.append(dict(policy=policy, date=date, retained=len(current & prior),
                    changed=len(current - prior), identical=current == prior,
                    jaccard=len(current & prior) / len(current | prior) if current | prior else 1.))
            prior = current
    frame = pd.DataFrame(rows); frame.to_csv(run / 'adjacent_day_top5.csv', index=False)
    stats = frame.groupby('policy').agg(pairs=('date','count'), identical_fraction=('identical','mean'),
        mean_new_stocks=('changed','mean'), mean_jaccard=('jaccard','mean'))
    stats.to_csv(run / 'adjacent_day_summary.csv')
    print(stats.to_string())


if __name__ == '__main__': main()

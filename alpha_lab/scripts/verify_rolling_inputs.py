"""对全部新日历做真实数据前缀截断及日历间隔审计。"""
import argparse
from pathlib import Path
import sys

import pandas as pd
import yaml

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from conditional_hold import build_conditional_holds
from rolling_review import decision_dates,schedule_specs
from research_profit_defense import digest,save
from research_rolling_review import verify_manifest


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    run=p.parse_args().run;verify_manifest(run)
    cfg=yaml.safe_load((run/'config.yaml').read_text())
    spec=yaml.safe_load((ROOT/'configs/conditional_hold_research.yaml').read_text())
    spec['variants']={'hybrid':spec['variants']['hybrid']}
    close,trend,known=[pd.read_parquet(run/'inputs'/f'{name}.parquet') for name in
                       ['daily_close','daily_trend','daily_observable']]
    calendar=close.index[close.index>=pd.Timestamp(cfg['start'])]
    rows=[]
    for schedule in schedule_specs(cfg):
        original=pd.read_parquet(run/'schedules'/schedule['name']/'hold.parquet')[close.columns]
        dates=decision_dates(calendar,schedule['kind'],schedule['parameter'],cfg['rolling_interval'],cfg['start'])
        assert dates.equals(original.index)
        gaps=pd.Series(calendar.get_indexer(dates)).diff().dropna()
        if schedule['kind']=='rolling':assert gaps.eq(cfg['rolling_interval']).all()
        for stop in pd.to_datetime(['2023-06-30','2025-06-30']):
            prefix=calendar[calendar<=stop]
            selected=decision_dates(prefix,schedule['kind'],schedule['parameter'],cfg['rolling_interval'],cfg['start'])
            assert selected.equals(dates[dates<=stop])
            holds,_=build_conditional_holds(close.loc[:stop],trend.loc[:stop],known.loc[:stop],selected,spec)
            pd.testing.assert_frame_equal(holds['hybrid'],original.loc[selected],check_exact=True)
            rows.append(dict(schedule=schedule['name'],cutoff=stop,rows=len(selected),codes=len(close.columns),
                             prefix_exact=True,min_review_gap=gaps.min(),max_review_gap=gaps.max()))
    save(run/'input_prefix_verification.json',dict(status='passed',checks=rows,
        source_sha256=digest(__file__),interpretation='dates and hold permissions only; not independent data authenticity'))
    print('Verified',len(rows),'real-history holding/calendar prefixes',flush=True)


if __name__=='__main__':main()

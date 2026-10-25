#!/usr/bin/env python3
"""只读复放高收益续训候选，比较原引擎与缓存引擎；不反馈训练选择。"""
from pathlib import Path
import argparse,json,sys
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import run_policy_convergence as E
from initialized_policy import NeuralPolicy
from run_classic_strategy_smoke import save_json,digest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run',type=Path)
    parser.add_argument('--final',action='store_true',help='全部训练完成后，核验各年最终状态中收益最高的冠军')
    args=parser.parse_args();run=args.run.resolve()
    if args.final:
        assert json.loads((run/'manifest.json').read_text())['status']=='complete'
    E.setup(ROOT/'configs/policy_convergence_v1.yaml',run)
    chosen=[]
    for year in E.G['plan']['outer_years']:
        paths=list((run/'search'/str(year)).glob('*/*/state_g064.json'))
        if args.final:
            paths=[sorted(p.parent.glob('state_g*.json'))[-1] for p in paths]
        if paths:
            state=max((json.loads(p.read_text()) for p in paths),key=lambda s:s['champion']['metrics']['total'])
            chosen.append((year,state['generation'],state['champion']))
    results=[]
    for year,generation,row in chosen:
        theta=np.load(E.resolve(row)/'theta.npy');package=E.G['training'][year]
        create=lambda:NeuralPolicy(package,theta,E.G['plan']['hidden'],E.G['portfolio']['capital'])
        a=E.R.execute(create(),E.G['plan']['train_start'],f'{year-2}-12-31')
        b=E.execute_cached(create(),E.G['plan']['train_start'],f'{year-2}-12-31')
        for table in E.R.original.TABLES:
            pd.testing.assert_frame_equal(getattr(a,table),getattr(b,table),check_exact=True)
            pd.testing.assert_frame_equal(getattr(b,table),pd.read_parquet(E.resolve(row)/f'{table}.parquet'),check_exact=True)
        audit=E.R.original.audit(a)
        for k in ['total','ann','mdd','n_trades','total_fees']:
            assert a.summary[k]==row['metrics'][k]
        record=dict(year=year,generation=generation,source=row['folder'],theta_sha256=digest(E.resolve(row)/'theta.npy'),
                    four_tables_original_cached_archive_exact=True,metrics=a.summary,audit=audit)
        results.append(record);print(json.dumps(record),flush=True)
    out=run/'diagnostics';out.mkdir(exist_ok=True)
    save_json(out/('high_return_engine_parity_final.json' if args.final else 'high_return_engine_parity.json'),results)


if __name__=='__main__':main()

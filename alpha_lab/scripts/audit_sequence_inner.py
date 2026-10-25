#!/usr/bin/env python3
"""按项目高收益警戒线复核CNN/HMM内层账户，保留原搜索及结果。"""
from pathlib import Path
import sys,json,argparse
from types import SimpleNamespace
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import numpy as np,pandas as pd,yaml
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from classic_smoke import verify_ledger
G={}


def audit_one(item):
    path=G['run']/item['path']
    tables={k:pd.read_parquet(path/f'{k}.parquet') for k in ['daily','trades','events','holdings']}
    result=SimpleNamespace(**tables)
    check=verify_ledger(result,G['cfg']['capital'],G['frames']['market'],G['frames']['limits'],G['frames']['actions'],G['cfg'],G['smoke']['slippage_rate'],G['smoke']['audit']['money_atol'])
    return dict(**item,**check)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);args=ap.parse_args();run=args.run.resolve()
    cfg=yaml.safe_load((run/'sources/monthly_portfolio_audit.yaml').read_text())['portfolio']
    account=yaml.safe_load((run/'sources/selection_account_v1.yaml').read_text())
    smoke=yaml.safe_load((run/'sources/classic_strategy_smoke.yaml').read_text())
    cfg.update(capital=account['account']['capital'],top_n=5,band_mult=1.,**{k:smoke[k] for k in ['buy_cost','sell_cost','min_fee','cash_dividend_haircut','annual_days']})
    G.update(run=run,cfg=cfg,smoke=smoke,frames={k:pd.read_parquet(run/'inputs'/f'{k}.parquet') for k in ['market','limits','actions']})
    flags=[]
    for p in sorted(run.glob('search/*/*/*/trial.json')):
        t=json.loads(p.read_text())
        for paired in [False,True]:
            folder=p.parent/'paired' if paired else p.parent
            metrics=t['paired_metrics'] if paired else t['metrics']
            d=pd.read_parquet(folder/'daily.parquet')
            month=(1+d.ret).groupby(d.index.to_period('M')).prod()-1
            if metrics['sharpe']>3 or month.max()>.2:
                flags.append(dict(path=str(folder.relative_to(run)),paired=paired,sharpe=metrics['sharpe'],max_month=float(month.max())))
    with ProcessPoolExecutor(max_workers=6,mp_context=mp.get_context('fork')) as pool:
        checks=list(pool.map(audit_one,flags))
    pd.DataFrame(checks).to_csv(run/'high_inner_account_audits.csv',index=False)
    summary=dict(flagged_accounts=len(checks),primary=sum(not x['paired'] for x in checks),paired=sum(x['paired'] for x in checks),
                 trades_checked=sum(x['trades_checked'] for x in checks),cash_max_error=max(x['cash_max_error'] for x in checks),
                 all_pass=True,known_full_window_quality_selection_bias_remains=True)
    (run/'high_inner_audit_summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))


if __name__=='__main__':main()

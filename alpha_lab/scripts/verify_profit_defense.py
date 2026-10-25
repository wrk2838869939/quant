"""从落盘CSV独立重算本轮结果，并验证父运行未改及无操作对照相同。"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import yaml

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from research_profit_defense import digest,save
from defense_diagnostics import risk_evidence


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    run=p.parse_args().run
    cfg=yaml.safe_load((run/'config.yaml').read_text())
    manifest=json.loads((run/'manifest.json').read_text())
    for file,h in manifest['source_hashes'].items():
        assert digest(ROOT/file)==h,file
    for file,h in manifest['inputs_sha256'].items():
        assert digest(file)==h,file
    parent=ROOT/cfg['parent']
    old=json.loads((parent/'completion.json').read_text())
    for file,h in old['artifact_sha256'].items():
        assert digest(parent/file)==h,file
    summaries=pd.read_csv(run/'summary.csv').set_index('name')
    attribution=pd.read_csv(run/'analysis/attribution_summary.csv').set_index('name')
    industry=pd.read_csv(run/'analysis/industry_pnl.csv')
    findings=[];orders_total=0;fills_total=0
    for case in manifest['cases']:
        f=run/'execution'/case['name']
        receipt=json.loads((f/'completion.json').read_text())
        for file,h in receipt['artifacts'].items():
            assert digest(f/file)==h,(case['name'],file)
        d=pd.read_csv(f/'daily.csv',index_col='date',parse_dates=True)
        ret=d.equity.pct_change();ret.iloc[0]=d.equity.iloc[0]/cfg['capital']-1
        metrics=dict(total=d.equity.iloc[-1]/cfg['capital']-1,
            ann=(d.equity.iloc[-1]/cfg['capital'])**(244/len(d))-1,
            mdd=(d.equity/d.equity.cummax().clip(lower=cfg['capital'])-1).min(),
            sharpe=ret.mean()/ret.std(ddof=1)*np.sqrt(244))
        errors={k:abs(v-summaries.at[case['name'],k]) for k,v in metrics.items()}
        assert max(errors.values())<1e-10
        a=attribution.loc[case['name']]
        np.testing.assert_allclose(a.net_profit,a.price_pnl+a.dividends-a.fees,atol=1e-7)
        np.testing.assert_allclose(a.net_profit,a.sector_proxy_pnl+a.residual_pnl+a.unclassified_pnl,atol=1e-7)
        np.testing.assert_allclose(a.net_profit,industry[industry.name.eq(case['name'])].pnl.sum(),atol=1e-7)
        t=pd.read_csv(f/'trades.csv',parse_dates=['date','decision_date'])
        assert (t.date>t.decision_date).all()
        assert t.loc[t.side.eq('buy'),'shares'].mod(100).eq(0).all()
        e=pd.read_csv(f/'events.csv',parse_dates=['date'])
        order=e[e.kind.eq('order_planned')]
        buy=order[order.side.eq('buy')]
        budget=buy.groupby('decision_date').reserved_cash.sum()
        for date,reserved in budget.items():
            assert reserved <= d.at[pd.Timestamp(date),'cash']+1e-7
        assert d.cash.ge(-1e-7).all() and d.n_hold.le(5).all()
        orders_total+=len(order);fills_total+=len(t)
        if case['variant']=='hybrid':
            for name in ['daily','trades','holdings','events']:
                pd.testing.assert_frame_equal(pd.read_csv(f/f'{name}.csv'),
                    pd.read_csv(parent/'execution'/case['name']/f'{name}.csv'),check_exact=True)
        if case['variant']=='initial_buyhold':
            original=pd.read_csv(run/'execution'/f"{case['window']}_hybrid/trades.csv")
            first=original[original.date.eq(original.date.min())].reset_index(drop=True)
            new=pd.read_csv(f/'trades.csv').reset_index(drop=True)
            pd.testing.assert_frame_equal(first,new,check_exact=True)
        findings.append(dict(name=case['name'],metric_errors=errors,orders=len(order),fills=len(t),
                             industry_coverage=a.industry_coverage,account_days=len(d)))
    # 本轮周度组合门未触发，三个账户应完全一致，不只终值相同。
    for window in cfg['windows']:
        for variant in ['weekly_pause','weekly_liquidate']:
            for name in ['daily','trades','holdings','events']:
                pd.testing.assert_frame_equal(
                    pd.read_csv(run/'execution'/f'{window}_weekly_exit'/f'{name}.csv'),
                    pd.read_csv(run/'execution'/f'{window}_{variant}'/f'{name}.csv'),check_exact=True)
    # 风险的基础trend/observable与父实验逐格核验，浮点边界不许悄悄更改。
    original=ROOT/cfg['signal_parent']
    prices=pd.read_parquet(original/'daily_close.parquet')
    evidence=risk_evidence(prices,cfg['defense'])
    for key,file in [('trend','daily_trend'),('known','daily_observable')]:
        pd.testing.assert_frame_equal(evidence[key],pd.read_parquet(original/f'{file}.parquet'),check_exact=True)
    save(run/'verification.json',dict(status='passed',parent_artifacts_unchanged=len(old['artifact_sha256']),
        cases=findings,orders=orders_total,fills=fills_total,original_hybrid_exact=True,
        initial_orders_exact=True,weekly_inactive_gates_exact=True,original_trend_evidence_exact=True,
        incomplete_industry_accounting_explicit=True,source_sha256=digest(__file__)))
    print('Verified',len(findings),'accounts;',orders_total,'orders;',fills_total,'fills;',
          len(old['artifact_sha256']),'unchanged parent artifacts',flush=True)


if __name__=='__main__':
    main()

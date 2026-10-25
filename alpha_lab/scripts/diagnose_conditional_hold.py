"""分治退出的固定持仓事件及实际分配路径审计；不据结果拟合新规则。"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import yaml


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    run=p.parse_args().run;manifest=json.loads((run/'manifest.json').read_text())
    regimes=pd.read_parquet(run/'weak_regime.parquet')
    long_return=pd.read_parquet(run/'long_return.parquet')
    gap=pd.read_parquet(run/'long_ma_gap.parquet')
    ma=pd.read_parquet(run/'long_ma.parquet')
    prices=pd.read_parquet(run/'daily_close.parquet')
    prior=prices.shift(manifest['parameters']['long_window'])
    market=pd.read_parquet(run/'execution/market.parquet').set_index(['date','ts_code'])
    tick=yaml.safe_load((run/'execution/execution_config.yaml').read_text())['portfolio']['price_tick']
    source=pd.read_csv(run/'fixed_first_failures.csv',parse_dates=['signal_date'])
    source['ret120']=[long_return.at[r.signal_date,r.ts_code] for r in source.itertuples()]
    source['ma120_gap']=[gap.at[r.signal_date,r.ts_code] for r in source.itertuples()]
    source.to_csv(run/'fixed_state_evidence.csv',index=False)
    groups=[]
    for (window,state),block in source.groupby(['window','weak_regime']):
        valid=block.forward20_adjusted_price_return.dropna()
        groups.append(dict(window=window,state=state,events=len(block),forward20_known=len(valid),
            forward20_positive=int(valid.gt(0).sum()),forward20_mean=valid.mean(),
            mean_ret120=block.ret120.mean(),mean_ma120_gap=block.ma120_gap.mean(),
            **{v+'_early_exits':int(block[v+'_exit'].sum()) for v in manifest['parameters']['variants']}))
    pd.DataFrame(groups).to_csv(run/'fixed_state_summary.csv',index=False)
    decisions=[]
    for case in manifest['cases']:
        if not case['name'].startswith(('cold_','continued_')):continue
        e=pd.read_csv(run/'execution'/case['name']/'events.csv',parse_dates=['date','signal_date'])
        a=e[e.kind.eq('allocation_decision') & e.old_shares.gt(0)]
        hold=pd.read_parquet(run/'hold_variants'/(case['variant']+'.parquet'))
        baseline=pd.read_parquet(run/'hold_variants'/'baseline.parquet')
        for r in a.itertuples():
            state=regimes.at[r.signal_date,r.ts_code]
            actual=hold.at[r.signal_date,r.ts_code];original=baseline.at[r.signal_date,r.ts_code]
            quote=market.loc[(r.signal_date,r.ts_code)] if (r.signal_date,r.ts_code) in market.index else None
            ma_ticks=((quote.close-ma.at[r.signal_date,r.ts_code]/quote.adj_factor)/tick
                      if quote is not None else np.nan)
            return_ticks=((quote.close-prior.at[r.signal_date,r.ts_code]/quote.adj_factor)/tick
                          if quote is not None else np.nan)
            decisions.append(dict(name=case['name'],variant=case['variant'],date=r.date,signal_date=r.signal_date,
                ts_code=r.ts_code,state=state,baseline_allowed=original,variant_allowed=actual,
                permission_changed=bool(pd.notna(actual) and pd.notna(original) and actual!=original),
                old_shares=r.old_shares,target_shares=r.target_shares,
                ret120=long_return.at[r.signal_date,r.ts_code],ma120_gap=gap.at[r.signal_date,r.ts_code],
                ma_boundary_distance_ticks=ma_ticks,return120_boundary_distance_ticks=return_ticks))
    pd.DataFrame(decisions).to_csv(run/'actual_held_review_states.csv',index=False)
    # 2×2交互是同本金完整账户政策路径的利润差，不是独立选股alpha。
    interactions=[]
    for prefix in ['cold','continued','small','large','cost2']:
        cases=[c for c in manifest['cases'] if c['name'].startswith(prefix+'_')]
        ends={c['variant']:pd.read_csv(run/'execution'/c['name']/'daily.csv').equity.iloc[-1] for c in cases}
        interactions.append(dict(scenario=prefix,accelerated_delta_yuan=ends['accelerated']-ends['baseline'],
            patient_delta_yuan=ends['patient']-ends['baseline'],hybrid_delta_yuan=ends['hybrid']-ends['baseline'],
            interaction_yuan=ends['hybrid']-ends['accelerated']-ends['patient']+ends['baseline']))
    pd.DataFrame(interactions).to_csv(run/'factorial_account_differences.csv',index=False)
    print(pd.DataFrame(groups).to_string(index=False))
    print(pd.DataFrame(interactions).to_string(index=False))


if __name__=='__main__':main()

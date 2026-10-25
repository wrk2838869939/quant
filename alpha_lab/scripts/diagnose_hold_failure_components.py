"""固定基线首次转弱事件的原三条件分解；只做描述，不生成新交易规则。"""
import argparse
from pathlib import Path
import pandas as pd


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    run=parser.parse_args().run
    prices=pd.read_parquet(run/'daily_close.parquet').where(lambda x:x>0)
    components={'ret60':prices/prices.shift(60)-1,'ret120':prices/prices.shift(120)-1,
                'ma120_gap':prices/prices.rolling(120,min_periods=120).mean()-1}
    events=pd.read_csv(run/'fixed_first_failures.csv',parse_dates=['signal_date'])
    for key,values in components.items():
        events[key]=[values.at[r.signal_date,r.ts_code] for r in events.itertuples()]
    assert events[list(components)].notna().all().all()
    events['failed_components']=['|'.join(k for k in components if getattr(r,k)<=0) for r in events.itertuples()]
    assert events.failed_components.ne('').all()
    events.to_csv(run/'fixed_failure_components.csv',index=False)
    rows=[]
    for (window,reason),block in events.groupby(['window','failed_components']):
        future=block.forward20_adjusted_price_return.dropna()
        rows.append(dict(window=window,failed_components=reason,events=len(block),
            persist10_exit=int(block.persist10_exit.sum()),forward20_known=len(future),
            forward20_positive=int(future.gt(0).sum()),forward20_mean=future.mean()))
    result=pd.DataFrame(rows)
    result.to_csv(run/'failure_component_summary.csv',index=False)
    print(result.to_string(index=False))


if __name__=='__main__':main()

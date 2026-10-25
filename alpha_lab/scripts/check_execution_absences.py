"""对隔离行情缺失查询数据商停牌记录；原缺失记录保留不改。"""
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime, timezone
from pathlib import Path
import argparse
import json
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from execution_market_feed import file_hash


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',type=Path,required=True)
    args=parser.parse_args()
    out=args.stage/'absence_followup'
    out.mkdir(exist_ok=False)
    gaps=pd.read_csv(args.stage/'validated/unexplained_absences.csv')
    queries=[]; assessments=[]
    for date,group in gaps.groupby('date'):
        receipt=dict(date=date,api='suspend_d',requested_at=datetime.now(timezone.utc).isoformat())
        try:
            with open('/dev/null','w') as sink,redirect_stdout(sink),redirect_stderr(sink):
                from ts_client import get_pro
                result=get_pro().query('suspend_d',trade_date=date.replace('-',''),suspend_type='S')
            if result is None:raise ValueError('missing response')
            path=out/('suspend_d_'+date+'.parquet');result.to_parquet(path,index=False)
            receipt.update(status='downloaded',rows=len(result),file=path.name,sha256=file_hash(path))
            required={'ts_code','trade_date','suspend_type','suspend_timing'}
            if not required.issubset(result):raise ValueError('schema')
            for row in group.to_dict('records'):
                match=result.loc[result.ts_code.eq(row['ts_code']) & result.trade_date.astype(str).eq(date.replace('-','')) & result.suspend_type.eq('S')]
                full=match.suspend_timing.isna()|match.suspend_timing.astype(str).str.strip().eq('')
                row['assessment']='provider_reported_full_day_suspension' if len(match) and full.any() else 'unresolved'
                assessments.append(row)
        except Exception as exc:
            receipt.update(status='failed',error_type=type(exc).__name__)
            assessments.extend(dict(row,assessment='unresolved') for row in group.to_dict('records'))
        queries.append(receipt)
    pd.DataFrame(assessments).to_csv(out/'assessments.csv',index=False)
    (out/'receipt.json').write_text(json.dumps(dict(source='configured Tushare provider',
        documentation='https://tushare.pro/document/2?doc_id=214',queries=queries,
        source_vintage_certified=False,trade_ready=False),ensure_ascii=False,indent=2))
    print(pd.DataFrame(assessments).to_string(index=False))


if __name__=='__main__':main()

"""把显式CSV凭据原子导入影子账本；失败保留原件，整批账本回滚。"""
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from broker_csv import KINDS, import_csv_batch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--db', required=True)
    parser.add_argument('--profile', default=str(ROOT/'configs/broker_csv_template.yaml'))
    parser.add_argument('--archive', required=True)
    for kind in KINDS: parser.add_argument('--'+kind,required=True)
    args = parser.parse_args()
    try:
        result = import_csv_batch(args.db,args.profile,{k:getattr(args,k) for k in KINDS},args.archive)
        print(json.dumps(result,ensure_ascii=False))
    except Exception as exc:
        print(json.dumps(dict(status='failed',error_type=type(exc).__name__,
            detail='导入失败；请查看本地归档凭据和失败记录。'),ensure_ascii=False))
        return 1
    return 0


if __name__ == '__main__': sys.exit(main())

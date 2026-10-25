#!/usr/bin/env python3
"""仅提取旧回放纯函数，用合成路径核对语义；不导入旧脚本或读取行情。"""
import ast
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
source=ROOT.parent/'audit/scripts/bear_test.py'
tree=ast.parse(source.read_text())
function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='backtest_with_signal')
namespace=dict(np=np,pd=pd,SL=-.02,TRAIL_TRIGGER=.04,TRAIL_DIST=.02)
exec(compile(ast.Module(body=[function],type_ignores=[]),str(source),'exec'),namespace)
replay=namespace['backtest_with_signal']
rows=[]
for name, returns, signal in [
    ('five_daily_gains',[0,.01,.01,.01,.01,.01],[1]*6),
    ('gap_loss',[0,-.10],[1,1]),
    ('risk_off_while_holding',[0,.01,.01],[1,0,0])]:
    idx=pd.date_range('2000-01-03',periods=len(returns),freq='B')
    result=replay(pd.Series(returns,index=idx),pd.Series(signal,index=idx))
    rows.append(dict(case=name,market_returns=returns,signal=signal,legacy_daily=result.tolist(),
        legacy_compounded_return=float((1+result).prod()-1),
        underlying_buy_hold_return=float(np.prod(1+np.array(returns))-1)))
assert np.isclose(rows[0]['legacy_compounded_return'],1.01**9-1)
assert np.isclose(rows[0]['underlying_buy_hold_return'],1.01**5-1)
assert rows[1]['legacy_daily'][1]==-.02
assert rows[2]['legacy_daily'][2]==.01
out=ROOT/'outputs/active_cash_research_20260912'
out.mkdir(parents=True,exist_ok=True)
report=dict(source=str(source.relative_to(ROOT.parent)),sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
    market_data_read=False,findings=rows,
    boundary_note='退出累计收益重复计入；跳空虚构止损成交；风险信号只约束新开仓。不能外推为所有旧引擎均有相同错误。')
(out/'legacy_semantics.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
print(json.dumps(report,ensure_ascii=False,indent=2))

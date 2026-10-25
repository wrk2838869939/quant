"""复建当日输入并与封存信号/风险比较，生成明确标识的空仓演练。"""
from pathlib import Path
import argparse,json,sys
import numpy as np
import pandas as pd
import yaml
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from shadow_inputs import build_bundle
from shadow_planner import make_plan,fingerprint,canonical
from monthly_risk_live import read_index_close,risk_history,reviews
from execution_market_feed import file_hash
from manual_execution import bulletin


def main():
    p=argparse.ArgumentParser();p.add_argument('--observation',type=Path,required=True);p.add_argument('--index-stage',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    obs=a.observation;cfg=yaml.safe_load((ROOT/'configs/monthly_risk_manual.yaml').read_text());calendar=json.loads((obs/'calendar.json').read_text());receipt=json.loads((obs/'receipt.json').read_text());date=receipt['asof']
    manifest=json.loads((obs/'manifest.json').read_text())
    for name,digest in manifest['output_sha256'].items():assert file_hash(obs/name)==digest
    for name,digest in manifest['input_sha256'].items():assert file_hash(name)==digest
    index=read_index_close(a.index_stage/'index_close.parquet');assert file_hash(a.index_stage/'index_close.parquet')==receipt['market_index']['sha256']
    model=json.loads((ROOT/'outputs/shadow_execution/20260914T054517_770961Z/model_2026.json').read_text())
    raw=pd.read_parquet(obs/'feature_panel.parquet');market=pd.read_parquet(obs/'market.parquet')
    result=build_bundle(raw,market,pd.read_parquet(obs/'basics.parquet'),pd.read_parquet(obs/'st.parquet'),pd.read_parquet(obs/'actions.parquet'),date,calendar,cfg,model,[],receipt,index)
    assert result==json.loads((obs/'bundle.json').read_text())
    base=ROOT/'outputs/selection_stability/20260914T070228_976007Z';active=pd.Timestamp(result['signal_date'])
    score=pd.read_parquet(base/'schedules/calendar_01/rank_buffer/signals.parquet').loc[active].dropna().sort_values(ascending=False,kind='stable').head(cfg['candidate_pool'])
    assert set(score.index)==set(result['signals']);score_error=max(abs(score[c]-v) for c,v in result['signals'].items());assert score_error<1e-12
    discounts=pd.read_parquet(base/'risk.parquet');risk_error=max(abs(discounts.at[active,c]-v) for c,v in result['risk_discounts'].items());assert risk_error<1e-12
    pred=pd.read_parquet(ROOT/'outputs/forward_risk/20260914T104245_763609Z/predictions.parquet').volatility
    checks=[]
    for offset in [1,8,15,22]:
        spec=json.loads(json.dumps(cfg));spec['schedule']['parameter']=offset
        history=risk_history(index,date,calendar,spec)
        err=max(abs(row['probability']-pred.loc[row['date']]) for row in history if pd.Timestamp(row['date']) in pred.index)
        assert err<1e-12;checks.append(dict(offset=offset,review_count=len(history),maximum_probability_error=err))
    snapshot=dict(account_id='illustrative_80k_empty_not_real',asof=date,source='synthetic empty account for software validation; not broker reconciliation',cash=80000.,equity=80000.,receivables=0.,frozen_cash=0.,open_orders=[],positions={})
    plan=make_plan(snapshot,result,calendar,cfg);assert not plan['orders'] and not result['is_review']
    # 使用真实当前时刻；历史演练标记强制保留，文件不会被误认作可执行邮件。
    mail=bulletin(plan,{},pd.Timestamp(date+'T21:01:00+08:00'),yaml.safe_load((ROOT/'configs/manual_execution.yaml').read_text()),backfill=True)
    (a.output/'illustrative_plan.json').write_text(canonical(plan));(a.output/'illustrative_briefing.txt').write_text(mail['body']);(a.output/'illustrative_briefing.eml').write_bytes(mail['eml'])
    future=reviews(calendar,cfg,calendar['sessions'][-1]);nextreview=str(future[future>pd.Timestamp(date)][0].date())
    report=dict(asof=date,signals_exact=True,candidate_count=len(score),maximum_score_error=score_error,maximum_stock_risk_error=risk_error,risk_probability_checks=checks,
        rebuilt_bundle_exact=True,illustrative_initial_cash=80000.,illustrative_plan_orders=len(plan['orders']),risk_active=plan['risk_active'],exposure_cap=plan['exposure_cap'],risk_probability=plan['risk_probability'],risk_review_date=plan['risk_review_date'],next_review=nextreview,
        real_account_reconciled=False,real_account_db_created=False,mail_sent=False,orders_submitted=False,source_hashes={str(p.relative_to(ROOT)):file_hash(p) for p in [ROOT/'src/shadow_inputs.py',ROOT/'src/monthly_risk_live.py',ROOT/'src/shadow_planner.py',ROOT/'src/shadow_account.py',ROOT/'src/manual_execution.py']})
    (a.output/'verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2));print(canonical(report),flush=True)


if __name__=='__main__':main()

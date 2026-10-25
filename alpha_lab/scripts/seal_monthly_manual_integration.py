"""核验所消费的封存输入，封存人工接入的源码、验证与异常原件。"""
from pathlib import Path
from datetime import datetime,timezone
import json,re,shutil,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from execution_market_feed import file_hash
from shadow_planner import canonical


def main():
    out=ROOT/'outputs/monthly_manual_current';validation=out/'validation';validation.mkdir(exist_ok=False)
    log=Path('/tmp/monthly_manual_full_tests.log').read_text();assert re.search(r'Ran 64 tests',log) and log.rstrip().endswith('OK')
    checks=json.loads((ROOT/'outputs/monthly_manual_parity/attempt04/verification.json').read_text());assert len(checks['checks'])==4
    assert sum(c['orders'] for c in checks['checks'])==540 and sum(c['restores'] for c in checks['checks'])==45
    rebuilt=json.loads((out/'input_verification/verification.json').read_text());assert rebuilt['signals_exact'] and rebuilt['rebuilt_bundle_exact']
    for name,digest in rebuilt['source_hashes'].items():assert file_hash(ROOT/name)==digest
    assert json.loads((out/'append_check/bundle.json').read_text())==json.loads((out/'observation_verified/bundle.json').read_text())
    (validation/'append_verification.json').write_text(canonical(dict(previous_cutoff='2026-09-11',new_date='2026-09-14',appended_bundle_exact=True,previous_is_historical_fixture=True)))
    consumed={}
    def verify(folder,names):
        record=json.loads((folder/'completion.json').read_text());hashes=record['artifact_sha256']
        for name in names:assert file_hash(folder/name)==hashes[name],str(folder/name)
        consumed[str(folder.relative_to(ROOT))]=dict(completion_sha256=file_hash(folder/'completion.json'),files=len(names))
    research=ROOT/'outputs/monthly_predicted_budget/20260914T104757_015442Z'
    names=['protocol.json']+[f'accounts/continued_{offset:02d}_rank_cost0_budget50/{kind}.parquet' for offset in [1,8,15,22] for kind in ['daily','holdings','events','trades']]
    verify(research,names)
    selection=ROOT/'outputs/selection_stability/20260914T070228_976007Z'
    verify(selection,['risk.parquet','signals/close.parquet','inputs/market.parquet','inputs/st.parquet','inputs/normalized_actions.parquet']+[f'schedules/calendar_{offset:02d}/rank_buffer/{kind}.parquet' for offset in [1,8,15,22] for kind in ['signals','hold','forecasts']])
    verify(ROOT/'outputs/forward_risk/20260914T104245_763609Z',['features.parquet','predictions.parquet','coefficients.csv'])
    verify(ROOT/'outputs/shadow_execution/20260914T054517_770961Z',['model_2026.json'])
    assert file_hash(ROOT.parent/'AGENTS.md')=='82928cd87d095193da9adc18307823f6db6f4a425455146f8f4b45ba0c6e8ad6'
    sources=['src/monthly_risk_live.py','src/shadow_inputs.py','src/shadow_planner.py','src/shadow_account.py','src/manual_execution.py','src/mail_outbox.py','src/restore_quota.py',
        'scripts/manual_execution.py','scripts/shadow_execution.py','scripts/stage_monthly_risk_index.py','scripts/build_monthly_manual_observation.py','scripts/verify_monthly_manual_parity.py','scripts/verify_monthly_manual_inputs.py','scripts/demo_monthly_manual_lifecycle.py','scripts/seal_monthly_manual_integration.py',
        'tests/test_monthly_risk_manual.py','tests/test_monthly_risk_inputs.py','tests/test_shadow_execution.py','tests/test_shadow_inputs.py','tests/test_manual_execution.py','tests/test_manual_continuation.py','tests/test_mail_outbox.py',
        'configs/models/monthly_market_risk.json','configs/monthly_risk_manual.yaml','configs/monthly_predicted_risk_candidate.yaml','configs/manual_execution.yaml','docs/MONTHLY_RISK_MANUAL_INTEGRATION_20260914.md','README.md']
    for name in sources:
        dest=out/'sources'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,dest)
    for name in ['monthly_manual_full_tests','monthly_manual_parity','monthly_manual_market','monthly_manual_market_check','monthly_manual_metadata','monthly_manual_index','monthly_manual_index_probe','monthly_manual_index_verified','monthly_manual_observation','monthly_manual_observation_verified','monthly_manual_input_verification','monthly_manual_lifecycle','monthly_manual_append_check']:
        source=Path('/tmp')/(name+'.log');shutil.copy2(source,validation/(name+'.log'))
    shutil.copy2('/tmp/monthly_manual_full_tests.log',validation/'tests.log')
    shutil.copy2('/tmp/check_monthly_append.py',out/'sources/prepare_append_fixture.py')
    shutil.copy2('/tmp/probe_monthly_index.py',out/'sources/probe_index_anomaly.py')
    shutil.copy2(ROOT/'outputs/monthly_manual_parity/attempt04/verification.json',validation/'order_parity.json')
    record=dict(status='manual_adapter_verified',completed_at=datetime.now(timezone.utc).isoformat(),tests_passed=64,historical_accounts=4,economic_orders=540,restore_plans=45,
        guarded_missing_quote_account_days=2,corporate_action_pause_tested_separately=True,current_data_asof='2026-09-14',current_inputs_rebuilt=True,daily_append_exact=True,
        real_account_reconciled=False,real_account_db_created=False,mail_sent=False,orders_submitted=False,default_production_changed=False,new_independent_agent_review=False,
        source_vintage_externally_certified=False,consumed_parent_checks=consumed,source_sha256={n:file_hash(ROOT/n) for n in sources},root_agents_sha256=file_hash(ROOT.parent/'AGENTS.md'))
    record['artifact_sha256']={str(p.relative_to(out)):file_hash(p) for p in out.rglob('*') if p.is_file() and p.name!='completion.json'}
    (out/'completion.json').write_text(json.dumps(record,ensure_ascii=False,indent=2))
    for name,digest in record['artifact_sha256'].items():assert file_hash(out/name)==digest
    print(canonical(dict(status=record['status'],artifacts=len(record['artifact_sha256']),tests=64,orders=540,root_agents_unchanged=True,real_orders=False)))


if __name__=='__main__':main()

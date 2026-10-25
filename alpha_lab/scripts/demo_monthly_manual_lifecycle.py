"""可查看的合成账户演练：建仓、减仓、部分成交、重启、恢复、撤单及邮件草稿。"""
from pathlib import Path
import argparse,json,sqlite3,sys
import yaml
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tests')]
from test_monthly_risk_manual import MonthlyManualTests
from test_shadow_execution import preflight
from manual_execution import ManualAccount,bulletin
from shadow_planner import canonical


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    case=MonthlyManualTests();case.setUp();config=yaml.safe_load((ROOT/'configs/manual_execution.yaml').read_text())
    def save(name,obj):(a.output/name).write_text(canonical(obj))
    try:
        save('01_initial_plan.json',case.initial)
        plan=case.next_plan('2024-02-01',True);save('02_reduction_plan.json',plan)
        mail=bulletin(plan,{},'2024-02-01T21:05:00+08:00',config,backfill=True)
        (a.output/'briefing.eml').write_bytes(mail['eml']);(a.output/'briefing.txt').write_text(mail['body'])
        case.a.preflight(plan['plan_id'],preflight(plan));fills=[]
        for i,o in enumerate(plan['orders']):
            q=50 if i==0 else o['shares']
            fill=dict(execution_id=o['order_id']+'-demo',order_id=o['order_id'],date=o['execution_date'],shares=q,price=10.,fee=5.,evidence='synthetic demonstration; not real broker fill')
            case.a.record_fill(fill);fills.append(fill)
            if q<o['shares']:case.a.terminal(o['order_id'],'cancelled','synthetic remainder cancellation')
        save('03_actual_fixture_fills.json',fills);state=case.a.state();save('04_after_partial_reduction.json',state)
        case.a.close();case.a=ManualAccount(case.path);assert case.a.state()==state
        for f in fills:assert case.a.record_fill(f)['status']=='duplicate'
        plan=case.next_plan('2024-03-01');save('05_recovery_plan.json',plan)
        assert any(o['reason']=='targeted_restore' for o in plan['orders'])
        before=case.a.state()['memory']['monthly_risk']['quota'];case.fill_plan(plan,partial=True)
        save('06_after_partial_restore.json',case.a.state())
        case.a.close();case.a=ManualAccount(case.path)
        with sqlite3.connect(a.output/'synthetic_account.sqlite') as target:case.a.db.backup(target)
        audit=[dict(sequence=r[0],kind=r[1],payload=json.loads(r[2])) for r in case.a.db.execute('SELECT seq,kind,payload FROM audit ORDER BY seq')]
        save('audit.json',audit)
        save('verification.json',dict(synthetic_only=True,real_account_touched=False,mail_sent=False,orders_submitted=False,restart_equal=True,duplicate_fills_ignored=True,partial_sales_credit_only=True,
            restore_plans=sum(o['reason']=='targeted_restore' for o in plan['orders']),prior_credit=before['shares'],remaining_credit=case.a.state()['memory']['monthly_risk']['quota']['shares'],audit_events=len(audit)))
    finally:case.tearDown()


if __name__=='__main__':main()

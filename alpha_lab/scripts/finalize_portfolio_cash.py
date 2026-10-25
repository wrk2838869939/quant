"""封存主实验及后续单变量消融，记录失败边界而非晋升实盘。"""
from pathlib import Path
import argparse
import difflib
import json
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from build_selection_stability import digest,save


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--ablation',type=Path,required=True)
    args=p.parse_args();run,child=args.run,args.ablation
    assert not (run/'completion.json').exists() and not (child/'completion.json').exists()
    cfg=json.loads((run/'protocol.json').read_text())
    assert digest(ROOT.parent/'AGENTS.md')==cfg['root_agents_sha256']
    ancestor=Path(cfg['parent'])
    assert digest(ancestor/'completion.json')==cfg['parent_seal_sha256']
    for name,value in json.loads((ancestor/'completion.json').read_text())['artifact_sha256'].items():assert digest(ancestor/name)==value
    primary=json.loads((run/'verification.json').read_text());secondary=json.loads((child/'verification.json').read_text())
    assert primary['accounts']==96 and primary['parent_parities']==16 and len(primary['account_prefixes'])==3
    assert secondary['accounts']==18 and secondary['controls']==2 and len(secondary['prefixes'])==2
    assert secondary['recovery_events']==99 and secondary['five_slots_below60pct']==29
    sources=['scripts/finalize_portfolio_cash.py','scripts/analyze_portfolio_cash.py',
        'scripts/verify_portfolio_cash.py','scripts/verify_cash_reentry.py','scripts/plot_portfolio_cash.py',
        'scripts/run_portfolio_cash_tests.py','scripts/run_selection_stability.py','scripts/audit_monthly_portfolio.py',
        'scripts/build_selection_stability.py','scripts/analyze_selection_stability.py',
        'src/portfolio.py','src/selection_first.py','src/retail_rebalance.py',
        'tests/test_portfolio_cash.py','tests/test_cash_reentry.py','tests/test_causal_portfolio_adversarial.py',
        'docs/PORTFOLIO_CASH_VALIDATION_20260914.md','README.md']
    for folder,count,test in [(child,18,32),(run,96,31)]:
        protocol=json.loads((folder/'protocol.json').read_text())
        for name,value in protocol['source_sha256'].items():assert digest(ROOT/name)==value
        assert not list((folder/'accounts').glob('*/failure.json'))
        done=json.loads((folder/'execution_complete.json').read_text());assert done['accounts']==count
        for name in set(sources+list(protocol['source_sha256'])):
            target=folder/'sources'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,target)
        shutil.copy2(ROOT/'scripts/run_portfolio_cash_tests.py',folder/'run_tests.py')
        log=folder/('reentry_tests.log' if folder==child else 'cash_tests.log')
        text=log.read_text();assert f'Ran {test} tests' in text and text.rstrip().endswith('OK')
        if folder==child:
            (folder/'engine_changes.diff').write_text(''.join(difflib.unified_diff(
                (ROOT/'src/portfolio_cash_engine.py').read_text().splitlines(True),
                (ROOT/'src/portfolio_cash_reentry_engine.py').read_text().splitlines(True),
                fromfile='cash_engine',tofile='recovery_entry_engine')))
        else:
            save(folder/'ablation_receipt.json',dict(path=str(child.resolve()),completion_sha256=digest(child/'completion.json')))
            save(folder/'deployment_decision.json',dict(promoted=False,production_changed=False,
                real_orders=False,email_sent=False,reason='收益保留与交易频率未通过；恢复权限和恢复旧仓存在结构差异',
                next_experiment='只恢复本次风控实际减掉且仍合格的旧仓；资金与整手约束下的定向恢复，不默认有效'))
        hashes={str(f.relative_to(folder)):digest(f) for f in sorted(folder.rglob('*')) if f.is_file()}
        save(folder/'completion.json',dict(status='complete',accounts=count,trades=done['trades'],tests=test,
            overlapping_test_suites=True,independent_oos=False,live_approved=False,artifact_sha256=hashes))
        for name,value in hashes.items():assert digest(folder/name)==value
        print(folder.name,'sealed',len(hashes),'artifacts',done['trades'],'trades')


if __name__=='__main__':main()

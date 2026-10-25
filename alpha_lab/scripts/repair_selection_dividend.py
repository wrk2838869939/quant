"""用发行人实施公告修复中煤能源漏记的年度分红，保留失败及旧输入。"""
from datetime import datetime, timezone
from pathlib import Path
import argparse
import json
import shutil
import sys

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from build_selection_stability import digest, save


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--run', type=Path, required=True)
    run = p.parse_args().run; cfg = yaml.safe_load((run / 'config.yaml').read_text())
    archive = run / 'attempts/incomplete_601898_dividend'; archive.mkdir(parents=True)
    evidence = run / 'evidence/601898_20240813_2024-029.pdf'
    assert evidence.read_bytes().startswith(b'%PDF')
    # 来源为已人工核对第1、2页的实施公告；不从价格缺口推算未知分红。
    row_date = pd.Timestamp('2024-08-20'); code = '601898.SH'
    path = run / 'inputs/normalized_actions.parquet'; actions = pd.read_parquet(path)
    mask = actions.ts_code.eq(code) & actions.ex_date.eq(row_date)
    assert mask.sum() == 1 and actions.loc[mask, 'cash_div_tax'].iloc[0] == .113
    assert actions.loc[mask, 'stk_div'].iloc[0] == 0
    assert actions.loc[mask, 'known_date'].iloc[0] == pd.Timestamp('2024-08-13')
    audited = []
    completed = list((run / 'accounts').glob('*/summary.json'))
    reused = list((ROOT / cfg['startup_parent'] / 'accounts').glob('*_fresh_once/summary.json'))
    for summary_path in completed + reused:
        daily = pd.read_parquet(summary_path.parent / 'daily.parquet')
        holdings = pd.read_parquet(summary_path.parent / 'holdings.parquet')
        at_record = holdings.loc[holdings.ts_code.eq(code) & holdings.date.eq(pd.Timestamp('2024-08-19'))]
        assert at_record.empty, summary_path
        audited.append(dict(account=summary_path.parent.name, source='current' if summary_path in completed else 'parent',
                            record_date_in_window=pd.Timestamp('2024-08-19') in daily.index, entitled_shares=0))
    pd.DataFrame(audited).to_csv(archive / 'completed_account_exposure.csv', index=False)
    names = ['inputs/normalized_actions.parquet', 'execution_input_completion.json', 'execution_manifest.json', 'execution.log']
    for name in names: shutil.copy2(run / name, archive / Path(name).name)
    old_digest = digest(path); actions.loc[mask, 'cash_div_tax'] = .555
    actions.to_parquet(path, index=False)
    original = pd.read_parquet(archive / 'normalized_actions.parquet')
    pd.testing.assert_frame_equal(actions.loc[~mask], original.loc[~mask], check_exact=True)
    market = pd.read_parquet(run / 'inputs/market.parquet')
    x = market.loc[market.ts_code.eq(code)].set_index('date')
    residual = float(x.loc['2024-08-19', 'close'] - .555 - x.loc['2024-08-20', 'pre_close'])
    assert abs(residual) <= .0100001
    failures = []
    for failure in list((run / 'accounts').glob('*/failure.json')):
        details = json.loads(failure.read_text())
        assert '2024-08-20 601898.SH' in details['message']
        failures.append(failure.parent.name)
        shutil.move(str(failure.parent), str(archive / failure.parent.name))
    assert len(failures) == 2
    # 重新跑父账户一致性；其旧回执也保留。
    shutil.move(str(run / 'parent_parity.json'), str(archive / 'parent_parity.json'))
    receipt = json.loads((run / 'execution_input_completion.json').read_text())
    receipt['input_sha256']['inputs/normalized_actions.parquet'] = digest(path)
    receipt['normalized_action_correction'] = 'dividend_correction.json'
    save(run / 'execution_input_completion.json', receipt)
    manifest = json.loads((run / 'execution_manifest.json').read_text())
    manifest['inputs_receipt_sha256'] = digest(run / 'execution_input_completion.json')
    manifest['input_revision'] = 'issuer_verified_601898_20240820_dividend'
    save(run / 'execution_manifest.json', manifest)
    save(run / 'dividend_correction.json', dict(time=datetime.now(timezone.utc).isoformat(), code=code,
        ex_date='2024-08-20', known_date='2024-08-13', original=.113, annual_component=.442, special_component=.113, corrected_total=.555,
        source='https://www.chinacoalenergy.com/cms_files/jcms1/web2/site/attach/0/5c706ac8437542fc8b925e9dc0c84260.pdf',
        evidence_sha256=digest(evidence), source_script_sha256=digest(Path(__file__)),
        original_actions_sha256=old_digest, corrected_actions_sha256=digest(path), reference_price_rounding_residual=residual,
        completed_accounts_audited=len(completed), reused_accounts_audited=len(reused), zero_entitled_shares_in_all_completed=True,
        failed_accounts_preserved=failures, unchanged_successful_results_reused=True,
        input_revision_preserved_in=str(archive.relative_to(run))))
    print('issuer dividend corrected; rerun', failures, flush=True)


if __name__ == '__main__': main()

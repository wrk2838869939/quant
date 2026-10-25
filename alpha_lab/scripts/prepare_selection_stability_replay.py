"""从已校正研究输入创建空结果目录，完整重跑无需重演历史失败。"""
from pathlib import Path
import argparse
import json
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from build_selection_stability import digest, save


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    args = p.parse_args(); src, out = args.source, args.output
    done = json.loads((src / 'execution_complete.json').read_text())
    assert done['new_accounts'] == 183 and done['all_reconciled']
    hashes = json.loads((src / 'signal_hashes.json').read_text())
    inputs = json.loads((src / 'execution_input_completion.json').read_text())
    for name, value in hashes.items(): assert digest(src / name) == value
    for name, value in inputs['input_sha256'].items(): assert digest(src / name) == value
    out.mkdir(parents=True, exist_ok=False)
    directories = ['signals', 'schedules', 'inputs', 'new_execution_raw', 'evidence',
                   'attempts/before_entry_exit_alignment']
    for name in directories: shutil.copytree(src / name, out / name)
    names = ['config.yaml', 'manifest.json', 'PROTOCOL.md', 'execution_manifest.json', 'execution_codes.csv',
             'signal_completion.json', 'signal_hashes.json', 'execution_input_completion.json', 'risk.parquet',
             'rank_entry_alignment.json', 'ENTRY_EXIT_ALIGNMENT.md', 'dividend_correction.json',
             'candidate_scope_correction.json', 'CANDIDATE_SCOPE_CORRECTION.md']
    for name in names: shutil.copy2(src / name, out / name)
    assert not (out / 'accounts').exists() and not (out / 'parent_parity.json').exists()
    for name, value in hashes.items(): assert digest(out / name) == value
    for name, value in inputs['input_sha256'].items(): assert digest(out / name) == value
    save(out / 'replay_source.json', dict(source=str(src.resolve()), source_execution_sha256=digest(src / 'execution_complete.json'),
        source_signal_hashes_sha256=digest(src / 'signal_hashes.json'), corrected_inputs_verified=True,
        empty_accounts=True, note='仅复制正确输入，所有183账户及3个父账户须重新执行；旧失败保留在源目录'))
    print(out, flush=True)


if __name__ == '__main__': main()

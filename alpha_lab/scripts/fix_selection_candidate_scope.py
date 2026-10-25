"""修正先过滤再截断导致的隐式扩池；按原Top20定池后施加入场禁止。"""
from datetime import datetime, timezone
from pathlib import Path
import argparse
import json
import shutil
import sys

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
from build_selection_stability import digest, save
from selection_stability import buffered_holds


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--run', type=Path, required=True)
    run = p.parse_args().run; cfg = yaml.safe_load((run / 'config.yaml').read_text())
    archive = run / 'attempts/implicit_candidate_expansion'; archive.mkdir(parents=True)
    receipt = json.loads((run / 'execution_manifest.json').read_text())
    scope = set(pd.read_csv(run / 'execution_codes.csv').ts_code)
    rows = []
    for day in cfg['calendar_days']:
        original_hold = pd.read_parquet(run / f'schedules/calendar_{day:02d}/baseline/hold.parquet')
        for policy in ['rank_buffer', 'combined']:
            folder = run / f'schedules/calendar_{day:02d}/{policy}'
            current = pd.read_parquet(folder / 'signals.parquet')
            original = pd.read_parquet(run / 'attempts/before_entry_exit_alignment' / f'{day:02d}_{policy}_signals.parquet')
            _, _, exits = buffered_holds(original, original_hold, cfg['rank_buffer'], cfg['rank_confirmations'], cfg['max_positions'])
            fixed = pd.DataFrame(np.nan, index=original.index, columns=original.columns)
            for date, scores in original.iterrows():
                pool = scores.dropna().sort_values(ascending=False, kind='stable').head(20).index
                assert set(pool) <= scope
                permitted = pool[~exits.loc[date, pool].to_numpy()]
                fixed.loc[date, permitted] = scores.loc[permitted]
            previous_scope = set().union(*(set(row.dropna().nlargest(20).index) for _, row in current.iterrows()))
            rows.append(dict(calendar=day, policy=policy, previous_codes_without_execution_data=len(previous_scope-scope),
                             corrected_all_candidates_within_original_top20=True))
            shutil.copy2(folder / 'signals.parquet', archive / f'{day:02d}_{policy}_signals.parquet')
            fixed.to_parquet(folder / 'signals.parquet')
    for name in ['signal_hashes.json', 'execution_manifest.json', 'execution_complete.json', 'execution.log',
                 'analysis.log', 'verification.json', 'verification.log', 'process_analysis.log']:
        if (run / name).exists(): shutil.move(str(run / name), str(archive / name))
    if (run / 'analysis').exists(): shutil.move(str(run / 'analysis'), str(archive / 'analysis'))
    invalidated = []
    for case in receipt['cases']:
        if case['policy'] in ['rank_buffer', 'combined']:
            folder = run / 'accounts' / case['name']
            shutil.move(str(folder), str(archive / case['name']))
            invalidated.append(case['name'])
    hashes = json.loads((archive / 'signal_hashes.json').read_text())
    for name in hashes: hashes[name] = digest(run / name)
    save(run / 'signal_hashes.json', hashes)
    receipt.update(runner_sha256=digest(ROOT / 'scripts/run_selection_stability.py'),
                   signal_hashes_sha256=digest(run / 'signal_hashes.json'), candidate_scope_revision='top20_before_entry_denial')
    save(run / 'execution_manifest.json', receipt)
    save(run / 'candidate_scope_correction.json', dict(time=datetime.now(timezone.utc).isoformat(),
        reason='先入场屏蔽再head20会隐式扩池；纠正为原排序head20后入场过滤，不按现有数据删股票',
        source_sha256=digest(Path(__file__)), invalidated_accounts=invalidated, invalidated_count=len(invalidated),
        matrices=rows, strategy_parameters_unchanged=True, previous_analysis_invalid=True))
    (run / 'CANDIDATE_SCOPE_CORRECTION.md').write_text('''# 候选范围修正与临时结果作废

最终覆盖核验发现，入场禁止信号先置空再由引擎取head20，隐式补入了原第20名之后的股票，其中406只不在原执行数据并集。
原引擎会将完全缺该股票报价的候选记录为no_known_closing_reference，这不能作为真实执行数据充分的证据。原临时rank_buffer/combined结果和分析作废，已全部移入attempts/implicit_candidate_expansion。
按照原引擎先排名Top20、再进行入场合格性检查的契约，现明确先定原Top20，再屏蔽当期排名退出股票，留空不向后扩池。这个规则与数据是否已下载无关。
原参数、完整截面排名、持有判断、基线/纯平滑分支与风险模型均不改。新增全局候选代码和风险覆盖硬检查；106个受影响账户全部重跑，旧临时通过标志不作准入依据。
''')
    print('invalidated and scheduled for replay', len(invalidated), flush=True)


if __name__ == '__main__': main()

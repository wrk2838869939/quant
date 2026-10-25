"""账户回放前消除排名退出后仍能同月补买该股的规则冲突，保留修正前件。"""
from datetime import datetime, timezone
from pathlib import Path
import argparse
import shutil
import sys

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
from build_selection_stability import digest, save
from selection_stability import buffered_holds


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--run', type=Path, required=True)
    run = p.parse_args().run
    assert not (run / 'execution_manifest.json').exists(), '必须在观察任何新账户结果前完成'
    assert not (run / 'rank_entry_alignment.json').exists()
    cfg = yaml.safe_load((run / 'config.yaml').read_text())
    archive = run / 'attempts/before_entry_exit_alignment'; archive.mkdir(parents=True)
    shutil.copy2(run / 'signal_hashes.json', archive / 'signal_hashes.json')
    rows = []
    for day in cfg['calendar_days']:
        original_hold = pd.read_parquet(run / f'schedules/calendar_{day:02d}/baseline/hold.parquet')
        for policy in ['rank_buffer', 'combined']:
            folder = run / f'schedules/calendar_{day:02d}/{policy}'
            signals = pd.read_parquet(folder / 'signals.parquet')
            _, _, exits = buffered_holds(signals, original_hold, cfg['rank_buffer'], cfg['rank_confirmations'], cfg['max_positions'])
            shutil.copy2(folder / 'signals.parquet', archive / f'{day:02d}_{policy}_signals.parquet')
            signals.where(~exits).to_parquet(folder / 'signals.parquet')
            rows.append(dict(calendar=day, policy=policy, excluded_current_entry_cells=int((signals.notna() & exits).sum().sum())))
    (run / 'ENTRY_EXIT_ALIGNMENT.md').write_text('''# 账户结果产生前的规则一致性修正

代码检查发现：原引擎在卖空某股后清除其退出标记，补位时仍可从同月排名表买回。
新排名退出规则仍可能让当日排名11—20名的股票有入场资格，故存在“排名要求退出，但随后又补入”的冲突。
在任何新账户回放前，将rank_buffer/combined当期确认排名退出的股票同时从当期新入场信号中屏蔽。原始排名仍用于计算退出，屏蔽后不重新计算排名/确认。
不改基线、平滑单独政策、原引擎、股数/现金/风险规则。研究解释应称排名缓冲的进入/持有一致约束，不将其全部贡献机械归为卖出效果。
修正前信号及散列保存在attempts/before_entry_exit_alignment；候选执行数据继续准备原并集，额外下载不影响交易选择。
''')
    save(run / 'rank_entry_alignment.json', dict(time=datetime.now(timezone.utc).isoformat(), before_any_new_account=True,
        source_sha256=digest(Path(__file__)), note_sha256=digest(run / 'ENTRY_EXIT_ALIGNMENT.md'), rows=rows))
    import json
    hashes = json.loads((archive / 'signal_hashes.json').read_text())
    for name in hashes: hashes[name] = digest(run / name)
    save(run / 'signal_hashes.json', hashes)
    print('entry/exit alignment frozen before accounts', flush=True)


if __name__ == '__main__': main()

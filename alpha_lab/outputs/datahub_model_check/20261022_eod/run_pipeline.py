"""调用既有日更、构建和模型入口；本次只生成日期配置与运行记录。"""
from pathlib import Path
import json
import os
import subprocess
import sys
import yaml

ROOT = Path('/root/autodl-tmp')
OUT = Path(__file__).resolve().parent
BASE = ROOT / 'alpha_lab/experiments/sector_live_v1'
ENV = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
RELEASE = 'datahub/releases/selection_live_20261022_eod'
CONFIG = 'alpha_lab/experiments/sector_live_v1/config_20261022_eod.yaml'
record = {'asof': '2026-10-22', 'status': 'running', 'steps': []}


def save():
    (OUT / 'pipeline.json').write_text(json.dumps(record, ensure_ascii=False, indent=2))


def run(name, args):
    record['current_step'] = name
    save()
    print('START', name, flush=True)
    with (OUT / f'{name}.log').open('w') as log:
        result = subprocess.run([sys.executable, *args], cwd=ROOT, env=ENV, stdout=log, stderr=subprocess.STDOUT)
    record['steps'].append({'step': name, 'args': args, 'returncode': result.returncode})
    if result.returncode:
        record['status'] = 'failed'
        save()
        raise RuntimeError(f'{name} failed; inspect {name}.log')
    save()
    print('COMPLETE', name, flush=True)


run('update', ['alpha_lab/experiments/sector_live_v1/update_market.py', '--config', CONFIG])
update = ROOT / (BASE / 'LATEST_UPDATE').read_text().strip()
um = json.loads((update / 'manifest.json').read_text())
assert um['asof'] == record['asof'] and um['status'] == 'complete'
record['update'] = str(update)
cfg = yaml.safe_load((BASE / 'dataset_config_20261021_eod.yaml').read_text())
cfg.update(release=RELEASE, content_id=um['content_id'], end=record['asof'])
dc = BASE / 'dataset_config_20261022_eod.yaml'
dc.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False))
prefix = ['datahub/scripts/frozen_research.py', 'run', '--release', RELEASE]
run('build_dataset', prefix + ['alpha_lab/scripts/build_sector_dataset.py', '--config', str(dc)])
lines = (OUT / 'build_dataset.log').read_text().splitlines()
dataset = next(line.removeprefix('COMPLETE ') for line in reversed(lines) if line.startswith('COMPLETE '))
assert json.loads((Path(dataset) / 'manifest.json').read_text())['status'] == 'complete'
record['dataset'] = dataset
run('scan_sectors', prefix + ['alpha_lab/experiments/sector_live_v1/scan_sectors.py', '--dataset', dataset, '--config', CONFIG])
scan = ROOT / (BASE / 'LATEST_SCAN').read_text().strip()
assert json.loads((scan / 'summary.json').read_text())['asof'] == record['asof']
record['scan'] = str(scan)
run('scan_stocks', prefix + ['alpha_lab/experiments/sector_live_v1/scan_stocks.py', '--scan', str(scan)])
model = ROOT / (BASE / 'LATEST_CANDIDATES').read_text().strip()
summary = json.loads((model / 'summary.json').read_text())
assert summary['asof'] == record['asof'] and not summary['annual_sector_gate_enabled']
record.update(status='complete', model=str(model), summary=summary)
save()
print(json.dumps(record, ensure_ascii=False), flush=True)

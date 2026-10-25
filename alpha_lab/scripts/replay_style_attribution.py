#!/usr/bin/env python3
"""修复执行元数据后复制冻结匹配输入到新目录回放，不重新匹配或选型。"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from improve_monthly_portfolio import digest
from ledger import ExperimentLedger
from run_investability_followup import evaluate
from audit_style_attribution import report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepared", type=Path, required=True)
    ap.add_argument("--reason", required=True)
    args = ap.parse_args()
    old = args.prepared
    manifest = json.loads((old / "manifest.json").read_text())
    if manifest["status"] not in {"failed", "complete"}:
        raise ValueError("仅对已完成或明确失败的记录创建修复回放")
    fixed = {**manifest["input_hashes"], **manifest["config_hashes"]}
    for name, expected in fixed.items():
        if digest(old / name) != expected:
            raise ValueError(f"冻结产物已修改：{name}")
    output = old.parent / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output.mkdir(exist_ok=False)
    extras = ["requested_experiment.yaml", "coverage.csv", "target_exposures.csv",
              "neutralization_status.csv", "dsr_synthetic_verification.json"]
    for name in sorted(set(fixed) | {name for name in extras if (old / name).exists()}):
        shutil.copy2(old / name, output / name)
    changed = {name: dict(before=expected, after=digest(ROOT / name))
        for name, expected in manifest["source_hashes"].items() if digest(ROOT / name) != expected}
    manifest.update(status="signals_frozen", replay_of=str(old), replay_reason=args.reason,
        source_changes=changed,
        source_hashes={name: digest(ROOT / name) for name in manifest["source_hashes"]})
    manifest["source_hashes"][str(Path(__file__).resolve().relative_to(ROOT))] = digest(Path(__file__))
    for key in ["results", "error_type", "error"]:
        manifest.pop(key, None)
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    ExperimentLedger().log("style_attribution_replay", "fixed_control_replay", "2022-2023 reused development",
        np.nan, np.nan, np.nan, np.nan, params=dict(output=str(output)), verdict="EXECUTION_FIX_REPLAY",
        experiment_id=str(output), parent_experiment=str(old), source_hashes=manifest["source_hashes"],
        input_hashes=manifest["input_hashes"], selection_note=args.reason)
    print(f"冻结名单修复回放 {output}", flush=True)
    evaluate(output)
    report(output)


if __name__ == "__main__":
    main()

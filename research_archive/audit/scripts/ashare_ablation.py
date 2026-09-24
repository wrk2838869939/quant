# 历史实验脚本归档：运行依赖原研究环境及未发布输入。
#!/usr/bin/env python3
"""A股主板 消融实验: 选股 × 择时 的贡献分解 (相同止盈止损)。

4 个变体:
  全策略   = 选股(LGBM) + 择时(regime)
  无选股   = 随机选股    + 择时(regime)
  无择时   = 选股(LGBM) + 始终持仓
  无选股+无择时 = 随机选股 + 始终持仓 (纯beta随机篮子)
"""
import sys, warnings, yaml
from pathlib import Path
import pandas as pd, numpy as np
warnings.filterwarnings("ignore")

PROJECT_ROOT = Path("a_share_v17")
sys.path.insert(0, str(PROJECT_ROOT))
from src.rotation.sector_backtest import SectorRotationBacktest


def resolve_paths(cfg):
    for key, val in cfg.get("paths", {}).items():
        if key in ("daily_features", "sector_map"):
            cfg["paths"][key] = Path(val)
        else:
            p = Path(val)
            if not p.is_absolute():
                p = PROJECT_ROOT / p
            p.mkdir(parents=True, exist_ok=True)
            cfg["paths"][key] = p
    return cfg


def run_variant(cfg, no_selection, no_timing, name):
    cfg = {k: v for k, v in cfg.items()}
    cfg["ablation"] = {"no_selection": no_selection, "no_timing": no_timing}
    cfg["paths"] = {k: Path(v) if isinstance(v, str) else v for k, v in cfg["paths"].items()}
    engine = SectorRotationBacktest(cfg)
    dr, dt = engine.run(
        train_start="2025-07-01", train_end="2026-02-28",
        test_start="2026-03-01", test_end="2026-07-31",
        output_name=name,
    )
    return dr


def metrics(dr):
    r = dr["ret"].dropna()
    cum = (1 + r).prod() - 1
    eq = (1 + r).cumprod()
    mdd = ((eq.cummax() - eq) / eq.cummax()).max()
    vol = r.std() * np.sqrt(252)
    sharpe = r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else 0
    sortino = r.mean() / r[r < 0].std() * np.sqrt(252) if (r < 0).any() and r[r < 0].std() > 0 else 0
    calmar = cum / mdd if mdd > 0 else 0
    return cum, mdd, vol, sharpe, sortino, calmar


def main():
    cfg = yaml.safe_load(open(PROJECT_ROOT / "configs/sector_rotation.yaml"))
    cfg = resolve_paths(cfg)

    variants = [
        ("全策略(选股+择时)", False, False, "ablation_full"),
        ("无选股(随机+择时)", True, False, "ablation_nosel"),
        ("无择时(选股+始终持仓)", False, True, "ablation_notiming"),
        ("无选股+无择时(纯beta)", True, True, "ablation_none"),
    ]
    results = {}
    for name, ns, nt, outname in variants:
        print(f"\n{'='*60}\n  {name}\n{'='*60}")
        dr = run_variant(cfg, ns, nt, outname)
        results[name] = metrics(dr)

    print("\n\n=== 消融实验结果 (3-7月, 相同止盈止损) ===")
    print(f"  {'变体':>22s} {'收益':>9s} {'MDD':>7s} {'波动':>7s} {'Sharpe':>7s} {'Sortino':>7s} {'Calmar':>7s}")
    for name, (cum, mdd, vol, sh, so, cal) in results.items():
        print(f"  {name:>22s} {cum:>+9.1%} {mdd:>7.1%} {vol:>7.0%} {sh:>7.2f} {so:>7.1f} {cal:>7.2f}")

    # 贡献分解
    print("\n=== 贡献分解 ===")
    full = results["全策略(选股+择时)"][0]
    nosel = results["无选股(随机+择时)"][0]
    notiming = results["无择时(选股+始终持仓)"][0]
    none = results["无选股+无择时(纯beta)"][0]
    print(f"  纯beta(无选股+无择时)收益: {none:+.1%}")
    print(f"  选股贡献 (全策略 - 无选股): {full - nosel:+.1%}")
    print(f"  择时贡献 (全策略 - 无择时): {full - notiming:+.1%}")
    print(f"  选股+择时交互: {full - none - (full-nosel) - (full-notiming):+.1%}")


if __name__ == "__main__":
    main()

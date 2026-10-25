#!/usr/bin/env python3
"""按冻结权重重算 OOS 十分组 (前瞻窗截断至 2023-12-31 内, 修正 2023 行触碰 TEST 的问题)。"""
import sys
from pathlib import Path
import pandas as pd

ROOT = Path("/root/autodl-tmp/alpha_lab")
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from walkforward_monthly_oos import (  # noqa: E402
    CFG, H, build_wide_and_factors, synth_signals)
from investable import investable_signals  # noqa: E402

RUN = ROOT / "outputs/walkforward_oos/20260919T052924_461677Z"
wide, facs = build_wide_and_factors()
all_dates = facs["lit:rev20"].index
usable = all_dates[all_dates <= pd.Timestamp("2023-12-31")]
cutoff = usable[max(0, len(usable) - 1 - H)]
fwd20 = wide["hclose"].shift(-H) / wide["hclose"] - 1.0

panel = ROOT / "data/panel/daily_panel.parquet"
basics = pd.read_parquet(ROOT / "data/raw/stock_basic_all.parquet")
for c in ("list_date", "delist_date"):
    basics[c] = pd.to_datetime(basics[c], errors="coerce")
names = pd.read_parquet(ROOT / "data/raw/namechange_all.parquet")
names["start"] = pd.to_datetime(names.start_date, errors="coerce")
names["end"] = pd.to_datetime(names.end_date, errors="coerce")
st = names[names.name.str.contains("ST", case=False, na=False)][["ts_code", "start", "end"]]
p_half = dict(CFG["portfolio"], capital=CFG["portfolio"]["capital"] * 0.5)

rows = []
for Y in range(2019, 2024):
    y0, y1 = pd.Timestamp(f"{Y}-01-01"), pd.Timestamp(f"{Y}-12-31")
    w = pd.read_csv(RUN / f"weights_{Y}.csv", index_col=0).iloc[:, 0].to_dict()
    yd = all_dates[(all_dates >= y0) & (all_dates <= y1)]
    grid = [d for i, d in enumerate(yd) if i % H == 0]
    grid_d = [d for d in grid if d <= cutoff]
    sig = synth_signals(facs, w, grid)
    mk = pd.read_parquet(panel, columns=["date", "ts_code", "close", "vol_share"],
                         filters=[("date", ">=", y0), ("date", "<=", y1)])
    close = mk.pivot(index="date", columns="ts_code", values="close").reindex(
        index=grid, columns=sig.columns)
    volume = mk.pivot(index="date", columns="ts_code", values="vol_share").reindex(
        index=grid, columns=sig.columns)
    _, allowed = investable_signals(sig, close, volume, basics, st, p_half, 1.05)
    q = sig.rank(axis=1, pct=True)
    elig = sig.notna() & allowed
    for g in range(10):
        mask = (q > g / 10) & (q <= (g + 1) / 10) & elig
        vals = fwd20.loc[grid_d][mask].stack().dropna()
        rows.append(dict(year=Y, decile=g + 1, mean_fwd20=float(vals.mean()), n=len(vals)))
    print(Y, "done", flush=True)
out = pd.DataFrame(rows)
out.to_csv(RUN / "deciles_fixed.csv", index=False)
print(out.pivot_table(index="year", columns="decile", values="mean_fwd20").round(4).to_string())

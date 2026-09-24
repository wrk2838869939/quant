# 历史实验脚本归档：运行依赖原研究环境及未发布输入。
"""标签泄露对选股 IC 的影响消融 (a_share_v18 配套验证)。

问题: v17→v18 修复训练标签泄露后回测收益大幅下滑, 但收益≠选股alpha。
本脚本直接度量: 同一预测日 T、同一特征, 仅训练标签口径不同(泄露 vs 诚实),
逐日截面 rank IC 配对对比。

口径:
  - 宇宙: cluster_mapping 400 池, 按板块各自训练 LGBM (与回测一致)
  - 特征: 生产配置 26 因子 (GBDT probe 选 top-10)
  - 泄露组: 训练池含最后5天(标签用到 T+1~T+4 收盘价, 实盘不可得)
  - 诚实组: 剔除最后5天 (v18 口径)
  - IC: 每日截面 pred vs future_ret 的 Spearman (5日=训练目标口径, 2日=实际持仓口径)

用法: python audit/scripts/ic_leak_ablation.py [--start 2026-04-01] [--end 2026-08-24]
"""
import argparse, sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(".")

# ── 模块级全局 (fork 继承, 供 worker 使用) ──
_G = {}


def _one_day(T):
    """单日双口径训练+预测+IC (并行 worker)。"""
    from lightgbm import LGBMRegressor
    from sklearn.ensemble import GradientBoostingRegressor
    dc, cc = _G["dc"], _G["cc"]
    sc, lp, feats, H = _G["sc"], _G["lp"], _G["feats"], _G["H"]
    by_sector, by_date = _G["by_sector"], _G["by_date"]
    day_rows = by_date[T]
    DROP_DAYS = _G.get("drop_days", [0, 5])
    preds = {d: {} for d in DROP_DAYS}
    for sec, gsec in by_sector.items():
        train = gsec[(gsec[dc] < T) &
                     (gsec[dc] >= T - pd.Timedelta(days=sc["train_lookback_days"]))]
        if len(train) < sc["per_sector_min_samples"]:
            continue
        today = day_rows[day_rows["sector"] == sec]
        if today.empty:
            continue
        for variant in DROP_DAYS:
            tr = train
            if variant > 0:
                pdates = np.sort(tr[dc].unique())
                if len(pdates) <= variant:
                    continue
                tr = tr[tr[dc] < pdates[-variant]]
            tr = tr.replace([np.inf, -np.inf], np.nan)
            for c in feats:
                tr[c] = tr[c].fillna(tr[c].median() if tr[c].notna().any() else 0)
            valid = tr["fwd5"].notna()
            tr = tr[valid]
            if len(tr) < 50:
                continue
            probe = GradientBoostingRegressor(n_estimators=50, max_depth=3,
                                              learning_rate=0.1, random_state=42)
            probe.fit(tr[feats].values, tr["fwd5"].values)
            top_idx = np.argsort(probe.feature_importances_)[-sc["feature_select_top"]:]
            sel = [feats[i] for i in top_idx]
            model = LGBMRegressor(
                n_estimators=lp["n_estimators"], max_depth=lp["max_depth"],
                learning_rate=lp["learning_rate"], num_leaves=lp["num_leaves"],
                min_child_samples=lp["min_samples_leaf"], subsample=lp["subsample"],
                colsample_bytree=lp["colsample_bytree"], reg_alpha=lp["reg_alpha"],
                reg_lambda=lp["reg_lambda"], random_state=lp["random_state"],
                n_jobs=1, verbose=-1)
            model.fit(tr[sel].values, tr["fwd5"].values)
            X = today[sel].fillna(0.0).values
            for code, p in zip(today[cc], model.predict(X)):
                preds[variant][code] = p
    row = {"date": T}
    for variant in _G.get("drop_days", [0, 5]):
        if len(preds[variant]) < 30:
            continue
        ps = pd.Series(preds[variant])
        sub = day_rows.set_index(cc).loc[ps.index]
        for hname, hcol in [("ic5", "fwd5"), ("ic2", "fwd2")]:
            v = sub[hcol].notna()
            if v.sum() >= 30 and ps[v].std() > 0:
                row[f"{hname}_{variant}"] = ps[v].corr(sub.loc[v, hcol], method="spearman")
        row[f"n_{variant}"] = len(preds[variant])
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2026-04-01")
    ap.add_argument("--end", default="2026-08-24")
    ap.add_argument("--label-days", type=int, default=5, help="前向标签天数(=config data.hold_days)")
    args = ap.parse_args()

    import yaml
    from lightgbm import LGBMRegressor
    from sklearn.ensemble import GradientBoostingRegressor

    cfg = yaml.safe_load(open(ROOT / "a_share_v18/configs/sector_rotation_210k.yaml"))
    sc = cfg["stock_selection"]
    feats_cfg = sc["features"]
    lp = cfg["selector"]["lgbm"]
    H = args.label_days

    # ── 数据 ──
    df = pd.read_parquet(cfg["paths"]["daily_features"])
    dc, cc, cl = cfg["data"]["date_col"], cfg["data"]["code_col"], cfg["data"]["close_col"]
    df[dc] = pd.to_datetime(df[dc])
    df[cc] = df[cc].astype(str).str.zfill(6)
    df = df.sort_values([cc, dc]).reset_index(drop=True)

    # 400 池 (与回测 restrict_to_pool 一致)
    cm = pd.read_parquet(ROOT / "a_share_v15/data/processed/cluster_mapping_history.parquet")
    pool = set(cm["code"].astype(str).str.zfill(6).unique())
    sw = pd.read_parquet(cfg["paths"]["sector_map"])
    sw["stock_code"] = sw["stock_code"].astype(str).str.zfill(6)
    code_to_sector = dict(zip(sw["stock_code"], sw["sw_l1_name"]))
    df = df[df[cc].isin(pool)].copy()
    df["sector"] = df[cc].map(code_to_sector)

    # 前向收益 (5日=训练标签口径, 2日=实际T+2持仓口径)
    g = df.groupby(cc)[cl]
    df["fwd5"] = g.shift(-H) / df[cl] - 1
    df["fwd2"] = g.shift(-2) / df[cl] - 1

    feats = [c for c in feats_cfg if c in df.columns]
    print(f"特征 {len(feats)}/{len(feats_cfg)}, 池 {len(pool)} 只")

    test_dates = sorted(df[(df[dc] >= args.start) & (df[dc] <= args.end)][dc].unique())
    print(f"测试区间: {args.start}~{args.end} ({len(test_dates)} 交易日)")

    by_sector = {s: gdf for s, gdf in df.groupby("sector")}
    by_date = {d: gdf for d, gdf in df.groupby(dc)}
    _G.update(dc=dc, cc=cc, sc=sc, lp=lp, feats=feats, H=H,
              by_sector=by_sector, by_date=by_date,
              drop_days=[0, 1, 2, 3, 5])

    from concurrent.futures import ProcessPoolExecutor
    records = []
    with ProcessPoolExecutor(max_workers=8) as ex:
        for di, row in enumerate(ex.map(_one_day, test_dates, chunksize=1)):
            records.append(row)
            if (di + 1) % 10 == 0:
                print(f"  {di+1}/{len(test_dates)}", flush=True)

    r = pd.DataFrame(records)
    out = ROOT / "audit/outputs/ic_leak_ablation.csv"
    r.to_csv(out, index=False)

    print("\n=== 逐日截面 rank IC vs 训练剔除天数 (0=泄露口径, 5=v18诚实口径) ===")
    for hname, hcol in [("5日前向", "ic5"), ("2日前向", "ic2")]:
        print(f"  [{hname}]")
        for dnum in [0, 1, 2, 3, 5]:
            col = f"{hcol}_{dnum}"
            if col not in r.columns:
                continue
            v = r[col].dropna()
            print(f"    剔除{dnum}天: IC均值 {v.mean():+.4f} | ICIR {v.mean()/v.std():+.3f} | "
                  f"t={v.mean()/v.std()*np.sqrt(len(v)):+.2f} | IC>0占比 {(v>0).mean():.0%} | n={len(v)}")
    d0 = (r["ic5_0"] - r["ic5_5"]).dropna()
    print(f"  配对差(0天-5天, ic5): {d0.mean():+.4f}, t={d0.mean()/d0.std()*np.sqrt(len(d0)):+.2f}")
    print(f"\n保存: {out}")


if __name__ == "__main__":
    main()

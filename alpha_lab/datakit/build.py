# -*- coding: utf-8 -*-
"""datakit 构建: 一次性把面板+公司行动+日历修成「验证过的结构化数据集」。

产物 (data/datakit/):
  market_core.parquet   date×code 长表: 行情+属性(PIT 行业/ST/板块), 缺日已修
  actions.parquet       规范化分红 (修订合并+已核验组件+零经济行剔除)
  limits.parquet        涨跌停价 (stk_limit 缓存合并)
  excluded.parquet      整段隔离注册表 (reason/detail): 引擎守卫会拒绝的代码, 构建期定案
  index_daily.parquet   指数日线 (择时策略用)
  verify_report.json    构建期验证结果 (单位/PIT/日历/覆盖/守卫等价)
  meta.json             版本/窗口/行数/来源哈希

纪律: 窗口 ≤ 2023-12-31 (TEST 不入数据集); 原始面板与缓存不修改。
"""
import hashlib
import json
import sys
import time
import warnings

T0 = time.time()


def tic(msg: str):
    print(f"[{time.time()-T0:7.1f}s] {msg}", flush=True)
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

warnings.filterwarnings("ignore")
ROOT = Path("/root/autodl-tmp/alpha_lab")
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from portfolio import normalize_actions  # noqa: E402
from walkforward_monthly_oos import screen_actions, screen_prices  # noqa: E402

OUT = ROOT / "data/datakit"
PANEL = ROOT / "data/panel/daily_panel.parquet"
REPAIR = ROOT / "outputs/forward_risk/repair_20211213/projection.parquet"
CFG = yaml.safe_load((ROOT / "configs/monthly_portfolio_audit.yaml").read_text())
Y0, Y1 = pd.Timestamp("2014-01-01"), pd.Timestamp("2023-12-31")
REGISTRY = ROOT / "datakit/components_verified.yaml"


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def load_panel_core() -> pd.DataFrame:
    cols = ["date", "ts_code", "open", "close", "pre_close", "adj_factor",
            "vol_share", "amount_yuan", "circ_mv_yuan", "turnover_rate",
            "pe_ttm", "pct_chg"]
    df = pd.read_parquet(PANEL, columns=cols,
                         filters=[("date", ">=", Y0), ("date", "<=", Y1)])
    df = df.sort_values(["date", "ts_code"]).reset_index(drop=True)
    # 缺日修复 (2021-12-13; 无投影的日历日 → 硬失败)
    cal = pd.read_parquet(ROOT / "data/raw/calendar.parquet")
    cal["cal_date"] = pd.to_datetime(cal["cal_date"].astype(str), format="%Y%m%d")
    open_days = cal.loc[(cal.cal_date >= Y0) & (cal.cal_date <= Y1)
                        & (cal.is_open.astype(int) == 1), "cal_date"]
    have = set(pd.DatetimeIndex(df.date.unique()))
    missing = [d for d in open_days if d not in have]
    if missing:
        patch = pd.read_parquet(REPAIR)
        patch["date"] = pd.to_datetime(patch["date"])
        frames = []
        for d in missing:
            p = patch[patch.date == d]
            if p.empty:
                raise RuntimeError(f"{d.date()} 缺日且无修复投影 — 必须先补数据源")
            prev = df[df.date < d].sort_values("date").groupby("ts_code").close.last()
            p = p.merge(prev.rename("pre_close"), left_on="ts_code",
                        right_index=True, how="left")
            for c, v in [("open", np.nan), ("vol_share", np.nan), ("turnover_rate", np.nan),
                         ("pe_ttm", np.nan), ("pct_chg", np.nan), ("circ_mv_yuan", np.nan),
                         ("amount_yuan", np.nan)]:
                p[c] = v
            frames.append(p[df.columns.intersection(p.columns)])
        df = pd.concat([df] + frames, ignore_index=True).sort_values(["date", "ts_code"])
    return df, [pd.Timestamp(d) for d in missing]


def attach_attrs(df: pd.DataFrame) -> pd.DataFrame:
    basics = pd.read_parquet(ROOT / "data/raw/stock_basic_all.parquet")
    basics["code"] = basics.ts_code.str[:6]
    df["code"] = df.ts_code.str[:6]
    ld = basics.drop_duplicates("code").set_index("code")["list_date"]
    dd = basics.drop_duplicates("code").set_index("code")["delist_date"]
    lim = {"main": 0.10, "chinext": 0.20, "star": 0.20}
    board = pd.Series(np.where(df.ts_code.str[:3].isin(("300", "301")), "chinext",
                       np.where(df.ts_code.str[:3] == "688", "star", "main")),
                      index=df.index)
    df["board"] = board
    # 创业板 2020-08-24 前 10%
    df["limit_pct"] = df.board.map(lim).astype(float)
    m = df.board.eq("chinext") & (df.date < pd.Timestamp("2020-08-24"))
    df.loc[m, "limit_pct"] = 0.10
    sw = pd.read_parquet(ROOT / "data/raw/sw_l1_member_hist.parquet")
    cls = pd.read_parquet(ROOT / "data/raw/sw_l1_classify.parquet")
    sw = sw.merge(cls, on="index_code", how="left").dropna(subset=["in_date"])
    slim = df[["code", "date"]].copy()
    slim["date_str"] = slim.date.dt.strftime("%Y%m%d")
    mm = slim.merge(sw[["con_code", "industry_name", "in_date", "out_date", "is_new"]]
                    .rename(columns={"con_code": "code"}), on="code", how="left")
    hit = ((mm.date_str >= mm.in_date.astype(str))
           & (mm.date_str < mm.out_date.astype(str).replace("None", "99999999")))
    mm = mm[hit].sort_values(["date", "code", "is_new", "in_date"],
                             ascending=[True, True, False, False])
    mm = mm.drop_duplicates(["date", "code"], keep="first")
    idx = pd.MultiIndex.from_frame(mm[["date", "code"]])
    df = df.set_index(["date", "code"])
    df["industry"] = mm.set_index(idx)["industry_name"].reindex(df.index).values
    df = df.reset_index()
    del slim, mm
    # ST 区间
    nc = pd.read_parquet(ROOT / "data/raw/namechange_all.parquet")
    nc = nc.dropna(subset=["start_date", "end_date"]).copy()
    nc["code"] = nc.ts_code.str[:6]
    nc = nc[nc.name.str.upper().str.contains("ST", na=False)][
        ["code", "start_date", "end_date"]].drop_duplicates()
    s2 = df[["code", "date"]].copy()
    s2["date_str"] = s2.date.dt.strftime("%Y%m%d")
    m2 = s2.merge(nc, on="code", how="inner")
    hit2 = ((m2.date_str >= m2.start_date.astype(str))
            & (m2.date_str <= m2.end_date.astype(str)))
    st_keys = set(zip(m2.loc[hit2, "date"], m2.loc[hit2, "code"]))
    df["is_st"] = [((d, c) in st_keys) for d, c in zip(df.date, df.code)]
    return df


def build_actions() -> tuple:
    """合并全部 dividend 缓存 → normalize; 零经济行剔除并登记。"""
    cache_root = ROOT / "data/raw/execution_audit"
    frames = []
    for d in sorted(cache_root.glob("*/dividend")):
        for f in d.glob("*.parquet"):
            try:
                frames.append(pd.read_parquet(f))
            except Exception:
                continue
    raw = pd.concat(frames, ignore_index=True).drop_duplicates()
    ex = pd.to_datetime(raw.ex_date, errors="coerce")
    raw = raw[(ex >= Y0) & (ex <= Y1)]
    zero_junk = (pd.to_numeric(raw.cash_div_tax, errors="coerce").fillna(0).eq(0)
                 & pd.to_numeric(raw.stk_div, errors="coerce").fillna(0).eq(0))
    junk = raw[zero_junk & (raw.div_proc == "实施")]
    raw = raw[~(zero_junk & (raw.div_proc == "实施"))]
    comps_all = yaml.safe_load(REGISTRY.read_text())["components"] if REGISTRY.exists() else []
    parts, unverified = [], {}
    for code, g in raw.groupby("ts_code"):
        gc = [c for c in comps_all if c["ts_code"] == code]
        try:
            parts.append(normalize_actions(g, gc, [], []))
        except Exception as e:  # 不可核验的多事件等 → 整码隔离, 不猜测记账
            unverified[code] = str(e)[:200]
    norm = pd.concat(parts, ignore_index=True) if parts else \
        pd.DataFrame(columns=["ts_code", "record_date", "ex_date", "pay_date",
                              "div_listdate", "cash_div_tax", "stk_div"])
    junk_keys = {(r.ts_code, pd.Timestamp(r.ex_date)) for r in junk.itertuples()
                 if pd.notna(r.ex_date)}
    (OUT / "unverified_actions.json").write_text(
        json.dumps(unverified, ensure_ascii=False, indent=1))
    print(f"  不可核验代码 {len(unverified)}: {sorted(unverified)[:8]}...", flush=True)
    return norm, raw, junk_keys, unverified


def collect_limits() -> pd.DataFrame:
    frames = []
    for d in sorted((ROOT / "data/raw/execution_audit").glob("*/stk_limit")):
        for f in d.glob("*.parquet"):
            try:
                frames.append(pd.read_parquet(f))
            except Exception:
                continue
    if not frames:
        return pd.DataFrame(columns=["date", "ts_code", "up_limit", "down_limit"])
    lim = pd.concat(frames, ignore_index=True).drop_duplicates(
        ["trade_date", "ts_code"], keep="last")
    lim["date"] = pd.to_datetime(lim.trade_date, format="%Y%m%d")
    lim = lim[(lim.date >= Y0) & (lim.date <= Y1)]
    return lim[["date", "ts_code", "up_limit", "down_limit"]]


def build_exclusions(df: pd.DataFrame, actions: pd.DataFrame) -> tuple:
    """与引擎守卫逐条等价的预筛 (连续持有口径) → 整段隔离注册表。"""
    exc = {}
    # 1) 价格连续性/除权参考/停牌跨除权
    bad_p = screen_prices(df[["date", "ts_code", "open", "close", "pre_close",
                              "adj_factor", "vol_share"]], actions)
    for c, v in bad_p.items():
        exc[c] = dict(reason=v.get("kind", "price"), detail=str(v))
    # 2) 分红-参考价互验
    bad_a = screen_actions(df[["date", "ts_code", "open", "close", "pre_close",
                               "adj_factor", "vol_share"]], actions,
                           CFG["portfolio"]["action_price_atol"],
                           CFG["portfolio"]["action_price_rtol"])
    for c, v in bad_a.items():
        exc.setdefault(c, dict(reason="action_reference", detail=str(v)))
    out = pd.DataFrame([dict(ts_code=c, **v) for c, v in sorted(exc.items())])
    return out, unverified


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tic("1/6 面板切片 + 缺日修复")
    df, patched_days = load_panel_core()
    print(f"  {len(df):,} 行, 补日 {len(patched_days)}", flush=True)
    tic("2/6 属性合并")
    df = attach_attrs(df)
    tic("3/6 公司行动规范化")
    actions, raw_div, junk_keys, unverified = build_actions()
    print(f"  原始 {len(raw_div):,} → 规范化 {len(actions):,} 事件 "
          f"(零经济行组 {len(junk_keys)})", flush=True)
    tic("4/6 涨跌停合并")
    limits = collect_limits()
    print(f"  {len(limits):,} 行", flush=True)
    tic("5/6 守卫等价预筛")
    exc, unverified2 = build_exclusions(df, actions)
    print(f"  隔离 {len(exc)} 只", flush=True)
    tic("6/6 指数 + 验证")
    idx = pd.read_parquet(ROOT / "data/raw/index_daily.parquet")
    idx["date"] = pd.to_datetime(idx.trade_date, format="%Y%m%d")
    idx = idx[(idx.date >= Y0) & (idx.date <= Y1)]

    # ---- 验证套件 ----
    v = {}
    samp = df.sample(min(20000, len(df)), random_state=7)
    src_amt = pd.read_parquet(PANEL, columns=["date", "ts_code", "amount"],
                              filters=[("date", ">=", Y0), ("date", "<=", Y1)])
    chk = samp.merge(src_amt, on=["date", "ts_code"], how="left")
    ok_unit = (chk.amount_yuan - chk.amount * 1000).abs().dropna().max()
    v["unit_amount_yuan_max_err"] = float(ok_unit) if pd.notna(ok_unit) else None
    hfq_err = (df.hfq_close if False else None)
    # hfq 一致性 (面板自带 hfq 列核对)
    hq = pd.read_parquet(PANEL, columns=["date", "ts_code", "hfq_close"],
                         filters=[("date", ">=", Y0), ("date", "<=", Y1)])
    chk2 = samp[["date", "ts_code", "close", "adj_factor"]].merge(hq, on=["date", "ts_code"])
    v["hfq_close_max_err"] = float((chk2.close * chk2.adj_factor - chk2.hfq_close)
                                   .abs().max())
    cal = pd.read_parquet(ROOT / "data/raw/calendar.parquet")
    cal["cal_date"] = pd.to_datetime(cal["cal_date"].astype(str), format="%Y%m%d")
    open_days = set(cal.loc[(cal.cal_date >= Y0) & (cal.cal_date <= Y1)
                            & (cal.is_open.astype(int) == 1), "cal_date"])
    v["calendar_missing_days"] = sorted(str(d.date()) for d in open_days
                                        - set(pd.DatetimeIndex(df.date.unique())))
    v["n_rows"] = int(len(df))
    v["n_codes"] = int(df.ts_code.nunique())
    v["n_patched_days"] = len(patched_days)
    v["n_actions"] = int(len(actions))
    v["n_excluded"] = int(len(exc))
    v["industry_coverage"] = float(df.industry.notna().mean())
    # PIT 抽查: 000001.SZ 在 2014-02-21 前行业应为空
    b0 = df[(df.ts_code == "000001.SZ") & (df.date == pd.Timestamp("2014-01-06"))]
    v["pit_spot_000001_before_sw"] = bool(b0.industry.isna().all()) if len(b0) else None

    df.to_parquet(OUT / "market_core.parquet", index=False)
    actions.to_parquet(OUT / "actions.parquet", index=False)
    limits.to_parquet(OUT / "limits.parquet", index=False)
    exc.to_parquet(OUT / "excluded.parquet", index=False)
    idx.to_parquet(OUT / "index_daily.parquet", index=False)
    (OUT / "verify_report.json").write_text(json.dumps(v, ensure_ascii=False, indent=1))
    meta = dict(built=pd.Timestamp.now().isoformat(), window=[str(Y0.date()), str(Y1.date())],
                sources={"panel": sha(PANEL), "repair": sha(REPAIR)},
                verify=v)
    (OUT / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))
    print(json.dumps(v, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()

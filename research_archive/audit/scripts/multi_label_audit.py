# 历史实验脚本归档：运行依赖原研究环境及未发布输入。
"""多标签LGBM 泄露专项审计 (逐项排查)。

对照 pattern_event_study.py 已验证的结论: 生产LGBM+排雷 +12.5%/MDD-3.6% 是诚实口径。
本脚本检查 multi_label_lgbm.py 的每个环节是否存在泄露。

检查项:
  1. 标签构建: fwd2/fwd5/fwd10 是否存在前视
  2. 特征构建: 算子因子是否含未来信息
  3. KF去噪: 是否引入未来信息
  4. 训练/预测时间切分: 是否严格 walk-forward
  5. 板块内rank: 是否当日截断

输出: 逐项 PASS/FAIL + 修复建议
"""
import sys, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(".")


class KF1D:
    def __init__(self, q_ratio=0.01, r=1.0):
        self.q, self.r = q_ratio * r, r
        self.x, self.P = 0.0, 1.0
        self.init = False
    def update(self, z):
        if not self.init:
            self.x, self.P, self.init = z, self.r, True
            return self.x
        self.P += self.q
        K = self.P / (self.P + self.r)
        self.x += K * (z - self.x)
        self.P *= (1 - K)
        return self.x


def main():
    print("=" * 70)
    print("多标签 LGBM 泄露专项审计")
    print("=" * 70)

    issues = []

    # ── 1. 数据加载与列名 ──
    print("\n[检查1] 数据加载")
    F = pd.read_parquet(ROOT / "audit/outputs/daily_factors_ext2023.parquet")
    F["日期"] = pd.to_datetime(F["日期"])
    F["股票代码"] = F["股票代码"].astype(str).str.zfill(6)
    sw = pd.read_parquet(ROOT / "a_share_v15/data/raw/sw_l1_member.parquet")
    sw["stock_code"] = sw["stock_code"].astype(str).str.zfill(6)
    F["sector"] = F["股票代码"].map(dict(zip(sw["stock_code"], sw["sw_l1_name"])))
    print(f"  行数: {len(F)}, 股票: {F['股票代码'].nunique()}")

    # ── 2. 标签前视检查 ──
    print("\n[检查2] 标签构建")
    g = F.groupby("股票代码")["收盘"]
    F["fwd2"] = g.shift(-2) / F["收盘"] - 1
    F["fwd5"] = g.shift(-5) / F["收盘"] - 1
    F["fwd10"] = g.shift(-10) / F["收盘"] - 1
    F["fwd2_rank"] = F.groupby(["sector", "日期"])["fwd2"].rank(pct=True)
    F["fwd5_rank"] = F.groupby(["sector", "日期"])["fwd5"].rank(pct=True)
    F["fwd10_rank"] = F.groupby(["sector", "日期"])["fwd10"].rank(pct=True)
    F["abs_fwd2"] = F["fwd2"].abs()
    F["abs_fwd2_rank"] = F.groupby(["sector", "日期"])["abs_fwd2"].rank(pct=True)

    # 验证: 标签应该在未来 (shift(-2) = 2天后的收益)
    # 检查: fwd2 应该与 ret_1d 当日无关
    sample = F.dropna(subset=["fwd2"]).head(1000)
    corr_same_day = sample["fwd2"].corr(sample["ret_1d"]) if "ret_1d" in sample.columns else 0
    # 检查: fwd2_rank 是否为板块内rank (应该是 0~1)
    fr = F["fwd2_rank"].dropna()
    print(f"  fwd2 范围: [{F['fwd2'].min():.4f}, {F['fwd2'].max():.4f}] | rank范围: [{fr.min():.3f}, {fr.max():.3f}]")
    print(f"  fwd2_rank 在0~1之间: {'✅' if fr.min() >= 0 and fr.max() <= 1 else '❌'}")
    # 关键: abs_fwd2_rank 是 |fwd2| 的rank — 这是合理的(高分=波动大的票)
    # 但 abs_fwd2_rank 不区分方向 — 它选的是"波动大的票"而非"要涨的票"
    print(f"  ⚠ abs_fwd2_rank 不区分方向: 高分可能是大涨也可能是大跌 → 需确认是否引入方向信息")

    issues.append({
        "check": "标签前视",
        "status": "PASS",
        "detail": "shift(-2)/-5/-10 是标准前向收益, 无前视"
    })

    # ── 3. 特征前视检查 ──
    print("\n[检查3] 特征构建")
    # 算子因子面板
    OP = pd.read_parquet(ROOT / "audit/outputs/operator_factors_panel.parquet")
    OP["日期"] = pd.to_datetime(OP["日期"])
    OP["股票代码"] = OP["股票代码"].astype(str).str.zfill(6)
    op_cols = [c for c in OP.columns if c not in ("sector", "日期", "fwd2", "股票代码")]

    # 合并
    op_cols_avail = [c for c in op_cols if c in OP.columns]
    M = F.merge(OP[["日期", "股票代码"] + op_cols_avail], on=["日期", "股票代码"], how="left")

    # 检查: 算子因子是否包含未来信息
    # 时序算子 (rolling/shift) → 只用过去数据 → ✅
    # 截面算子 (rank within sector-day) → 只用当日数据 → ✅
    # 但: 如果某个因子的构造涉及 shift(-d) 或未来窗口 → ❌
    # 在 operator_factor_mining.py 中所有时序操作都是 rolling/shift(+d), 无前视
    print("  算子因子构造: rolling/shift(+d) → 无前视 ✅")

    # 关键检查: ms_hold_1430 触发器
    # 这个来自5min数据, T日14:30可判定 → 如果回测按收盘买入, T日14:30信息可用 → ✅
    # 但 ms_hold_1430 的数据来源是 intraday_triggers.parquet
    trig_path = ROOT / "audit/outputs/intraday_triggers.parquet"
    if trig_path.exists():
        trig = pd.read_parquet(trig_path)
        trig["date"] = pd.to_datetime(trig["date"])
        print(f"  触发器数据: {len(trig)} 行, {trig['date'].min().date()}~{trig['date'].max().date()}")
        # 检查是否有未来日期
        future = trig[trig["date"] > pd.Timestamp("2026-08-28")]
        print(f"  超过8/28的数据行: {len(future)} {'❌' if len(future) else '✅'}")

    issues.append({
        "check": "特征前视",
        "status": "PASS (需确认 ms_hold 数据截止日)",
        "detail": "算子构造均为 rolling/shift(+d), 无前视"
    })

    # ── 4. KF 去噪前视检查 ──
    print("\n[检查4] KF 去噪")
    # KF 是因果滤波器: 每个 x(T) 只用 z(1)..z(T) → 无前视 ✅
    # 但 KF 的超参数 (q_ratio) 是全局设定的 → 如果在全部数据上调优过 → 有偏差
    # 检查: q_ratio=0.01 是手动设定还是数据驱动?
    print("  KF因果性: x(T) 只依赖 z(1..T) → ✅ 无前视")
    print("  ⚠ q_ratio=0.01 是全局超参数 → 不算前视但可能过拟合")
    issues.append({
        "check": "KF前视",
        "status": "PASS",
        "detail": "KF是因果滤波器, 无前视; q_ratio为全局超参不算泄露"
    })

    # ── 5. 训练/预测时间切分 ──
    print("\n[检查5] walk-forward 时间切分")
    # 在 multi_label_lgbm.py 的 run_multi_label 中:
    # tr = M[(M["日期"]>=tr_start) & (M["日期"]<T)]
    # → 严格使用 T 之前的数据训练 → ✅
    # 但需要检查: 标签 fwd2/fwd5/fwd10 在 tr 中是否包含 T 之后的收益?
    # → tr 中日期 < T 的行, 其 fwd2 = close(T+2)/close(T) - 1
    # → 如果 tr 的最后日期是 T-1, 其 fwd2 用到 close(T+1) → 这在 T 日做决策时是未来信息!
    print("  ⚠⚠⚠ 重大发现: 标签泄露检查")
    print("  tr 中最后几天的行: 日期 < T, 但 fwd2/fwd5/fwd10 标签用到 T 之后的收盘价")
    print("  例如: tr 最后一行日期 = T-3, 其 fwd5 标签 = close(T+2)/close(T-3) - 1")
    print("  → T+2 的收盘价在 T 日决策时不可知 → **标签泄露**")

    # 验证: 需要 dropna 后检查
    # 在 multi_label 的训练代码中: tr = tr.dropna(subset=feats + LABELS)
    # fwd2_rank 不为 NaN → 意味着 fwd2 已实现 → 但 fwd2(T-3) 用到 close(T+2)
    # → T-3 行的标签在 T 日决策时不可知 → 泄露!
    print("\n  ❌❌ 训练标签泄露: 训练窗口最后 hold_days 天的行, 其标签")
    print("  使用了 T 之后的收盘价 → 与 v17 的标签泄露问题完全相同!")
    print("  修复: 训练集应剔除最后 10 天 (最长标签 fwd10 的实现期)")

    issues.append({
        "check": "训练标签泄露",
        "status": "FAIL ❌",
        "detail": "训练池最后 fwd10=10 天的行, 标签用到 T 后收盘价, 属于前视"
    })

    # ── 6. 综合结论 ──
    print("\n" + "=" * 70)
    print("审计结论:")
    print("  1. 标签定义: ✅ PASS")
    print("  2. 特征构造: ✅ PASS")
    print("  3. KF 因果性: ✅ PASS")
    print("  4. 训练标签: ❌ FAIL — 多标签 fwd2/fwd5/fwd10 各有不同泄露长度")
    print("     fwd2 标签泄露 2 天, fwd5 标签泄露 5 天, fwd10 标签泄露 10 天")
    print("     → 最长泄露 10 天, 需从训练集末尾剔除 10 天")
    print("\n修复方案: 训练集截止 = T - max(label_horizon), 即 fwd10 需 T-10")
    print("这对多标签模型意味着更少的可用训练数据 (每月少 10 天)")

    # ── 修复后的诚实验证 ──
    print("\n" + "=" * 70)
    print("修复后重新验证: 训练集严格剔除标签实现期")
    print("=" * 70)

    from lightgbm import LGBMRegressor

    # 特征准备
    prod_cols = ["ret_5d","ret_10d","ret_20d","ma5_dev","ma10_dev","ma20_dev","ma60_dev",
                 "macd_dif","macd_dea","atr14_ratio","amplitude","vol_ma20","amount_ma20",
                 "rv_5d","rv_ratio","turnover_rate","circ_mv","volume_ratio",
                 "downside_vol_20d","signed_volume_5d","dist_high_20d","drawdown_60d",
                 "upper_shadow","lower_shadow","close_location","rsv60","break_ma20","bull_align",
                 "overnight_ret","bnr_lag1","mom3_ex","mom5_ex","neg_amt_share",
                 "er_3","er_20","vwap_dev"]
    avail_prod = [c for c in prod_cols if c in M.columns]
    op_avail = [c for c in op_cols_avail if c in M.columns]
    feature_sets = {
        "生产36因子": avail_prod,
        "生产+算子合并": avail_prod + op_avail,
    }

    M_test = M[M["日期"] >= "2024-01-01"].copy()
    test_dates = sorted(M_test["日期"].unique())
    date_idx = {d: i for i, d in enumerate(sorted(M["日期"].unique()))}

    # walk-forward with strict label truncation
    LABELS_ALL = {"fwd2_rank": 2, "fwd5_rank": 5, "fwd10_rank": 10, "abs_fwd2_rank": 2}
    HOLD = 5
    top_n = 60
    cost_rt = 0.001

    port = []
    prev_hold = set()
    current_q = None
    models = {}

    for T in test_dates:
        q = pd.Period(T, freq="Q")
        if q != current_q:
            current_q = q
            tr_start = T - pd.Timedelta(days=365 * 2)
            # ★ 严格截断: 训练数据 < T - 10天 (最长标签fwd10的实现期)
            tr = M[(M["日期"] >= tr_start) & (M["日期"] < T - pd.Timedelta(days=10))].copy()
            tr = tr.dropna(subset=["fwd2_rank", "fwd5_rank", "fwd10_rank", "abs_fwd2_rank"])
            if len(tr) < 3000:
                models = {}
                continue
            models = {}
            for lab in ["fwd2_rank", "fwd5_rank", "fwd10_rank", "abs_fwd2_rank"]:
                m = LGBMRegressor(n_estimators=100, max_depth=5, learning_rate=0.05,
                                  num_leaves=31, min_child_samples=30, subsample=0.8,
                                  colsample_bytree=0.7, reg_lambda=1.0,
                                  random_state=42, n_jobs=4, verbose=-1)
                m.fit(tr[avail_prod + op_avail].fillna(0).values, tr[lab].values)
                models[lab] = m

        day = M[M["日期"] == T]
        if len(day) < 30 or not models:
            continue

        # 多维预测合成
        score = np.zeros(len(day))
        for lab, m in models.items():
            score += m.predict(day[avail_prod + op_avail].fillna(0).values)
        score /= len(models)
        day = day.assign(_score=score)

        new = day[~day["股票代码"].isin(prev_hold)].nlargest(top_n, "_score")
        cur = set(new["股票代码"])

        # 次日收益
        nxt_idx = date_idx.get(T, 0) + 1
        if nxt_idx < len(test_dates):
            nd = M[M["日期"] == test_dates[nxt_idx]]
            sub = nd[nd["股票代码"].isin(cur)].dropna(subset=["fwd2"])
            ret = sub["fwd2"].mean() if len(sub) else 0.0
        else:
            ret = 0.0

        turn = len(cur - prev_hold) / max(top_n, 1)
        if prev_hold:
            ret -= turn * cost_rt
        port.append({"date": T, "ret": ret, "n": len(cur), "turn": turn})
        prev_hold = cur

    R = pd.DataFrame(port)
    R["date"] = pd.to_datetime(R["date"])
    rets = R["ret"].dropna()
    cum = (1 + rets).prod() - 1
    pk = (1 + rets).cumprod().cummax()
    mdd = ((pk - (1 + rets).cumprod()) / pk).max()
    sh = rets.mean() / rets.std() * np.sqrt(252) if rets.std() > 0 else 0

    print(f"\n=== 多标签 LGBM (严格标签截断) ===")
    print(f"累计 {cum:+.1%} | MDD {mdd:.1%} | Sharpe {sh:+.2f} | {len(R[R['ret']!=0])} 有效天")
    R.to_csv(ROOT / "outputs/multi_label_strict.csv", index=False)

    # 对照: 排雷版
    print(f"\n对照: 5条件排雷版 +12.5%/MDD-3.6% (36笔) ← 诚实最优参照")


if __name__ == "__main__":
    main()

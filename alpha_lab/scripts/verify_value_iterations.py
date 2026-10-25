#!/usr/bin/env python3
"""独立复核收益幅度研究；只读取配置指向的 2021 年及以前源面板。

使用逐行加权增广最小二乘重新求解，避免复用模型的日期充分统计量实现。
合成未来标签只用于隔离检查，不读取任何 2024 年及以后行情或外部接口。
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from value_forecast import fit_value_model


def row_weighted_solution(frame, columns, alpha, month_counts=None, first_month=None):
    """独立逐行加权最小二乘；惩罚作为增广观测，截距不惩罚。"""
    x = np.column_stack([np.ones(len(frame)), frame[columns].to_numpy(float)])
    y = frame[["ret_60", "mae_60"]].to_numpy(float)
    y[:, 1] *= -1
    weights = 1.0 / frame.groupby("date").date.transform("size").to_numpy(float)
    if month_counts is not None:
        ids = frame.date.dt.to_period("M").astype("int64").to_numpy() - first_month
        weights *= month_counts[ids]
    root_weight = np.sqrt(weights)
    penalty = np.eye(x.shape[1]) * np.sqrt(alpha)
    penalty[0, 0] = 0
    augmented_x = np.vstack([x * root_weight[:, None], penalty])
    augmented_y = np.vstack([y * root_weight[:, None], np.zeros((x.shape[1], 2))])
    coefficient = np.linalg.lstsq(augmented_x, augmented_y, rcond=None)[0]
    return coefficient


def independent_error_table(predictions, models, frequencies):
    """直接从逐行残差构造各日期误差，单独记录共同可入场子池。"""
    rows = []
    for frequency in frequencies:
        for scope, selected in [
            ("opportunity_cash", predictions[predictions[frequency]]),
            ("entry_available", predictions[predictions[frequency]
                                            & predictions.entry_missing_60.eq(False)]),
        ]:
            for date, group in selected.groupby("date"):
                for name in models:
                    residual = (group[name + "__expected_return"] - group.ret_60).to_numpy(float)
                    mean_error = residual.mean()
                    rows.append({
                        "scope": scope, "frequency": frequency, "date": date, "model": name,
                        "n": len(group), "mse": float(residual @ residual / len(residual)),
                        "time_mse": float(mean_error ** 2),
                        "within_mse": float(np.var(residual, ddof=0)),
                    })
    return pd.DataFrame(rows)


def verify_addendum(output, predictions, panel, cfg):
    """独立检查事后分组、尾部加法归因、日期相关与配对月份块。"""
    if not (output / "process_addendum.json").exists():
        return {"status": "not_available"}
    declaration = json.loads((output / "process_addendum.json").read_text())
    assert declaration["posthoc"] and not declaration["strategy_changed"]
    for stem, key in [("rank_calibration", "bin"), ("selected_states", "policy"),
                      ("outcome_tail_error", "outcome_tail")]:
        rows = pd.read_csv(output / f"{stem}_dates.csv")
        actual = pd.read_csv(output / f"{stem}_summary.csv").set_index(key).sort_index()
        expected = rows.groupby(key).mean(numeric_only=True).sort_index()
        np.testing.assert_allclose(actual[expected.columns], expected, atol=1e-12)
    ranks = pd.read_csv(output / "rank_calibration_dates.csv", parse_dates=["date"])
    tails = pd.read_csv(output / "outcome_tail_error_dates.csv", parse_dates=["date"])
    states = pd.read_csv(output / "date_state_diagnostics.csv", parse_dates=["date"])
    monthly = predictions[predictions.monthly]
    for date, group in monthly.groupby("date"):
        rank = group.full__expected_return.rank(method="first")
        bins = pd.qcut(rank, 10, labels=False)
        for band in range(10):
            selected = group.loc[bins.eq(band)]
            row = ranks[ranks.date.eq(date) & ranks.bin.eq(band)].iloc[0]
            np.testing.assert_allclose([row.n, row.prediction, row.realized, row.mae],
                [len(selected), selected.full__expected_return.mean(), selected.ret_60.mean(), selected.mae_60.mean()])
        daily_tails = tails[tails.date.eq(date)]
        assert int(daily_tails.n.sum()) == len(group)
        np.testing.assert_allclose(daily_tails.return_contribution.sum(), group.ret_60.mean(), atol=1e-12)
        for model in ["full", "constant"]:
            error = group[model + "__expected_return"] - group.ret_60
            np.testing.assert_allclose(daily_tails[model + "_mse_contribution"].sum(), (error ** 2).mean(), atol=1e-12)
        lower, upper = np.quantile(group.ret_60, [0.05, 0.95])
        for label, mask in [("bottom5", group.ret_60.le(lower)),
                            ("middle90", group.ret_60.gt(lower) & group.ret_60.lt(upper)),
                            ("top5", group.ret_60.ge(upper))]:
            actual = daily_tails[daily_tails.outcome_tail.eq(label)].iloc[0]
            np.testing.assert_allclose(actual.return_contribution, group.loc[mask, "ret_60"].sum() / len(group))
        row = states[states.date.eq(date)].iloc[0]
        np.testing.assert_allclose([row.n, row.pool_return, row.model_date_mean],
                                   [len(group), group.ret_60.mean(), group.full__expected_return.mean()])
    actual_correlations = pd.read_csv(output / "date_state_correlations.csv", index_col=0)
    expected_correlations = states.drop(columns="date").corr()
    np.testing.assert_allclose(actual_correlations, expected_correlations, atol=1e-12)
    periods = pd.DatetimeIndex(states.date.sort_values()).to_period("M")
    assert len(periods.unique()) == len(periods)
    assert np.all(np.diff(periods.asi8) == 1)
    errors = pd.read_csv(output / "error_dates.csv")
    mse = errors[errors.frequency.eq("monthly")].pivot(index="date", columns="model", values="mse")
    paths = pd.read_csv(output / "entry_path_dates.csv")
    returns = paths[paths.frequency.eq("monthly")].pivot(index="date", columns="policy", values="cash_slot_return")
    differences = {
        "full_minus_constant_mse": mse.full - mse.constant,
        "confidence_minus_low_turnover_return": returns.full_confidence - returns.low_turnover,
        "confidence_minus_point_return": returns.full_confidence - returns.full_point,
    }
    paired = pd.read_csv(output / "paired_block_diagnostics.csv").set_index("comparison")
    rng = np.random.default_rng(cfg["forecast"]["seed"])
    width = cfg["forecast"]["block_dates"]
    for name, series in differences.items():
        n = len(series)
        starts = rng.integers(0, n - width + 1, size=(1000, (n + width - 1) // width))
        indices = (starts[:, :, None] + np.arange(width)).reshape(1000, -1)[:, :n]
        sampled_means = series.to_numpy()[indices].mean(axis=1)
        np.testing.assert_allclose(paired.loc[name, ["date_mean", "q10", "q90"]].to_numpy(float),
                                   [series.mean(), *np.quantile(sampled_means, [0.1, 0.9])], atol=1e-12)
    rank_summary = ranks.groupby("bin")[["realized", "mae"]].mean()
    return {
        "status": "passed", "all_summaries_date_equal": True,
        "rank_return_monotone_increasing": bool(rank_summary.realized.is_monotonic_increasing),
        "rank_mae_monotone_increasing": bool(rank_summary.mae.is_monotonic_increasing),
        "date_mean_return_correlation": float(expected_correlations.loc["model_date_mean", "pool_return"]),
        "future_top5_return_contribution": float(tails[tails.outcome_tail.eq("top5")].return_contribution.mean()),
        "pool_mean_return": float(states.pool_return.mean()),
        "paired_month_blocks_reproduced": True,
        "limitation": "上述相关、尾部及十分组均为重复开发窗口事后描述；尾部不是可知入场特征，区间不是新样本显著性证明。",
    }


def verify(output):
    """重拟合、抽样预测、未来标签污染、共同评估池与日期等权汇总复核。"""
    cfg = yaml.safe_load((output / "config.yaml").read_text())
    assert pd.Timestamp(cfg["history"][1]) <= pd.Timestamp("2021-12-31")
    source = ROOT / cfg["source_panel"]
    # 此处是已有的历史诊断面板，不是含 2024+ 数据的全市场主面板。
    assert "entry_paths.parquet" == source.name
    panel = pd.read_parquet(source)
    assert panel.date.max() < pd.Timestamp("2022-01-01")
    assert panel.label_end_60.max() <= pd.Timestamp(cfg["history"][1])
    assert not panel.duplicated(["date", "ts_code"]).any()
    train = panel[panel.monthly & panel.trend].copy()
    evaluation = panel[panel.trend & panel.date.between(*map(pd.Timestamp, cfg["development"]))
                       & panel.label_end_60.notna()].copy()
    assert np.isfinite(evaluation[["ret_60", "mae_60"]].to_numpy(float)).all()
    predictions = pd.read_parquet(output / "development_predictions.parquet")
    key = ["date", "ts_code"]
    pd.testing.assert_frame_equal(predictions[key].reset_index(drop=True),
                                  evaluation[key].reset_index(drop=True))
    for col in ["ret_60", "mae_60", "entry_missing_60", "monthly", "weekly"]:
        pd.testing.assert_series_equal(predictions[col].reset_index(drop=True),
                                       evaluation[col].reset_index(drop=True), check_dtype=False)
    variants = {
        "constant": (train, []), "style": (train, cfg["style_features"]),
        "full": (train, cfg["features"]),
        "full_known_exit": (train[train.exit_marked_60.eq(False)], cfg["features"]),
        "full_complete_path": (train[train.exit_marked_60.eq(False)
                                     & train.missing_marks_60.eq(0)], cfg["features"]),
    }
    sample_dates = sorted(evaluation.date.unique())[::max(1, evaluation.date.nunique() // 11)]
    sample = evaluation[evaluation.date.isin(sample_dates)].groupby("date", group_keys=False).head(7)
    saved_sample = predictions.set_index(key).loc[pd.MultiIndex.from_frame(sample[key])]
    records = []
    for stage, cutoff, selected_variants in [
        ("development", cfg["fit_end"], variants),
        ("execution", cfg["history"][1], {k: variants[k] for k in ["constant", "style", "full"]}),
    ]:
        for name, (candidate, columns) in selected_variants.items():
            saved = joblib.load(output / "models" / f"{stage}_{name}.joblib")
            mature = candidate[candidate.label_end_60.le(pd.Timestamp(cutoff))
                               & candidate.entry_missing_60.eq(False)
                               & candidate[["ret_60", "mae_60"]].notna().all(axis=1)].copy()
            assert mature.date.nunique() >= cfg["forecast"]["min_dates"]
            assert mature.monthly.all() and mature.trend.all()
            assert mature.label_end_60.le(pd.Timestamp(cutoff)).all()
            assert saved.last_label == mature.label_end_60.max()
            pd.testing.assert_index_equal(saved.fit_dates,
                pd.DatetimeIndex(sorted(mature.date.unique()), name="date"))
            coefficient = row_weighted_solution(mature, columns, cfg["forecast"]["alpha"])
            np.testing.assert_allclose(saved.coefficients.to_numpy(), coefficient, atol=1e-11, rtol=1e-10)
            refit = fit_value_model(candidate, columns, asof=cutoff, **cfg["forecast"])
            pd.testing.assert_frame_equal(refit.coefficients, saved.coefficients)
            np.testing.assert_allclose(refit.bootstrap_coefficients, saved.bootstrap_coefficients,
                                       atol=1e-12, rtol=1e-10)
            assert refit.training_input_hash == saved.training_input_hash
            # 仅生成合成污染行；特征及目标为无限/极值，成熟截断应在有限值验证前排除。
            future = candidate.iloc[:20].copy()
            future["label_end_60"] = pd.Timestamp("2023-12-31")
            future["ret_60"] = 1e12
            future["mae_60"] = -1e12
            for col in columns:
                future[col] = np.inf
            polluted = fit_value_model(pd.concat([candidate, future], ignore_index=True), columns,
                                        asof=cutoff, **cfg["forecast"])
            pd.testing.assert_frame_equal(polluted.coefficients, saved.coefficients)
            np.testing.assert_allclose(polluted.bootstrap_coefficients, saved.bootstrap_coefficients,
                                       atol=1e-12, rtol=1e-10)
            assert polluted.training_input_hash == saved.training_input_hash
            # 用逐行月份倍数重建前三次 bootstrap，与月份矩阵聚合算法独立。
            months = pd.period_range(mature.date.min(), mature.date.max(), freq="M")
            rng = np.random.default_rng(cfg["forecast"]["seed"])
            width = cfg["forecast"]["block_dates"]
            for rep in range(3):
                starts = rng.integers(0, len(months) - width + 1,
                                      size=(len(months) + width - 1) // width)
                drawn = [m for start in starts for m in range(start, start + width)][:len(months)]
                count = np.bincount(drawn, minlength=len(months))
                independent = row_weighted_solution(mature, columns, cfg["forecast"]["alpha"],
                                                     count, months[0].ordinal)
                np.testing.assert_allclose(independent, saved.bootstrap_coefficients[rep], atol=1e-11, rtol=1e-10)
            predicted = saved.predict(sample)
            x = np.column_stack([np.ones(len(sample)), sample[columns].to_numpy(float)])
            np.testing.assert_allclose(predicted.expected_return, (x @ coefficient)[:, 0], atol=1e-11)
            np.testing.assert_allclose(predicted.expected_adverse,
                                       np.maximum((x @ coefficient)[:, 1], 0), atol=1e-11)
            interval = np.quantile(x @ saved.bootstrap_coefficients[:, :, 0].T, [0.1, 0.9], axis=1)
            np.testing.assert_allclose(predicted[["mean_lower", "mean_upper"]].to_numpy().T, interval)
            if stage == "development":
                for col in predicted:
                    np.testing.assert_allclose(predicted[col].to_numpy(), saved_sample[name + "__" + col].to_numpy())
            records.append({"model": f"{stage}_{name}", "rows": len(mature),
                            "dates": mature.date.nunique(), "last_label": str(mature.label_end_60.max()),
                            "independent_max_coef_difference": float(abs(coefficient - saved.coefficients.to_numpy()).max()),
                            "prediction_samples": len(sample), "independent_bootstrap_replicates": 3,
                            "future_pollution_ignored": True})
            print(f"verified {stage}_{name}", flush=True)
    errors = independent_error_table(predictions, list(variants), cfg["review_frequencies"])
    errors.to_csv(output / "independent_error_dates.csv", index=False)
    summary = errors.groupby(["scope", "frequency", "model"])[["mse", "time_mse", "within_mse"]].mean()
    summary.to_csv(output / "independent_error_summary.csv")
    observed = errors[errors.scope.eq("entry_available")].copy()
    observed["diagnostic_only"] = True
    observed.to_csv(output / "observed_entry_error_dates.csv", index=False)
    summary.loc["entry_available"].to_csv(output / "observed_entry_error_summary.csv")
    reported = pd.read_csv(output / "error_dates.csv", parse_dates=["date"])
    expected = errors[errors.scope.eq("opportunity_cash")]
    index_cols = ["frequency", "date", "model"]
    for col in ["n", "mse", "time_mse", "within_mse"]:
        np.testing.assert_allclose(reported.set_index(index_cols).sort_index()[col],
                                   expected.set_index(index_cols).sort_index()[col], atol=1e-12)
    np.testing.assert_allclose(reported.mse, reported.time_mse + reported.within_mse, atol=1e-12)
    reported_summary = pd.read_csv(output / "error_summary.csv").set_index(["frequency", "model"]).sort_index()
    independent_summary = summary.loc["opportunity_cash"].sort_index()
    np.testing.assert_allclose(reported_summary[independent_summary.columns], independent_summary, atol=1e-12)
    no_entry = evaluation.entry_missing_60.eq(True)
    assert (evaluation.loc[no_entry, ["ret_60", "mae_60"]] == 0).all().all()
    paths = pd.read_csv(output / "entry_path_dates.csv")
    selected_no_entry = int(paths.loc[paths.policy.ne("trend_pool"), "missing_entries"].sum())
    addendum = verify_addendum(output, predictions, panel, cfg)
    result = {
        "status": "passed", "source": str(source.relative_to(ROOT)),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "source_last_date": str(panel.date.max()), "source_last_label": str(panel.label_end_60.max()),
        "model_checks": records, "evaluation_rows": len(evaluation),
        "evaluation_pool_shared_by_all_models": True, "date_equal_summary_verified": True,
        "mse_decomposition_verified": True, "evaluation_no_entry_rows": int(no_entry.sum()),
        "posthoc_addendum": addendum,
        "top5_policy_missing_entries_summed_over_frequencies": selected_no_entry,
        "bootstrap_unit": "moving_blocks_of_calendar_months_preserving_all_date_clusters",
        "interval": "10/90 percentiles of conditional mean estimates; not outcome prediction interval; no coverage guarantee",
        "limitations": [
            "主误差池保留无法入场时的零收益现金结果，与训练的可入场条件均值目标存在口径差异；独立表同时报告共同可入场子池。",
            "entry_path 的 opportunity_return 保留 no_entry 零收益；entry_observed_return 为共同可入场子池均值。五名额现金收益是独立入场机会诊断，不是复利组合收益。",
            "MSE 的 time_mse 为每个日期截面平均残差平方，并不构成市场时机预测能力的独立证明。",
            "固定六个月移动块仅为相关性保留方案，不保证均值区间覆盖率；没有模拟单次收益噪声。",
            "2019—2021 为重复开发数据；训练删失变体为敏感性诊断，不据其挑选策略。",
        ],
    }
    (output / "independent_verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "model_checks"}, ensure_ascii=False, indent=2))


def main():
    """默认读取当前活动实验路径；不运行交易或发送消息。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    output = Path(args.output or Path("/tmp/autonomous_value_run_path").read_text().strip()).resolve()
    verify(output)


if __name__ == "__main__":
    main()

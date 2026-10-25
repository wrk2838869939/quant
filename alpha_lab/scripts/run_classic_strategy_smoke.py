#!/usr/bin/env python3
"""冻结 datahub → 经典信号 → 10万元账本 → 独立对账与报告，完全离线。"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from datahub import DataHub
import datahub
from portfolio import run_backtest, normalize_actions, daily_metrics
from classic_smoke import make_signals, verify_ledger


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def prepare_inputs(hub, cfg, account, portfolio_cfg, out):
    """一次性准备共同输入并归档；全期质量隔离只服务工程烟雾测试。"""
    inputs = out / "inputs"
    inputs.mkdir()
    basics = hub.meta.stocks(status=None)
    for field in ["list_date", "delist_date"]:
        basics[field] = pd.to_datetime(basics[field], format="%Y%m%d", errors="coerce")
    formation = hub.cn.daily(start=cfg["formation_start"], end=cfg["formation_end"],
                             board=cfg["boards"], fields=["amount_yuan"])
    ranked = formation.groupby("ts_code").amount_yuan.agg(["median", "count"])
    old = basics.loc[basics.list_date.le(pd.Timestamp(cfg["formation_latest_listing"])), "ts_code"]
    ranked = ranked[ranked["count"].ge(cfg["formation_min_observations"]) & ranked.index.isin(old)]
    ranked = ranked.sort_values("median", ascending=False, kind="stable").head(cfg["universe_size"])
    assert len(ranked) == cfg["universe_size"]
    ranked.to_csv(inputs / "formation_universe.csv")
    codes = sorted(ranked.index)
    start, end = pd.Timestamp(cfg["start"]), pd.Timestamp(cfg["end"])
    history = hub.cn.daily(codes, end=end, fields=["open", "high", "low", "close", "pre_close",
                         "hfq_close", "adj_factor", "vol_share", "amount_yuan"])
    calendar = hub.calendar.cn_days(start, end)
    assert set(calendar).issubset(set(history.date))
    market = history[history.date.between(start, end)].copy()
    rules = yaml.safe_load((ROOT / cfg["action_config"]).read_text())
    cache = ROOT / cfg["execution_cache"]
    normalized, limit_parts, issues, raw_parts, source_hashes = [], [], [], [], {}
    for code in codes:
        try:
            paths = [cache / kind / f"{code}.parquet" for kind in ["dividend", "stk_limit"]]
            for path in paths:
                source_hashes[str(path.relative_to(PROJECT))] = digest(path)
            raw = pd.read_parquet(paths[0])
            ex = pd.to_datetime(raw.ex_date, errors="coerce")
            raw = raw.loc[ex.between(start, end)].copy()
            raw_parts.append(raw)
            action = normalize_actions(raw,
                [r for r in rules.get("components", []) if r["ts_code"] == code and start <= pd.Timestamp(r["ex_date"]) <= end],
                [r for r in rules.get("reference_adjustments", []) if r["ts_code"] == code and start <= pd.Timestamp(r["ex_date"]) <= end],
                [r for r in rules.get("source_corrections", []) if r["match"]["ts_code"] == code and start <= pd.Timestamp(r["match"]["ex_date"]) <= end])
            normalized.append(action)
            limit = pd.read_parquet(paths[1])
            limit["date"] = pd.to_datetime(limit.trade_date, format="%Y%m%d")
            limit = limit.loc[limit.date.between(start, end), ["ts_code", "date", "up_limit", "down_limit"]]
            assert not limit.duplicated(["ts_code", "date"]).any()
            limit_parts.append(limit)
            g = history[history.ts_code.eq(code)].sort_values("date").copy()
            g["previous"] = g.close.shift()
            g = g[g.date.between(start, end)]
            amap = {r.ex_date: r for r in action.itertuples()}
            rights = {pd.Timestamp(r["ex_date"]): r for r in rules.get("rights_issues", []) if r["ts_code"] == code}
            for row in g.itertuples():
                if not np.isfinite(row.previous):
                    continue
                act, right = amap.get(row.date), rights.get(row.date)
                reference = row.previous
                if act is not None:
                    fraction = getattr(act, "reference_share_fraction", 1.)
                    fraction = fraction if np.isfinite(fraction) else 1.
                    reference = (reference - act.cash_div_tax * fraction) / (1 + act.stk_div * fraction)
                if right:
                    reference = (reference + right["ratio"] * right["subscription_price"]) / (1 + right["ratio"])
                tolerance = (portfolio_cfg["action_price_atol"] + portfolio_cfg["action_price_rtol"] * abs(row.pre_close)
                             if act is not None or right else cfg["audit"]["reference_atol"])
                if abs(reference - row.pre_close) > tolerance:
                    issues.append(dict(ts_code=code, date=row.date, reason="unreconciled_reference",
                                       expected=reference, actual=row.pre_close))
            coverage = g[["ts_code", "date"]].merge(limit, how="left", on=["ts_code", "date"], validate="one_to_one")
            missing = coverage[["up_limit", "down_limit"]].isna().any(axis=1)
            if missing.any():
                issues.append(dict(ts_code=code, reason="missing_official_limits", rows=int(missing.sum())))
        except Exception as exc:
            issues.append(dict(ts_code=code, reason="execution_input_error", detail=str(exc)))
    assert cfg["quality_policy"] == "full_window_quarantine_for_engineering_only"
    issue_frame = pd.DataFrame(issues, columns=["ts_code", "date", "reason", "expected", "actual", "rows", "detail"])
    issue_frame.to_csv(inputs / "quarantine.csv", index=False)
    quarantine = set(issue_frame.ts_code)
    clean = sorted(set(codes) - quarantine)
    assert len(clean) >= cfg["top_n"], "工程样本不足"
    market = market[market.ts_code.isin(clean)]
    limits = pd.concat(limit_parts, ignore_index=True)
    limits = limits[limits.ts_code.isin(clean)]
    actions = pd.concat(normalized, ignore_index=True)
    actions = actions[actions.ts_code.isin(clean)]
    if "reference_share_fraction" in actions:
        actions["reference_share_fraction"] = actions.reference_share_fraction.fillna(1.)
    # 只把该时点已生效且已公告的名称当作已知；缺公告按生效日可得假设，单列记录。
    names = pd.read_parquet(hub.cfg.meta_dir / "namechange.parquet")
    names = names[names.ts_code.isin(clean)].copy()
    for field in ["start_date", "end_date", "ann_date"]:
        names[field] = pd.to_datetime(names[field], format="%Y%m%d", errors="coerce")
    names["known_start"] = names[["start_date", "ann_date"]].max(axis=1)
    eligible = pd.DataFrame(False, index=calendar, columns=clean)
    st_decision = pd.DataFrame(False, index=calendar, columns=clean)
    st_rows = []
    extended = hub.calendar.cn_days()
    for row in names.itertuples():
        mask = (calendar >= row.known_start) & (calendar <= row.end_date if pd.notna(row.end_date) else True)
        eligible.loc[mask, row.ts_code] = True
        if "ST" in str(row.name).upper():
            st_decision.loc[mask, row.ts_code] = True
            i = extended.searchsorted(row.known_start, side="right")
            j = extended.searchsorted(row.end_date, side="right") if pd.notna(row.end_date) else len(extended)
            if i < len(extended):
                st_rows.append(dict(ts_code=row.ts_code, start=extended[i],
                                    end=extended[j] if j < len(extended) else pd.NaT))
    st = pd.DataFrame(st_rows, columns=["ts_code", "start", "end"])
    st["start"] = pd.to_datetime(st.start)
    st["end"] = pd.to_datetime(st.end)
    quote = market.pivot(index="date", columns="ts_code", values="close").reindex_like(eligible)
    eligible &= ~st_decision & quote.notna()
    feature_specs = [f"ret{cfg['momentum_window']}", f"ma{cfg['ma_fast']}", f"ma{cfg['ma_slow']}", "macd", "hfq_close"]
    assert cfg["macd"] == [12, 26, 9]
    features_long = hub.cn.features(clean, start=start, end=end, specs=feature_specs,
                                    n_jobs=cfg["feature_workers"])
    features = {column: features_long.pivot(index="date", columns="ts_code", values=column).reindex_like(eligible)
                for column in features_long.columns if column not in ["date", "ts_code"]}
    # 实际指标与独立 pandas 公式对账，逐股使用冻结行情的历史前缀。
    errors = []
    for code in clean:
        g = history[history.ts_code.eq(code)].set_index("date").sort_index()
        price = g.hfq_close
        dif = price.ewm(span=12, adjust=False).mean() - price.ewm(span=26, adjust=False).mean()
        manual = {f"ret{cfg['momentum_window']}": price.pct_change(cfg["momentum_window"], fill_method=None),
                  f"ma{cfg['ma_fast']}": price.rolling(cfg["ma_fast"]).mean(),
                  f"ma{cfg['ma_slow']}": price.rolling(cfg["ma_slow"]).mean(),
                  "macd_dif": dif, "macd_dea": dif.ewm(span=9, adjust=False).mean()}
        for field, value in manual.items():
            actual = features[field][code]
            expected = value.reindex(actual.index)
            np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-9, equal_nan=True)
            errors.append(dict(ts_code=code, field=field, max_error=float((actual - expected).abs().max())))
    pd.DataFrame(errors).to_csv(inputs / "indicator_reconciliation.csv", index=False)
    for name, frame in [("market", market), ("basics", basics), ("st", st), ("limits", limits),
                        ("actions", actions), ("raw_actions", pd.concat(raw_parts)),
                        ("features", features_long), ("eligible", eligible)]:
        frame.to_parquet(inputs / f"{name}.parquet", index=name == "eligible")
    pd.DataFrame(dict(ts_code=clean)).to_csv(inputs / "universe.csv", index=False)
    index = hub.index.daily(cfg["benchmark"], start=start, end=end)
    index.to_parquet(inputs / "benchmark.parquet", index=False)
    preparation = dict(initial_codes=len(codes), kept_codes=len(clean), quarantine_codes=sorted(quarantine),
                       selection_bias="全期质量隔离；仅工程验收，非无偏策略证据", dates=len(calendar),
                       raw_source_hashes=source_hashes,
                       name_rows_without_announcement=int(names.ann_date.isna().sum()))
    save_json(inputs / "preparation.json", preparation)
    save_json(inputs / "frozen_manifest.json", {p.name: digest(p) for p in inputs.iterdir() if p.is_file()})
    return market, basics, st, limits, actions, eligible, features, rules, index, preparation


def load_prepared(prepared, cfg, account, run):
    """复用已归档执行输入；逐文件校验并拒绝改变数据范围或预计算指标口径。"""
    previous = json.loads((prepared / "manifest.json").read_text())
    if previous["status"] != "complete" or previous["data_snapshot"] != account["dataset"]:
        raise ValueError("旧运行未完整通过或冻结数据版本不同")
    keys = ["formation_start", "formation_end", "start", "end", "boards", "universe_size",
            "formation_min_observations", "formation_latest_listing", "quality_policy",
            "momentum_window", "ma_fast", "ma_slow", "macd"]
    if any(previous["config"][key] != cfg[key] for key in keys):
        raise ValueError("复用输入时不能改变股票池/日期/预计算指标参数")
    source = prepared / "inputs"
    frozen = json.loads((source / "frozen_manifest.json").read_text())
    if any(digest(source / name) != value for name, value in frozen.items()):
        raise ValueError("归档执行输入哈希不符")
    old_rule = prepared / "sources" / Path(cfg["action_config"]).name
    if digest(old_rule) != digest(ROOT / cfg["action_config"]):
        raise ValueError("公司行动处理规则已改变，不能混用归档执行输入")
    inputs = run / "inputs"
    shutil.copytree(source, inputs)
    frames = [pd.read_parquet(inputs / f"{name}.parquet")
              for name in ["market", "basics", "st", "limits", "actions", "eligible"]]
    features_long = pd.read_parquet(inputs / "features.parquet")
    eligible = frames[-1]
    features = {column: features_long.pivot(index="date", columns="ts_code", values=column).reindex_like(eligible)
                for column in features_long.columns if column not in ["date", "ts_code"]}
    rules = yaml.safe_load(old_rule.read_text())
    index = pd.read_parquet(inputs / "benchmark.parquet")
    preparation = json.loads((inputs / "preparation.json").read_text())
    return (*frames, features, rules, index, preparation)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/classic_strategy_smoke.yaml")
    parser.add_argument("--prepared", type=Path, help="复用已通过运行的归档输入，不再扫描源执行缓存")
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    account = yaml.safe_load((ROOT / cfg["account_config"]).read_text())
    assert os.environ.get("DATAHUB_CONTENT_ID") == account["dataset"]["content_id"], "须经冻结启动器运行"
    hub = DataHub()
    assert Path(datahub.__file__).resolve().is_relative_to(hub.cfg.root.resolve() / "src")
    run = ROOT / cfg["output_root"] / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    run.mkdir(parents=True)
    print(f"RUN={run}", flush=True)
    sources = run / "sources"
    sources.mkdir()
    source_files = [args.config, ROOT / cfg["account_config"], ROOT / cfg["portfolio_config"],
                    ROOT / cfg["action_config"], Path(__file__), ROOT / "src/classic_smoke.py",
                    ROOT / "src/portfolio.py", ROOT / "scripts/audit_monthly_portfolio.py"]
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in source_files}
    for path in source_files:
        shutil.copy2(path, sources / path.name)
    manifest = dict(status="running", data_snapshot=account["dataset"], source_hashes=hashes,
                    purpose="经典策略工程验收；不是新样本外有效性检验", config=cfg)
    save_json(run / "manifest.json", manifest)
    portfolio_cfg = yaml.safe_load((ROOT / cfg["portfolio_config"]).read_text())["portfolio"]
    portfolio_cfg.update(capital=account["account"]["capital"], top_n=cfg["top_n"], band_mult=1.,
                         buy_cost=cfg["buy_cost"], sell_cost=cfg["sell_cost"], min_fee=cfg["min_fee"],
                         cash_dividend_haircut=cfg["cash_dividend_haircut"], annual_days=cfg["annual_days"])
    try:
        if args.prepared:
            values = load_prepared(args.prepared, cfg, account, run)
            manifest["inputs_reused_from"] = str(args.prepared.resolve())
        else:
            values = prepare_inputs(hub, cfg, account, portfolio_cfg, run)
        market, basics, st, limits, actions, eligible, features, rules, index, preparation = values
        print(f"工程股票池 {preparation['kept_codes']}/{preparation['initial_codes']}；交易日 {len(eligible)}", flush=True)
        variants = [(a, f) for a in cfg["algorithms"] for f in cfg["frequencies"]]
        all_results, summary, yearly, checks, failures = {}, [], [], [], []
        for algorithm, frequency in variants + [("cash", "daily")]:
            tag = f"{algorithm}_{frequency}"
            target = run / tag
            target.mkdir()
            try:
                signals = (make_signals(features, eligible, cfg, algorithm, frequency)
                           if algorithm != "cash" else pd.DataFrame(columns=eligible.columns, dtype=float,
                                                                      index=pd.DatetimeIndex([])))
                signals.to_parquet(target / "signals.parquet")
                top = []
                for date, row in signals.iterrows():
                    for rank, (code, score) in enumerate(row.dropna().sort_values(ascending=False, kind="stable").head(cfg["top_n"]).items(), 1):
                        top.append(dict(signal_date=date, ts_code=code, rank=rank, score=score))
                pd.DataFrame(top, columns=["signal_date", "ts_code", "rank", "score"]).to_csv(target / "targets.csv", index=False)
                result = run_backtest(signals, market, basics, st, limits, actions, cfg["start"], cfg["end"],
                                      portfolio_cfg, rules.get("rights_issues", []), slippage_rate=cfg["slippage_rate"])
                for name, frame in [("daily", result.daily), ("trades", result.trades),
                                    ("events", result.events), ("holdings", result.holdings)]:
                    frame.to_csv(target / f"{name}.csv", index=name == "daily")
                    frame.to_parquet(target / f"{name}.parquet", index=name == "daily")
                audit = verify_ledger(result, portfolio_cfg["capital"], market, limits, actions,
                                      portfolio_cfg, cfg["slippage_rate"], cfg["audit"]["money_atol"])
                # 用较早结束日重跑，历史信号与历史账本必须保持一致。
                cut = pd.Timestamp(cfg["audit"]["future_check_end"])
                short_eligible = eligible.loc[:cut]
                short_features = {k: v.loc[:cut] for k, v in features.items()}
                short_signals = (make_signals(short_features, short_eligible, cfg, algorithm, frequency)
                                 if algorithm != "cash" else signals)
                common = short_signals.index.intersection(signals.index)
                pd.testing.assert_frame_equal(short_signals.loc[common], signals.loc[common])
                # 部分月最后一天不是原月末时，不能把截断日变为额外调仓日；不会在窗口内执行。
                short = run_backtest(short_signals, market.loc[market.date.le(cut)], basics, st, limits, actions,
                                     cfg["start"], cut, portfolio_cfg, rules.get("rights_issues", []),
                                     slippage_rate=cfg["slippage_rate"])
                pd.testing.assert_frame_equal(short.daily, result.daily.loc[short.daily.index])
                if algorithm == "cash":
                    assert result.daily.equity.eq(portfolio_cfg["capital"]).all() and result.trades.empty
                maxret = float(result.daily.ret.abs().max())
                audit.update(strategy=tag, prefix_invariant=True, max_abs_daily_return=maxret,
                             extreme_return_flag=maxret >= cfg["audit"]["extreme_daily_return"])
                assert not audit["extreme_return_flag"], "日收益异常，需要进一步排查"
                checks.append(audit)
                save_json(target / "audit.json", audit)
                report = dict(strategy=tag, **result.summary,
                              risk_pass=bool(np.isfinite(result.summary["mdd"]) and result.summary["mdd"] >= -account["risk"]["max_drawdown"]),
                              average_holdings=float(result.daily.n_hold.mean()),
                              fee_to_initial_capital=result.summary["total_fees"] / portfolio_cfg["capital"],
                              engineering_pass=True)
                summary.append(report)
                all_results[tag] = result
                prior = portfolio_cfg["capital"]
                for year, g in result.daily.groupby(result.daily.index.year):
                    metrics, _ = daily_metrics(g.equity, prior, cfg["annual_days"])
                    yearly.append(dict(strategy=tag, year=year, **metrics))
                    prior = g.equity.iloc[-1]
                print(f"{tag}: 收益 {report['total']:.2%}, MDD {report['mdd']:.2%}, 交易 {report['n_trades']}, 对账通过", flush=True)
            except Exception as exc:
                failures.append(dict(strategy=tag, error_type=type(exc).__name__, error=str(exc)))
                save_json(target / "failure.json", failures[-1])
                print(f"{tag}: FAILED {type(exc).__name__}: {exc}", flush=True)
        pd.DataFrame(summary).to_csv(run / "summary.csv", index=False)
        pd.DataFrame(yearly).to_csv(run / "yearly.csv", index=False)
        pd.DataFrame(checks).to_csv(run / "process_checks.csv", index=False)
        save_json(run / "failures.json", failures)
        if all_results:
            nav = pd.DataFrame({k: v.daily.nav for k, v in all_results.items()})
            benchmark = index.set_index("date").close.reindex(nav.index)
            assert benchmark.notna().all()
            nav["CSI300_price_reference"] = benchmark / benchmark.iloc[0]
            nav.to_csv(run / "nav_comparison.csv")
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
            for name in nav:
                axes[0].plot(nav.index, nav[name], label=name, linewidth=1)
                dd = nav[name] / np.maximum.accumulate(np.r_[1., nav[name]])[1:] - 1
                axes[1].plot(nav.index, dd, linewidth=1)
            axes[0].set_ylabel("NAV (initial = 1)")
            axes[0].legend(fontsize=8, ncol=3)
            axes[1].axhline(-account["risk"]["max_drawdown"], color="red", linestyle="--")
            axes[1].set_ylabel("Drawdown")
            fig.suptitle("Classic strategy pipeline smoke test — quality-screened sample")
            fig.tight_layout()
            fig.savefig(run / "nav_drawdown.png", dpi=150)
            plt.close(fig)
        assert all(digest(ROOT / path) == value for path, value in hashes.items()), "实验期间代码或配置改变"
        frozen = json.loads((run / "inputs/frozen_manifest.json").read_text())
        assert all(digest(run / "inputs" / path) == value for path, value in frozen.items()), "输入改变"
        manifest.update(status="complete" if not failures else "completed_with_failures", preparation=preparation,
                        failed_strategies=failures, strategies_passed=len(summary))
    except Exception as exc:
        manifest.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        save_json(run / "manifest.json", manifest)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

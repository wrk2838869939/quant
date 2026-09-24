"""日频现金/股数账本：收盘信号、次日开盘交易、分红权益独立记账。"""
from dataclasses import dataclass, field
import math

import numpy as np
import pandas as pd


class AccountingError(RuntimeError):
    """缺少关键执行或公司行动数据时停止，不返回看似有效的收益。"""


@dataclass
class Position:
    shares: int
    last_price: float
    last_adj: float
    locks: list = field(default_factory=list)
    stale_days: int = 0
    action_since_quote: bool = False

    def available(self, date):
        return self.shares - sum(q for q, release in self.locks if date < release)


@dataclass
class BacktestResult:
    daily: pd.DataFrame
    trades: pd.DataFrame
    events: pd.DataFrame
    holdings: pd.DataFrame
    summary: dict


def board_of(code):
    if code.endswith(".BJ"):
        return "beijing"
    if code.startswith("68"):
        return "star"
    if code.startswith("30"):
        return "chinext"
    return "main"


def affordable_shares(budget, price, rate, minimum_fee, rule, volume_cap=np.inf, extra_rate=0.):
    """手续费包含在预算内，按板块起买量和步长取整。"""
    if not np.isfinite(price) or price <= 0 or budget <= minimum_fee:
        return 0
    maximum = min(budget / (price * (1 + rate + extra_rate)),
                  (budget - minimum_fee) / (price * (1 + extra_rate)), volume_cap)
    if maximum < rule["min_buy"]:
        return 0
    return int(rule["min_buy"] + math.floor((maximum - rule["min_buy"]) / rule["buy_step"]) * rule["buy_step"])


def daily_metrics(equity, capital, annual_days):
    """按实际观察的交易日数年化；回撤峰值包含初始资金。"""
    returns = equity.pct_change(fill_method=None)
    returns.iloc[0] = equity.iloc[0] / capital - 1
    peaks = np.maximum.accumulate(np.r_[capital, equity.to_numpy()])[1:]
    total = equity.iloc[-1] / capital - 1
    std = returns.std(ddof=1)
    return dict(total=float(total), ann=float((1 + total) ** (annual_days / len(equity)) - 1)
                if total > -1 else -1.0,
                mdd=float(np.min(equity.to_numpy() / peaks - 1)),
                sharpe=float(returns.mean() / std * np.sqrt(annual_days)) if std > 0 else np.nan,
                n_days=len(equity)), returns


def _apply_source_corrections(frame, source_corrections):
    """凭公告及完整原始键修正已核验接口错误；在任何去重前严格核对。"""
    out = frame.copy(deep=True)
    required = {"ts_code", "end_date", "ann_date", "imp_ann_date", "ex_date",
                "record_date", "pay_date", "div_proc"}
    date_keys = required - {"ts_code", "div_proc"}
    seen, audit = set(), []
    for rule in source_corrections:
        try:
            match = rule["match"]
            expected, replacement = rule["expected_values"], rule["replacement_values"]
            if set(match) != required or match["div_proc"] != "实施":
                raise ValueError("必须提供完整已实施原始行匹配键")
            dates = {key: pd.Timestamp(str(match[key])) for key in date_keys}
            announcement = pd.Timestamp(str(rule["source_ann_date"]))
            if (any(pd.isna(value) for value in dates.values()) or pd.isna(announcement)
                    or announcement > dates["ex_date"]
                    or dates["ann_date"] > dates["ex_date"]
                    or dates["imp_ann_date"] > dates["ex_date"]):
                raise ValueError("修正公告或原始公告日期不得晚于除权日")
            if not rule.get("id") or not rule.get("evidence") or not rule.get("check"):
                raise ValueError("修正必须保留标识、公告来源和核对说明")
            if (not replacement or not set(replacement).issubset({"cash_div", "cash_div_tax"})
                    or not set(replacement).issubset(expected)):
                raise ValueError("修正只允许有预期旧值的现金分红字段")
            for values in [expected, replacement]:
                if any(not np.isfinite(float(value)) for value in values.values()):
                    raise ValueError("修正前后值必须为有限数值")
            key = tuple((name, str(dates[name]) if name in dates else match[name]) for name in sorted(required))
            if key in seen:
                raise ValueError("同一原始行不能登记重复修正")
            seen.add(key)
        except (KeyError, TypeError, ValueError) as exc:
            raise AccountingError(f"公司行动源修正规则无效: {exc}") from exc
        # 数据子集可以不含该股票；一旦含该股票，缺行、改键或重复均必须报错。
        if out.empty or not out.ts_code.eq(match["ts_code"]).any():
            continue
        if not (required | set(expected)).issubset(out.columns):
            raise AccountingError(f"{rule['id']}: 原始公司行动缺少核验列")
        mask = out.ts_code.eq(match["ts_code"])
        for name, value in match.items():
            if name in date_keys:
                actual = pd.to_datetime(out[name].astype(str), errors="coerce", format="mixed")
                mask &= actual.eq(dates[name])
            else:
                mask &= out[name].eq(value)
        if int(mask.sum()) != 1:
            raise AccountingError(f"{rule['id']}: 原始修正行匹配数应为1，实际{int(mask.sum())}")
        row = out.loc[mask].iloc[0]
        for name, value in expected.items():
            try:
                actual = float(row[name])
            except (TypeError, ValueError) as exc:
                raise AccountingError(f"{rule['id']}: {name} 原始值不是有限数值") from exc
            if not np.isfinite(actual) or actual != float(value):
                raise AccountingError(f"{rule['id']}: {name} 原始值改变，预期{value}，实际{row[name]}")
        for name, value in replacement.items():
            out.loc[mask, name] = float(value)
        audit.append({"id": rule["id"], "match": dict(match),
                      "expected_values": dict(expected), "replacement_values": dict(replacement),
                      "source_ann_date": str(rule["source_ann_date"]), "evidence": rule["evidence"],
                      "check": rule["check"], "matched_rows": 1})
    return out, audit


def normalize_actions(frame, component_events=(), reference_events=(), source_corrections=()):
    """核验显式源修正后合并已实施分红；保留修正审计，不改原缓存或推猜权益。"""
    frame, correction_audit = _apply_source_corrections(frame, source_corrections)
    if frame.empty:
        return pd.DataFrame(columns=["ts_code", "record_date", "ex_date", "pay_date",
                                     "div_listdate", "cash_div_tax", "stk_div"])
    out = frame[frame["div_proc"] == "实施"].copy()
    for col in ("record_date", "ex_date", "pay_date", "div_listdate"):
        out[col] = pd.to_datetime(out[col], errors="coerce")
    for col in ("cash_div_tax", "stk_div"):
        out[col] = pd.to_numeric(out[col], errors="coerce")
    # 厂商同事件修订重复: 除现金分红外全部字段精确一致、现金相对差≤0.1% 的多行,
    # 按厂商行序保留最后修订值并显式审计; 任何其他差异仍走组件核验或报错, 不静默合并。
    revision_audit = []
    merged_idx = []
    for key, g in out.groupby(["ts_code", "ex_date"], dropna=False):
        if len(g) < 2:
            continue
        same = all(g[c].astype(str).where(g[c].notna(), "<NA>").nunique() == 1
                   for c in ("end_date", "ann_date", "imp_ann_date", "record_date",
                             "pay_date", "div_listdate", "stk_div") if c in g.columns)
        cash = g["cash_div_tax"]
        tol_ok = (cash.notna().all() and float(cash.max() - cash.min()) > 0
                  and float(cash.max() - cash.min()) <= 1e-3 * max(1.0, abs(float(cash.max()))))
        if same and tol_ok:
            merged_idx.extend(g.index[:-1])
            revision_audit.append({
                "kind": "vendor_revision_duplicate", "ts_code": key[0],
                "ex_date": str(pd.Timestamp(key[1]).date()), "n_rows": int(len(g)),
                "cash_div_tax_values": [float(v) for v in cash],
                "kept": float(g["cash_div_tax"].iloc[-1]),
                "check": "非现金字段全部精确一致且现金相对差≤0.1%, 按厂商行序保留最后修订值"})
    if merged_idx:
        out = out.drop(index=merged_idx)
    correction_audit.extend(revision_audit)
    cols = ["ts_code", "record_date", "ex_date", "pay_date", "div_listdate", "cash_div_tax", "stk_div"]
    out = out[cols].drop_duplicates()
    approved = {(e["ts_code"], pd.Timestamp(e["ex_date"])): e for e in component_events}
    rows = []
    for key, group in out.groupby(["ts_code", "ex_date"], dropna=False):
        if len(group) == 1:
            rows.append(group.iloc[0].to_dict())
            continue
        rule = approved.get(key)
        if rule is None or len(group) != rule["component_count"]:
            raise AccountingError(f"{key}: 同股票同除权日存在未核验的多项分红")
        cash, bonus = group.cash_div_tax.sum(), group.stk_div.sum()
        if (group[["cash_div_tax", "stk_div"]].isna().any().any()
                or not np.isclose(cash, rule["cash_div_tax"], atol=1e-10, rtol=0)
                or not np.isclose(bonus, rule["stk_div"], atol=1e-10, rtol=0)):
            raise AccountingError(f"{key}: 分红组成与核验配置不符")
        row = group.iloc[0].to_dict()
        for col, relevant in [("record_date", group), ("pay_date", group[group.cash_div_tax > 0]),
                               ("div_listdate", group[group.stk_div > 0])]:
            values = relevant[col].dropna().unique()
            if len(relevant) and (len(values) != 1 or relevant[col].isna().any()):
                raise AccountingError(f"{key}: 分红组成的 {col} 不一致")
            row[col] = pd.Timestamp(values[0]) if len(values) else pd.NaT
        row.update(cash_div_tax=cash, stk_div=bonus)
        rows.append(row)
    result = pd.DataFrame(rows, columns=cols).sort_values("ex_date").reset_index(drop=True)
    result["reference_share_fraction"] = 1.0
    seen = set()
    for rule in reference_events:
        key = (rule["ts_code"], pd.Timestamp(rule["ex_date"]))
        fraction = rule["participating_shares"] / rule["total_shares"]
        if (key in seen or not 0 < fraction <= 1 or not rule.get("evidence")
                or pd.Timestamp(rule["ann_date"]) > key[1]):
            raise AccountingError(f"{key}: 差异化除权依据不合法")
        seen.add(key)
        mask = result.ts_code.eq(key[0]) & result.ex_date.eq(key[1])
        if not mask.any():
            continue
        row = result.loc[mask].iloc[0]
        if (not np.isclose(row.cash_div_tax, rule["cash_div_tax"], atol=1e-10, rtol=0)
                or not np.isclose(row.stk_div, rule["stk_div"], atol=1e-10, rtol=0)):
            raise AccountingError(f"{key}: 差异化分红实际权益与公告不符")
        # 除权参考价按总股本摊薄；投资者实际现金及红股仍按参与分配的每股数额。
        result.loc[mask, "reference_share_fraction"] = fraction
    result.attrs["source_corrections"] = correction_audit
    return result


def run_backtest(signals, market, basics, st_intervals, limits, actions, start, end, cfg,
                 rights_issues=(), *, risk_allowed=None, risk_signal_max_age=None,
                 retail_forecasts=None, retail_config=None, extra_execution_delay=0,
                 reentry_signals=None, reentry_forecasts=None, slippage_rate=0.,
                 reentry_on_schedule_only=False, hold_allowed=None, top_n_schedule=None):
    """运行单一固定方案，输出每日净值、逐笔现金账和事件。

    market: 日期×代码长表，含 open/close/pre_close/adj_factor/vol_share。
    limits: 同样的长表，含 up_limit/down_limit；缺失时保守拒绝成交。
    actions: normalize_actions 后的分红事件；摘牌按 cfg 的回收率减记。
    hold_allowed: 与 signals 对齐的可选持有许可；True/缺失保留旧仓，False确认退出。
        仅用于整手分配；保留不允许补仓，必要时按单股风险上限整手减仓。
    """
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    market = market[(market.date >= start) & (market.date <= end)].copy()
    if market.duplicated(["date", "ts_code"]).any():
        raise AccountingError("行情日期/代码重复")
    dates = pd.DatetimeIndex(sorted(market.date.unique()))
    if len(dates) == 0 or cfg["capital"] <= 0:
        raise ValueError("回测窗口为空或初始资金非正")
    if signals.index.has_duplicates or signals.columns.has_duplicates:
        raise AccountingError("信号日期/代码重复")
    signals = signals.sort_index()
    # 策略在收盘时给出持仓数量上限日程，便于换参数时连续保留现金、旧仓和公司行动。
    # 不提供日程时完全沿用既有固定上限；成交、费用与账户处理均不改变。
    if top_n_schedule is not None:
        if (not isinstance(top_n_schedule, pd.Series)
                or not top_n_schedule.index.equals(signals.index)
                or top_n_schedule.isna().any()
                or not np.isfinite(top_n_schedule).all()
                or not (top_n_schedule == top_n_schedule.astype(int)).all()
                or not top_n_schedule.between(1, 5).all()
                or retail_forecasts is not None or hold_allowed is not None):
            raise ValueError('动态持股数须与信号日对齐、为1至5整数，且仅用于简单排名分配')
    if hold_allowed is not None:
        if (not isinstance(hold_allowed, pd.DataFrame)
                or not hold_allowed.index.equals(signals.index)
                or not hold_allowed.columns.equals(signals.columns)
                or not (hold_allowed.isna() | hold_allowed.isin([True, False])).all().all()
                or retail_forecasts is None or retail_config is None
                or reentry_signals is not None):
            raise ValueError('持有许可必须与信号对齐，并使用同日历整手分配')
        max_positions = retail_config.get('max_positions')
        if (max_positions is None or int(max_positions) != max_positions
                or not 0 <= max_positions <= 5 or cfg['top_n'] > 5):
            raise ValueError('耐心持有模式必须显式限定最多5只股票')
    if int(extra_execution_delay) != extra_execution_delay or extra_execution_delay < 0:
        raise ValueError("额外执行延迟必须为非负整数")
    if not np.isfinite(slippage_rate) or slippage_rate < 0:
        raise ValueError("附加成交价差率必须有限且非负")
    if not isinstance(reentry_on_schedule_only, (bool, np.bool_)):
        raise ValueError('恢复入场的日历约束必须为布尔值')
    if retail_forecasts is not None:
        from retail_rebalance import plan_rebalance
        if (retail_config is None or not 0 < retail_config['max_stock_fraction'] <= 1
                or not retail_forecasts.index.equals(signals.index)
                or not retail_forecasts.columns.equals(signals.columns)):
            raise ValueError("整手分配需要明确风险上限及与信号对齐的TRAIN收益预测")
        selection_first = retail_config.get('selection_first', False)
        if not isinstance(selection_first, (bool, np.bool_)) or (selection_first and hold_allowed is None):
            raise ValueError('先选股模式必须使用显式布尔开关和独立持有许可')
    if reentry_signals is not None:
        if (reentry_signals.index.has_duplicates or not reentry_signals.index.is_monotonic_increasing
                or not reentry_signals.columns.equals(signals.columns)
                or reentry_forecasts is None or not reentry_forecasts.index.equals(reentry_signals.index)
                or not reentry_forecasts.columns.equals(signals.columns)):
            raise ValueError("恢复入场需要按日期排序的每日信号及对应预测")
    if risk_allowed is not None:
        if (risk_allowed.index.has_duplicates or not risk_allowed.index.is_monotonic_increasing
                or not risk_allowed.isin([0, 1]).all()):
            raise AccountingError("风险计划必须是按日期排序、无缺失的0/1收盘决策")
        risk_allowed = risk_allowed.reindex(dates)
        if risk_allowed.isna().any():
            raise AccountingError("风险计划未覆盖全部交易日，禁止默认安全")
        if (risk_signal_max_age is None or int(risk_signal_max_age) != risk_signal_max_age
                or risk_signal_max_age < 1):
            raise ValueError("必须显式给出重入时选股信号的有效交易日数")
    panels = {col: market.pivot(index="date", columns="ts_code", values=col).reindex(dates)
              for col in ("open", "close", "pre_close", "adj_factor", "vol_share")}
    codes = panels["close"].columns
    arrays = {k: v.to_numpy(dtype=float) for k, v in panels.items()}
    code_idx = {code: j for j, code in enumerate(codes)}
    day_idx = {date: i for i, date in enumerate(dates)}
    price_limits = limits.set_index(["date", "ts_code"])[["up_limit", "down_limit"]].to_dict("index")
    if limits.duplicated(["date", "ts_code"]).any():
        raise AccountingError("每日涨跌停价格重复")
    listings = basics.set_index("ts_code")["list_date"].to_dict()
    delistings = basics.dropna(subset=["delist_date"]).set_index("ts_code")["delist_date"].to_dict()
    record_events, ex_events = {}, {}
    rights_by_date = {}
    for right in rights_issues:
        if right["policy"] != "decline":
            raise AccountingError("当前仅实现不参与配股情景，不能隐式追加资金")
        rights_by_date.setdefault(pd.Timestamp(right["ex_date"]), []).append(right)
    for event_id, row in actions.iterrows():
        event = row.to_dict()
        event["id"] = event_id
        record_events.setdefault(event["record_date"], []).append(event)
        ex_events.setdefault(event["ex_date"], []).append(event)
    execution_days = {}
    for signal_date in signals.index:
        # 只使用行情日历中 T 的下一交易日，不按信号文件的行号二次采样。
        if signal_date in day_idx and day_idx[signal_date] + 1 + extra_execution_delay < len(dates):
            execution_days[dates[day_idx[signal_date] + 1 + extra_execution_delay]] = signal_date
    cash = float(cfg["capital"])
    positions, pending_sell, entitlements = {}, {}, {}
    confirmed_hold_exits = set()
    receivables, daily, trades, events, holdings = [], [], [], [], []
    top_n = cfg["top_n"]
    risk_exit_active, reentry_due = False, False
    previous_gate = 1

    def event(date, code, kind, **detail):
        events.append(dict(date=date, ts_code=code, kind=kind, **detail))

    def st_set(date):
        active = ((st_intervals.start <= date)
                  & (st_intervals.end.isna() | (st_intervals.end >= date)))
        return set(st_intervals.loc[active, "ts_code"])

    def quote(i, code, col):
        j = code_idx.get(code)
        return arrays[col][i, j] if j is not None else np.nan

    def allowed(i, date, code, side):
        op, volume = quote(i, code, "open"), quote(i, code, "vol_share")
        if not np.isfinite(op) or op <= 0 or not np.isfinite(volume) or volume <= 0:
            return False, "suspended"
        bound = price_limits.get((date, code))
        if bound is None or not all(np.isfinite(v) and v > 0 for v in bound.values()):
            if cfg["require_official_limits"]:
                event(date, code, "missing_limit", side=side)
                return False, "missing_limit"
        elif side == "buy" and op >= bound["up_limit"] - cfg["price_tick"] / 2:
            return False, "limit_up"
        elif side == "sell" and op <= bound["down_limit"] + cfg["price_tick"] / 2:
            return False, "limit_down"
        return True, "ok"

    for i, date in enumerate(dates):
        # 已确认的权益与支付独立：卖出股票不会丢失登记日获得的分红。
        for action in ex_events.get(date, []):
            code = action["ts_code"]
            shares = entitlements.pop(action["id"], 0)
            if not shares:
                continue
            div, bonus = action["cash_div_tax"], action["stk_div"]
            if not np.isfinite(div) or not np.isfinite(bonus) or min(div, bonus) < 0:
                raise AccountingError(f"{date.date()} {code}: 分红字段缺失或为负")
            pos = positions.get(code)
            if pos is None:
                raise AccountingError(f"{date.date()} {code}: 登记后除权前已卖出，需独立红股权益处理")
            reference_fraction = action.get("reference_share_fraction", 1.0)
            if not np.isfinite(reference_fraction) or not 0 < reference_fraction <= 1:
                raise AccountingError(f"{date.date()} {code}: 除权参考参数无效")
            reference_div = div * reference_fraction
            reference_bonus = bonus * reference_fraction
            reference = (pos.last_price - reference_div) / (1 + reference_bonus)
            pc = quote(i, code, "pre_close")
            if np.isfinite(pc) and not np.isclose(reference, pc, atol=cfg["action_price_atol"], rtol=cfg["action_price_rtol"]):
                raise AccountingError(f"{date.date()} {code}: 分红金额与除权参考价不符 ({reference:.4f} vs {pc:.4f})")
            pos.last_price = reference
            pos.action_since_quote = True
            if bonus:
                release = action["div_listdate"]
                if pd.isna(release) or release < date:
                    raise AccountingError(f"{date.date()} {code}: 红股可卖日期未知")
                exact = shares * bonus
                added = int(math.floor(exact + 1e-8))
                pos.shares += added
                pos.locks.append((added, release))
                event(date, code, "bonus_shares", shares=added, fractional_discard=exact - added)
            if div:
                pay = action["pay_date"]
                if pd.isna(pay) or pay < date:
                    raise AccountingError(f"{date.date()} {code}: 分红到账日期未知")
                amount = shares * div * (1 - cfg["cash_dividend_haircut"])
                receivables.append(dict(ts_code=code, amount=amount, pay_date=pay))
                event(date, code, "dividend_receivable", amount=amount)
        waiting = []
        for claim in receivables:
            if claim["pay_date"] <= date:
                cash += claim["amount"]
                event(date, claim["ts_code"], "dividend_paid", amount=claim["amount"])
            else:
                waiting.append(claim)
        receivables = waiting

        for right in rights_by_date.get(date, []):
            code = right["ts_code"]
            pos = positions.get(code)
            if pos is None:
                continue
            reference = (pos.last_price + right["ratio"] * right["subscription_price"]) / (1 + right["ratio"])
            pc = quote(i, code, "pre_close")
            if not np.isfinite(pc) or not np.isclose(reference, pc, atol=cfg["action_price_atol"], rtol=cfg["action_price_rtol"]):
                raise AccountingError(f"{date.date()} {code}: 配股方案与除权参考价不符")
            event(date, code, "rights_declined", reference_loss=pos.shares * (pos.last_price - pc))
            pos.last_price = pc
            pos.action_since_quote = True

        for code, pos in list(positions.items()):
            if date >= delistings.get(code, pd.Timestamp.max):
                recovery = pos.shares * pos.last_price * cfg["delist_recovery"]
                cash += recovery
                event(date, code, "delist_writeoff", shares=pos.shares,
                      amount=pos.shares * pos.last_price - recovery, cash_recovery=recovery)
                del positions[code]
                pending_sell.pop(code, None)
                confirmed_hold_exits.discard(code)
                continue
            adj = quote(i, code, "adj_factor")
            reference = quote(i, code, "pre_close")
            atol = cfg["action_price_atol"] if pos.action_since_quote else cfg["price_tick"] / 2
            rtol = cfg["action_price_rtol"] if pos.action_since_quote else 0
            if np.isfinite(reference) and not np.isclose(reference, pos.last_price, atol=atol, rtol=rtol):
                # 即使复权列漏更新，原始参考价的无解释跳变仍须拦截。
                if cfg["strict_actions"]:
                    raise AccountingError(f"{date.date()} {code}: 前收参考价与持仓估值不连续")
                event(date, code, "unexplained_adjustment", previous_price=pos.last_price, reference=reference)
            if np.isfinite(adj) and adj > 0:
                changed = not np.isclose(adj, pos.last_adj, rtol=cfg["adjustment_rtol"], atol=0)
                if changed and not pos.action_since_quote:
                    if np.isfinite(reference) and np.isclose(reference, pos.last_price,
                                                             atol=cfg["price_tick"] / 2, rtol=0):
                        # 原始参考价连续：只记录上游复权元数据变动，不改变现金或股数。
                        event(date, code, "adjustment_metadata_change", old_adj=pos.last_adj, new_adj=adj)
                    else:
                        event(date, code, "unexplained_adjustment", old_adj=pos.last_adj, new_adj=adj)
                        if cfg["strict_actions"]:
                            raise AccountingError(f"{date.date()} {code}: 参考价不连续且无对应公司行动")
                pos.last_adj = adj
                pos.action_since_quote = False

        st_today = st_set(date)
        signal_date = execution_days.get(date)
        signal_source, forecast_source = signals, retail_forecasts
        decision_idx = i - 1 - extra_execution_delay
        gate = int(risk_allowed.iloc[decision_idx]) if risk_allowed is not None and decision_idx >= 0 else 1
        risk_date = dates[decision_idx] if decision_idx >= 0 else pd.NaT
        if risk_allowed is not None:
            if gate != previous_gate:
                event(date, "__portfolio__", "risk_transition", signal_date=risk_date,
                      target=gate, previous_target=previous_gate)
            if gate == 0:
                risk_exit_active = bool(positions) or risk_exit_active
                reentry_due = False
            elif previous_gate == 0:
                reentry_due = True
            previous_gate = gate
        liquidation_today = risk_exit_active or gate == 0
        if liquidation_today:
            signal_date = None
            for code in positions:
                pending_sell.setdefault(code, risk_date)
                if hold_allowed is not None:
                    if code not in confirmed_hold_exits:
                        event(date, code, 'hold_exit_requested', signal_date=risk_date,
                              reason='portfolio_risk', old_shares=positions[code].shares)
                    confirmed_hold_exits.add(code)
        elif reentry_due and signal_date is None and reentry_on_schedule_only:
            event(date, '__portfolio__', 'reentry_wait', reason='scheduled_review_required')
        elif reentry_due and signal_date is None:
            if reentry_signals is not None:
                signal_source, forecast_source = reentry_signals, reentry_forecasts
            past = signal_source.index[signal_source.index <= risk_date]
            if len(past):
                latest = past[-1]
                if latest in day_idx and i - day_idx[latest] <= risk_signal_max_age:
                    signal_date = latest
                else:
                    event(date, "__portfolio__", "reentry_wait", reason="stale_selection_signal")
        buy_codes = []
        planned_buys, planned_sells = {}, {}
        if signal_date is not None:
            if top_n_schedule is not None:
                top_n = int(top_n_schedule.loc[signal_date])
            ranked = signal_source.loc[signal_date].replace([np.inf, -np.inf], np.nan).dropna().sort_values(ascending=False, kind="stable")
            keep = set(ranked.head(int(top_n * cfg["band_mult"])).index)
            buy_codes = list(ranked.head(top_n).index)
            if reentry_due:
                event(date, "__portfolio__", "reentry_attempt", signal_date=signal_date)
                reentry_due = False
            if retail_forecasts is None:
                for code in positions:
                    if code not in keep:
                        pending_sell[code] = signal_date
                    else:
                        pending_sell.pop(code, None)
            else:
                candidates = list(ranked.head(retail_config['candidate_pool']).index)
                universe = list(dict.fromkeys(candidates + list(positions)))
                opening_equity = cash + sum(pos.shares * (quote(i, code, 'open')
                    if np.isfinite(quote(i, code, 'open')) and quote(i, code, 'open') > 0
                    else pos.last_price) for code, pos in positions.items())
                records = []
                for code in universe:
                    pos = positions.get(code)
                    price = quote(i, code, 'open')
                    if not np.isfinite(price) or price <= 0:
                        if pos is None:
                            continue
                        price = pos.last_price
                    rule = cfg['boards'][board_of(code)]
                    listing = listings.get(code, pd.NaT)
                    eligible = (pd.notna(listing) and (signal_date-listing).days >= cfg['min_listing_calendar_days']
                        and date >= pd.Timestamp(rule['start']) and date < delistings.get(code, pd.Timestamp.max)
                        and code not in st_today and code not in st_set(signal_date))
                    buy_ok, buy_reason = allowed(i, date, code, 'buy')
                    sell_ok, sell_reason = allowed(i, date, code, 'sell') if pos else (True, 'no_position')
                    mu = forecast_source.loc[signal_date].get(code, np.nan)
                    if not np.isfinite(mu):
                        # 未知旧仓收益不冒充正优势；不允许新建未知预测仓位。
                        mu = 0.
                        buy_ok = False
                    if code in st_today:
                        mu = -1.
                    volume = quote(i-1, code, 'vol_share') if i else 0.
                    old_shares = pos.shares if pos else 0
                    capacity = max(0., volume * cfg['max_volume_participation']) if np.isfinite(volume) else 0.
                    cap = min(opening_equity * retail_config['max_stock_fraction'],
                              (old_shares + capacity) * price)
                    minimum = 0
                    if hold_allowed is not None and pos is not None:
                        state = hold_allowed.loc[signal_date, code]
                        # 一经确认退出，后续趋势恢复不能撤销尚未完成的卖单。
                        requested = (code in confirmed_hold_exits or code in pending_sell
                            or code in st_today or (pd.notna(state) and not bool(state)))
                        buy_ok = False
                        if requested:
                            reason = ('st' if code in st_today else 'confirmed_pending'
                                if code in confirmed_hold_exits or code in pending_sell else 'hold_false')
                            confirmed_hold_exits.add(code)
                            pending_sell.setdefault(code, signal_date)
                            cap = 0.
                            event(date, code, 'hold_exit_requested', signal_date=signal_date,
                                  reason=reason, old_shares=old_shares)
                        else:
                            available = pos.available(date) if sell_ok else 0
                            locked = old_shares - available
                            cap_shares = max(locked, min(old_shares,
                                int(math.floor(cap / price + 1e-9))))
                            required = old_shares - cap_shares
                            step = rule['buy_step']
                            # 可卖量的零股允许单独出售；从两种可达余数中选最小必要卖量。
                            odd = available % step
                            sales = [offset + max(0, math.ceil((required-offset)/step))*step
                                     for offset in {0, odd}]
                            minimum = old_shares - min(q for q in sales if required <= q <= available)
                            event(date, code, 'hold_retained', signal_date=signal_date,
                                  reason='unknown' if pd.isna(state) else 'trend_allowed',
                                  old_shares=old_shares, min_target_shares=minimum,
                                  risk_cap=cap, unavoidable_locked_excess=max(0., locked*price-cap))
                            if old_shares * price > cap + 1e-8:
                                event(date, code, 'hold_risk_cap_reduction', signal_date=signal_date,
                                      old_shares=old_shares, target_shares=minimum,
                                      sell_shares=old_shares-minimum,
                                      reason='locked_or_untradable' if locked*price>cap else 'single_stock_cap')
                    records.append(dict(ts_code=code, price=price, expected_return=mu,
                        shares=old_shares, available=pos.available(date) if pos else 0,
                        min_buy=rule['min_buy'], buy_step=rule['buy_step'],
                        sell_step=rule['buy_step'], max_notional=cap,
                        min_target_shares=minimum,
                        can_buy=bool(buy_ok and eligible and code in candidates), can_sell=bool(sell_ok)))
                if records:
                    allocator = plan_rebalance
                    selection_kwargs = {}
                    if retail_config.get('selection_first', False):
                        from selection_first import plan_selection_first
                        allocator = plan_selection_first
                        selection_kwargs['priorities'] = ranked.to_dict()
                    allocation = allocator(pd.DataFrame(records), cash=cash,
                        buy_rate=cfg['buy_cost'], sell_rate=cfg['sell_cost'], min_fee=cfg['min_fee'],
                        forecast_kind='expected_simple_return', slippage_rate=slippage_rate,
                        solver_relative_gap=retail_config.get('solver_relative_gap', .001),
                        max_positions=retail_config.get('max_positions'), **selection_kwargs)
                    if retail_config.get('selection_first', False):
                        for decision in allocation['selection_audit'].to_dict('records'):
                            code = decision.pop('ts_code')
                            event(date, code, 'selection_decision', signal_date=signal_date, **decision)
                    for row in allocation['plan'].itertuples():
                        planned_buys[row.ts_code] = row.buy_shares
                        planned_sells[row.ts_code] = row.sell_shares
                        if row.sell_shares:
                            if row.ts_code in confirmed_hold_exits:
                                pending_sell.setdefault(row.ts_code, signal_date)
                            else:
                                pending_sell[row.ts_code] = signal_date
                        event(date, row.ts_code, 'allocation_decision', signal_date=signal_date,
                              expected_return=row.expected_return, old_shares=row.shares,
                              target_shares=row.target_shares, buy_shares=row.buy_shares,
                              sell_shares=row.sell_shares, risk_cap=row.max_notional,
                              min_target_shares=row.min_target_shares,
                              unavoidable_locked_excess=row.unavoidable_locked_excess,
                              solver_gap=allocation['solver_gap'], solver_absolute_gap=allocation['solver_absolute_gap'])
                    buy_codes = [code for code, qty in planned_buys.items() if qty > 0]
        for code in positions:
            if code in st_today:
                pending_sell.setdefault(code, date)
                if hold_allowed is not None:
                    if code not in confirmed_hold_exits:
                        event(date, code, 'hold_exit_requested', signal_date=date,
                              reason='st', old_shares=positions[code].shares)
                    confirmed_hold_exits.add(code)

        # 卖出失败每天顺延，而不是等下一次月度调仓；当日新买股尚未进入此循环。
        for code, order_date in list(pending_sell.items()):
            pos = positions.get(code)
            if pos is None:
                pending_sell.pop(code, None)
                confirmed_hold_exits.discard(code)
                continue
            available = pos.available(date)
            if code in planned_sells:
                available = min(available, planned_sells[code])
            ok, reason = allowed(i, date, code, "sell")
            if not ok or available <= 0:
                event(date, code, "sell_deferred", reason=reason if not ok else "locked")
                continue
            price = quote(i, code, "open")
            gross = available * price
            fee = min(gross, max(cfg["min_fee"], gross * cfg["sell_cost"]))
            fee += gross * slippage_rate
            cash += gross - fee
            pos.shares -= available
            trades.append(dict(date=date, signal_date=order_date, ts_code=code, side="sell",
                               shares=available, price=price, gross=gross, fee=fee, cash_after=cash))
            if slippage_rate:
                trades[-1]['slippage_cost'] = gross * slippage_rate
            if pos.shares == 0:
                del positions[code]
                pending_sell.pop(code, None)
                confirmed_hold_exits.discard(code)
            elif code in planned_sells and code not in confirmed_hold_exits:
                pending_sell.pop(code, None)
        if risk_exit_active and not positions:
            risk_exit_active = False
            event(date, "__portfolio__", "risk_exit_complete", target=gate)
            # 撤退尚未完成时恢复信号也不能取消未卖订单；完成后下一日再考虑重入。
            reentry_due = gate == 1
        slots = max(0, top_n - len(positions))
        budget = cash / slots if slots else 0
        for code in buy_codes:
            if retail_forecasts is None and (not slots or code in positions):
                continue
            rule = cfg["boards"][board_of(code)]
            listing = listings.get(code, pd.NaT)
            if pd.isna(listing) or (signal_date - listing).days < cfg["min_listing_calendar_days"]:
                event(date, code, "buy_skipped", reason="new_or_unknown_listing")
                continue
            if date < pd.Timestamp(rule["start"]) or date >= delistings.get(code, pd.Timestamp.max) or code in st_today or code in st_set(signal_date):
                event(date, code, "buy_skipped", reason="ineligible")
                continue
            ok, reason = allowed(i, date, code, "buy")
            if not ok:
                event(date, code, "buy_skipped", reason=reason)
                continue
            price, adj = quote(i, code, "open"), quote(i, code, "adj_factor")
            volume_day = i - 1 if risk_allowed is not None else day_idx[signal_date]
            prior_volume = quote(volume_day, code, "vol_share")
            if not np.isfinite(adj) or adj <= 0 or not np.isfinite(prior_volume):
                event(date, code, "buy_skipped", reason="missing_adjustment_or_volume")
                continue
            qty = (planned_buys.get(code, 0) if retail_forecasts is not None else
                affordable_shares(min(budget, cash), price, cfg["buy_cost"], cfg["min_fee"],
                                  rule, prior_volume * cfg["max_volume_participation"], extra_rate=slippage_rate))
            if not qty:
                event(date, code, "buy_skipped", reason="lot_or_capacity")
                continue
            gross = qty * price
            fee = max(cfg["min_fee"], gross * cfg["buy_cost"])
            fee += gross * slippage_rate
            cash -= gross + fee
            release = dates[i + 1] if i + 1 < len(dates) else date + pd.Timedelta(days=1)
            if code in positions:
                positions[code].shares += qty
                positions[code].locks.append((qty, release))
            else:
                positions[code] = Position(qty, price, adj, locks=[(qty, release)])
            trades.append(dict(date=date, signal_date=signal_date, ts_code=code, side="buy",
                               shares=qty, price=price, gross=gross, fee=fee, cash_after=cash))
            if slippage_rate:
                trades[-1]['slippage_cost'] = gross * slippage_rate
            slots -= 1

        stock_value = 0.0
        for code, pos in positions.items():
            close = quote(i, code, "close")
            if np.isfinite(close) and close > 0:
                pos.last_price = close
                pos.stale_days = 0
            else:
                pos.stale_days += 1
                if pos.stale_days == cfg["stale_warning_days"]:
                    event(date, code, "stale_valuation", days=pos.stale_days)
            stock_value += pos.shares * pos.last_price
            holdings.append(dict(date=date, ts_code=code, shares=pos.shares,
                                 available=pos.available(date), price=pos.last_price,
                                 value=pos.shares * pos.last_price, stale_days=pos.stale_days))
        claims = sum(item["amount"] for item in receivables)
        equity = cash + stock_value + claims
        if cash < -1e-6 or not np.isfinite(equity) or equity < 0:
            raise AccountingError(f"{date.date()}: 现金/净值约束失效")
        daily.append(dict(date=date, cash=cash, stock_value=stock_value, receivables=claims,
                          equity=equity, n_hold=len(positions), pending_sell=len(pending_sell)))
        if risk_allowed is not None:
            state = ("exit_pending" if risk_exit_active else
                     "cash" if gate == 0 else "reentry_wait" if reentry_due else
                     "invested" if positions else "flat")
            daily[-1].update(risk_target=gate, risk_state=state, risk_signal_date=risk_date)
        for action in record_events.get(date, []):
            pos = positions.get(action["ts_code"])
            entitlements[action["id"]] = pos.shares if pos else 0
    eq = pd.DataFrame(daily).set_index("date")
    summary, eq["ret"] = daily_metrics(eq.equity, cfg["capital"], cfg["annual_days"])
    eq["nav"] = eq.equity / cfg["capital"]
    summary.update(capital=cfg["capital"], top_n=top_n, n_trades=len(trades),
                   total_fees=sum(t["fee"] for t in trades),
                   mean_invested=float((eq.stock_value / eq.equity).mean()),
                   missing_limit_events=sum(e["kind"] == "missing_limit" for e in events),
                   stale_warnings=sum(e["kind"] == "stale_valuation" for e in events),
                   delist_events=sum(e["kind"] == "delist_writeoff" for e in events),
                   adjustment_metadata_events=sum(e["kind"] == "adjustment_metadata_change" for e in events),
                   rights_declined_events=sum(e["kind"] == "rights_declined" for e in events),
                   n_rebalances=len(execution_days))
    summary["execution_data_sufficient"] = bool(summary["missing_limit_events"] == 0
        and summary["stale_warnings"] == 0
        and not any(e["kind"] == "unexplained_adjustment" for e in events))
    return BacktestResult(eq, pd.DataFrame(trades), pd.DataFrame(events), pd.DataFrame(holdings), summary)

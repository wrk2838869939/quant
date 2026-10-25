"""使用当日风格构造互斥股票对照；匹配不足必须显式记录。"""
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


def style_ranks(wide, eligible, window):
    """仅使用因果滚动与当日合格股票截面，无收益标签。"""
    raw = dict(market_cap=wide['circ_mv_yuan'],
               volatility=wide['hfq_close'].pct_change(fill_method=None).rolling(window).std(),
               earnings_yield=1 / wide['pe_ttm'].where(wide['pe_ttm'] > 0),
               turnover=wide['turnover_rate'].rolling(window).mean())
    return {k: value.where(eligible.reindex(index=value.index, columns=value.columns, fill_value=False)).rank(axis=1, pct=True)
            for k, value in raw.items()}


def make_control(signals, schedule, ranks, dimensions, seed, cfg):
    """每个信号日剔除原 top-N，再作不重复的一对一风格替换。

    优先最大化合格匹配数，其次最小化风格距离及新股票惩罚。
    静态扰动仅用于生成不同对照，不读取未来收益；不合格日期继续
    回放以避免择时筛选，但不能用于宣称完成风格控制。
    """
    codes = signals.columns
    lookup = {c: i for i, c in enumerate(codes)}
    rng = np.random.default_rng(seed)
    priority = rng.uniform(size=len(codes))
    noise = rng.uniform(size=(len(codes), len(codes)))
    control = pd.DataFrame(np.nan, index=signals.index, columns=codes)
    records, previous = [], set()
    for date, score in signals.iterrows():
        ranked = score.replace([np.inf, -np.inf], np.nan).dropna().sort_values(ascending=False, kind='stable')
        targets = list(ranked.head(int(schedule.loc[date])).index)
        frame = pd.DataFrame({k: r.loc[date] for k, r in ranks.items()}).reindex(codes)
        candidates = [c for c in ranked.index if c not in targets and frame.loc[c].notna().all()]
        chosen = []
        if targets and candidates:
            if dimensions:
                a = frame.loc[targets, dimensions].to_numpy()
                b = frame.loc[candidates, dimensions].to_numpy()
                delta = np.nan_to_num(a, nan=.5)[:, None, :] - b[None, :, :]
                rms = np.sqrt(np.mean(delta ** 2, axis=2))
                maximum = np.max(np.abs(delta), axis=2)
                valid = (rms <= cfg['rms_caliper']) & (maximum <= cfg['max_coordinate_gap']) & np.isfinite(a).all(axis=1)[:, None]
                perturb = noise[np.ix_([lookup[c] for c in targets], [lookup[c] for c in candidates])]
                new_names = np.array([c not in previous for c in candidates], dtype=float)
                cost = rms ** 2 + (~valid) * cfg['invalid_edge_penalty'] + perturb * cfg['persistent_noise'] + new_names * cfg['new_name_penalty']
                ii, jj = linear_sum_assignment(cost)
            else:
                ii = np.arange(min(len(targets), len(candidates)))
                jj = np.argsort([priority[lookup[c]] for c in candidates], kind='stable')[:len(ii)]
            for i, j in zip(ii, jj):
                target, candidate = targets[i], candidates[j]
                chosen.append(candidate)
                control.loc[date, candidate] = len(targets) - i
                records.append(dict(date=date, target=target, replacement=candidate,
                                    rms=float(rms[i, j]) if dimensions else np.nan,
                                    max_gap=float(maximum[i, j]) if dimensions else np.nan,
                                    passed=bool(valid[i, j]) if dimensions else False,
                                    candidates=len(candidates), missing=False))
        assigned = {r['target'] for r in records[-len(chosen):]} if chosen else set()
        for target in targets:
            if target not in assigned:
                records.append(dict(date=date, target=target, replacement=None, rms=np.nan,
                                    max_gap=np.nan, passed=False, candidates=len(candidates), missing=True))
        assert len(chosen) == len(set(chosen)) and not set(chosen).intersection(targets)
        previous = set(chosen)
    return control, pd.DataFrame(records)


def account_exposure(result, ranks):
    """按实际持仓市值归一后计算逐日风格，并保留有效数据覆盖率。"""
    days = result.daily.index
    if result.holdings.empty:
        return pd.DataFrame(np.nan, index=days, columns=list(ranks))
    values = result.holdings.pivot(index='date', columns='ts_code', values='value').reindex(days).fillna(0.)
    weights = values.div(values.sum(axis=1).replace(0, np.nan), axis=0)
    rows = {}
    for key, rank in ranks.items():
        r = rank.reindex(index=days, columns=values.columns)
        coverage = weights.where(r.notna()).sum(axis=1)
        rows[key] = (weights * r).sum(axis=1).div(coverage.replace(0, np.nan))
        rows[key + '_coverage'] = coverage
    rows['invested'] = result.daily.stock_value / result.daily.equity
    return pd.DataFrame(rows, index=days)

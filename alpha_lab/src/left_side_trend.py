"""先上行后回落的因果形态约束；与五日缩量减速触发分离。"""
import numpy as np
import pandas as pd


def trend_pullback(close, cfg, minimum_advance, minimum_pullback):
    """截至当日识别已出现的高点，在高点时评价历史趋势。

    不要求当前价格在均线上方；未来高点和未来低点不参与。
    高点用最近固定窗口内最高收盘，等高时取最近一次；
    上涨起点用高点前固定窗口最低收盘，等低时取最近一次。
    对趋势和波段参数的数值选择均是待检验假设。
    """
    c = close.to_numpy(dtype=float)
    avg = close.rolling(cfg['average_window']).mean()
    hist = close / close.shift(cfg['historical_window']) - 1
    slope = avg / avg.shift(cfg['average_slope_days']) - 1
    historical = (hist.ge(cfg['minimum_historical_return']) & slope.gt(0) & close.gt(avg)).to_numpy()
    keys = ['peak_i', 'anchor_i', 'peak_price', 'anchor_price', 'advance_days',
            'pullback_days', 'advance_return', 'pullback_return', 'historical_return_at_peak',
            'average_slope_at_peak', 'trend_at_peak', 'recent_rise_pullback', 'passes']
    out = {k: np.full(c.shape, np.nan) for k in keys}
    h, s = hist.to_numpy(), slope.to_numpy()
    # 高点与上涨起点只依赖截至高点的价格；预先计算可避免逐日重复寻找同一段上涨。
    for j in range(c.shape[1]):
        anchors = np.full(len(c), -1, dtype=int)
        for p in range(cfg['advance_lookback'], len(c)):
            window = c[p - cfg['advance_lookback']:p, j]
            if np.isfinite(window).all() and np.isfinite(c[p, j]):
                a = p - 1 - int(np.argmin(window[::-1]))
                anchors[p] = a
        for i in range(cfg['peak_lookback'], len(c)):
            # 排除当日以确保确实存在高点后的回落；缺口使整个波段不可判定。
            window = c[i - cfg['peak_lookback']:i, j]
            if not np.isfinite(window).all() or not np.isfinite(c[i, j]) or (window <= 0).any():
                continue
            p = i - 1 - int(np.argmax(window[::-1]))
            a = anchors[p]
            if a < 0 or c[a, j] <= 0:
                continue
            advance = c[p, j] / c[a, j] - 1
            pullback = c[i, j] / c[p, j] - 1
            ordered = (cfg['minimum_pullback_days'] <= i - p <= cfg['maximum_pullback_days']
                       and p - a >= cfg['minimum_advance_days']
                       and advance >= minimum_advance and pullback <= -minimum_pullback)
            passes = bool(historical[p, j] and ordered)
            if cfg['require_current_above_average']:
                passes &= bool(c[i, j] > avg.iloc[i, j])
            values = [p, a, c[p, j], c[a, j], p - a, i - p, advance, pullback,
                      h[p, j], s[p, j], float(historical[p, j]), float(ordered), float(passes)]
            for key, value in zip(keys, values):
                out[key][i, j] = value
    return {k: pd.DataFrame(v, index=close.index, columns=close.columns) for k, v in out.items()}


def project_sector_features(features, membership):
    """将当日板块形态映射到股票，未知及冲突行业保留缺失。"""
    result = {}
    sid = membership.to_numpy(float)
    for key, frame in features.items():
        values = frame.reindex(membership.index).to_numpy(float)
        out = np.full(sid.shape, np.nan)
        valid = np.isfinite(sid) & (sid >= 0) & (sid < values.shape[1])
        day, stock = np.where(valid)
        out[day, stock] = values[day, sid[day, stock].astype(int)]
        result[key] = pd.DataFrame(out, index=membership.index, columns=membership.columns)
    return result


def filter_scores(base_scores, stock_features, sector_features, mode='both'):
    """板块与个股分别约束，不允许用某一层的上涨替代另一层。"""
    result = base_scores.copy()
    if mode in ['both', 'stock']:
        result = result.where(stock_features['passes'].eq(1))
    if mode in ['both', 'sector']:
        result = result.where(sector_features['passes'].eq(1))
    return result

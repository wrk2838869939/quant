"""无需预测训练的规则信号与非单调联合搜索；只消费截至信号日的数据。"""
from __future__ import annotations

import hashlib
import itertools
import json
from functools import lru_cache

import numpy as np
import pandas as pd


def config_id(params):
    return hashlib.sha256(json.dumps(params, sort_keys=True, separators=(',', ':')).encode()).hexdigest()[:20]


def decision_dates(dates, frequency, calendar):
    """使用独立完整交易日历，不把截断窗口末日伪装为周末/月末。"""
    dates, calendar = pd.DatetimeIndex(dates), pd.DatetimeIndex(calendar)
    if frequency == 'daily':
        return dates
    period = {'weekly': 'W-FRI', 'monthly': 'M'}[frequency]
    ends = pd.Series(calendar, index=calendar).groupby(calendar.to_period(period)).last()
    return pd.DatetimeIndex(sorted(set(dates.intersection(ends)) | {dates[0]}))


class RuleSignals:
    """冻结宽表派生因子；滚动窗口按市场交易日，缺失不前向补价格。"""
    def __init__(self, wide, eligible, calendar):
        self.w = wide
        self.eligible = eligible
        self.calendar = calendar
        self.close = wide['hfq_close'].where(wide['hfq_close'] > 0)
        self.ret = self.close.pct_change(fill_method=None)
        self.turn = wide['turnover_rate'].where(wide['turnover_rate'] >= 0)

    @lru_cache(maxsize=64)
    def mean(self, field, window):
        return self.w[field].rolling(int(window), min_periods=int(window)).mean()

    @lru_cache(maxsize=48)
    def momentum(self, window):
        return self.close / self.close.shift(int(window)) - 1

    @lru_cache(maxsize=48)
    def volatility(self, window):
        return self.ret.rolling(int(window), min_periods=int(window)).std(ddof=1)

    def rank(self, frame):
        # 截面标准化只用当日有资格的股票；热身段无资格数据，排名只在正式信号段使用。
        allowed = self.eligible.reindex(index=frame.index, columns=frame.columns, fill_value=False)
        return frame.where(allowed).rank(axis=1, pct=True)

    def score(self, p):
        family, variant = p['family'], p.get('variant', '')
        w, c, t = self.w, self.close, self.turn
        n = int(p.get('window', 20))
        if family == 'S01':
            score = -t.rolling(n, min_periods=n).mean()
        elif family == 'S02':
            score = (-self.volatility(n) if variant == 'low_vol' else
                     -((w['high'] - w['low']) / w['pre_close'].where(w['pre_close'] > 0)).rolling(n, min_periods=n).mean())
        elif family == 'S03':
            score = (self.ret.abs() / w['amount_yuan'].where(w['amount_yuan'] > 0)).rolling(n, min_periods=n).mean()
        elif family == 'S04':
            field = 'pe_ttm' if variant == 'positive_ep' else 'pb'
            score = 1 / w[field].where(w[field] > 0)
            q = p['exclude_smallest_mv_quantile']
            mv = w['circ_mv_yuan'].where(w['circ_mv_yuan'] > 0)
            allowed = self.eligible.reindex(index=mv.index, columns=mv.columns, fill_value=False)
            cutoff = mv.where(allowed).quantile(q, axis=1)
            score = score.where(mv.ge(cutoff, axis=0))
        elif family == 'S05':
            score = -self.momentum(n)
        elif family == 'S06':
            score = c.shift(int(p['skip'])) / c.shift(int(p['lookback'])) - 1
        elif family == 'S07':
            short, long, ma = (int(p[k]) for k in ['short_window', 'long_window', 'ma_window'])
            rshort, rlong = self.momentum(short), self.momentum(long)
            gate = (rshort > 0) & (rlong > 0) & (c > c.rolling(ma, min_periods=ma).mean())
            score = (self.rank(rshort) + self.rank(rlong)) / 2
            if variant == 'continuous_trend':
                signs = np.sign(self.ret)
                continuity = signs.rolling(long, min_periods=long).mean()
                pos = np.log(c / c.shift()).clip(lower=0)
                jump = pos.rolling(long, min_periods=long).max() / pos.rolling(long, min_periods=long).sum().replace(0, np.nan)
                extension = c / c.rolling(short, min_periods=short).mean() - 1
                score = (score + self.rank(continuity) - self.rank(jump) - self.rank(extension)) / 4
            score = score.where(gate)
        elif family == 'S10':
            buy = w['buy_lg_amount_yuan'] + w['buy_elg_amount_yuan']
            sell = w['sell_lg_amount_yuan'] + w['sell_elg_amount_yuan']
            flow = (buy - sell).rolling(n, min_periods=n).sum()
            if variant == 'net_flow_amount':
                score = flow
            else:
                denom = {'net_flow_over_amount': w['amount_yuan'].rolling(n, min_periods=n).sum(),
                         'net_flow_over_float_mv': w['circ_mv_yuan'],
                         'net_flow_over_bucket_gross': (buy + sell).rolling(n, min_periods=n).sum()}[variant]
                score = flow / denom.where(denom > 0)
        elif family == 'S19':
            nstate = int(p['state_window'])
            vol = self.volatility(nstate)
            turn = t.rolling(nstate, min_periods=nstate).mean()
            allowed = self.eligible.reindex(index=c.index, columns=c.columns, fill_value=False)
            lowvol = vol.le(vol.where(allowed).quantile(p['vol_quantile'], axis=1), axis=0)
            highturn = turn.gt(turn.where(allowed).quantile(p['turnover_quantile'], axis=1), axis=0)
            ret = self.momentum(p['return_window'])
            score = ret.where(lowvol & highturn, -ret).where(vol.notna() & turn.notna())
        elif family == 'S20':
            # 先按当日已知滚动分位标记状态，再计算过去窗口内的条件振幅均值。
            amp = (w['hfq_high'] / w['hfq_low'] - 1).where(w['hfq_high'] > w['hfq_low'])
            valid = c.notna() & amp.notna()
            hi = c.ge(c.rolling(n, min_periods=n).quantile(p['high_quantile'])) & valid
            lo = c.le(c.rolling(n, min_periods=n).quantile(p['low_quantile'])) & valid
            hi_count = hi.astype(float).rolling(n, min_periods=n).sum()
            lo_count = lo.astype(float).rolling(n, min_periods=n).sum()
            h = amp.where(hi, 0.).rolling(n, min_periods=n).sum() / hi_count.replace(0, np.nan)
            l = amp.where(lo, 0.).rolling(n, min_periods=n).sum() / lo_count.replace(0, np.nan)
            history_ok = c.notna().astype(int).rolling(2 * n, min_periods=2 * n).sum().eq(2 * n)
            score = (l - h).where(history_ok)
        elif family == 'S21':
            intraday = w['close'] / w['open'].where(w['open'] > 0) - 1
            overnight = w['open'] / w['pre_close'].where(w['pre_close'] > 0) - 1
            score = -(intraday.rolling(n, min_periods=n).corr(t)
                      - overnight.rolling(n, min_periods=n).corr(t.shift()))
        elif family == 'S22':
            short, long = int(p['short_window']), int(p['long_window'])
            denom = t.rolling(long, min_periods=long).mean()
            score = t.rolling(short, min_periods=short).mean() / denom.where(denom > 0) - 1
            if variant == 'turnover_change_reversal':
                score = -score
        elif family == 'S23':
            score = c / c.rolling(n, min_periods=n).max()
        elif family == 'momentum':
            score = self.momentum(n).where(self.momentum(n) > 0)
        elif family == 'moving_average':
            fast = c.rolling(int(p['fast']), min_periods=int(p['fast'])).mean()
            slow = c.rolling(int(p['slow']), min_periods=int(p['slow'])).mean()
            score = (fast / slow - 1).where((fast > slow) & (c > fast))
        elif family == 'macd':
            dif = c.ewm(span=int(p['fast']), adjust=False).mean() - c.ewm(span=int(p['slow']), adjust=False).mean()
            dea = dif.ewm(span=int(p['signal']), adjust=False).mean()
            score = ((dif - dea) / c).where((dif > dea) & (dif > 0))
        else:
            raise ValueError(f'未实现的规则家族 {family}')
        return score.reindex_like(self.eligible).where(self.eligible).replace([np.inf, -np.inf], np.nan)

    def signals(self, p):
        score = self.score(p)
        dates = decision_dates(score.index, p['frequency'], self.calendar)
        return score.loc[dates]


def parameter_axes(plan, family):
    if family in plan['family_spaces']:
        space = plan['family_spaces'][family]
        axes = {k: list(v) for k, v in space.items() if k not in plan['numeric_domains'][family]}
        for k, d in plan['numeric_domains'][family].items():
            values = np.arange(round((d['max'] - d['min']) / d['step']) + 1) * d['step'] + d['min']
            axes[k] = [int(x) for x in values] if d['type'] == 'integer' else [round(float(x), 6) for x in values]
    else:
        axes = {k: list(v) for k, v in plan['classic_tunable_variants'][family].items() if k != 'constraints'}
    axes.update(plan['joint_portfolio_dimensions'])
    return axes


def legal(p):
    if 'skip' in p and p['skip'] >= p['lookback']:
        return False
    if 'short_window' in p and p['short_window'] >= p['long_window']:
        return False
    if 'fast' in p and p['fast'] >= p['slow']:
        return False
    return True


def sample_configs(plan, family, count, seed, seen=()):
    """结构类别均衡，数值全域随机，完整配置去重；不使用表现推断方向。"""
    if count <= 0:
        return []
    rng = np.random.default_rng(seed)
    axes = parameter_axes(plan, family)
    seen = set(seen)
    result = []
    categories = list(itertools.product(axes.get('variant', [None]), axes['frequency'], axes['top_n']))
    rng.shuffle(categories)
    for attempt in range(max(10000, count * 200)):
        p = {'family': family, **{k: v[int(rng.integers(len(v)))] for k, v in axes.items()}}
        variant, p['frequency'], p['top_n'] = categories[attempt % len(categories)]
        if variant is not None:
            p['variant'] = variant
        key = config_id(p)
        if legal(p) and key not in seen:
            result.append(p)
            seen.add(key)
            if len(result) == count:
                break
    return result


def all_configs(plan, family, maximum):
    axes = parameter_axes(plan, family)
    if np.prod([len(v) for v in axes.values()], dtype=float) > maximum * 3:
        return None
    result = []
    for values in itertools.product(*axes.values()):
        p = {'family': family, **dict(zip(axes, values))}
        if legal(p):
            result.append(p)
            if len(result) > maximum:
                return None
    return result


def local_configs(plan, family, centers, count, seed, seen):
    """在多个候选周围做单轴与双轴扰动，不使用梯度或单调假设。"""
    if count <= 0:
        return []
    rng = np.random.default_rng(seed)
    axes = parameter_axes(plan, family)
    numeric = [k for k, v in axes.items() if k not in ['variant', 'frequency']]
    result, seen = [], set(seen)
    if not centers:
        return result
    for attempt in range(max(10000, count * 200)):
        p = dict(centers[attempt % len(centers)])
        chosen = rng.choice(numeric, size=min(len(numeric), 1 + attempt % 2), replace=False)
        for k in chosen:
            axis = axes[k]
            at = axis.index(p[k])
            delta = int(rng.choice([-1, 1])) * (1 if attempt % 3 else max(1, round(len(axis) * .1)))
            p[k] = axis[min(len(axis) - 1, max(0, at + delta))]
        key = config_id(p)
        if legal(p) and key not in seen:
            seen.add(key)
            result.append(p)
            if len(result) == count:
                break
    return result


def inner_metrics(result, annual_days, capital):
    from portfolio import daily_metrics
    previous, yearly = capital, []
    for year, frame in result.daily.groupby(result.daily.index.year):
        metrics, _ = daily_metrics(frame.equity, previous, annual_days)
        yearly.append({'year': int(year), **metrics})
        previous = frame.equity.iloc[-1]
    buys = int(result.trades.side.eq('buy').sum()) if not result.trades.empty else 0
    turnover = float(result.trades.gross.sum() / result.daily.equity.mean() / max(len(yearly), 1)) if not result.trades.empty else 0.
    return dict(**result.summary, median_ann=float(np.median([x['ann'] for x in yearly])),
                worst_block_mdd=min(x['mdd'] for x in yearly), buys=buys, turnover=turnover,
                yearly=yearly, feasible=bool(buys and result.summary['mdd'] >= -.30
                                           and min(x['mdd'] for x in yearly) >= -.30))


def select_trial(rows, tie_band=.005, complexity=None):
    """只选择内层可行配置；邻域证据缺失不插值，外层结果不在入参中。"""
    good = [r for r in rows if r.get('status') == 'ok' and r['metrics']['feasible']]
    if not good:
        return None
    complexity = complexity or {}
    def complexity_key(row):
        return complexity.get(row['params'].get('variant', row['params']['family']), 1)
    best = max(r['metrics']['median_ann'] for r in good)
    near = [r for r in good if r['metrics']['median_ann'] >= best - tie_band]
    # 邻域须同分支/频率，数值相对距离不超过10%；只用实际评估点。
    for row in near:
        p = row['params']
        scores = []
        for other in rows:
            if other.get('status') != 'ok' or other['id'] == row['id']:
                continue
            q = other['params']
            if p.get('variant') != q.get('variant') or p['frequency'] != q['frequency']:
                continue
            keys = [k for k, v in p.items() if isinstance(v, (int, float))]
            if all(abs(p[k] - q[k]) <= max(1 if isinstance(p[k], int) else .01, abs(p[k]) * .1) for k in keys):
                scores.append(other['metrics']['median_ann'])
        row['neighbor_count'] = len(scores)
        row['neighbor_q25'] = float(np.quantile(scores, .25)) if len(scores) >= 3 else None
    # 证据不足时不强行给平台奖励；仅证据均充分才比较邻域，否则按换手打破近似并列。
    if all(r['neighbor_q25'] is not None for r in near):
        near.sort(key=lambda r: (-r['neighbor_q25'], r['metrics']['turnover'], complexity_key(r), r['id']))
    else:
        near.sort(key=lambda r: (r['metrics']['turnover'], complexity_key(r), r['id']))
    return near[0]

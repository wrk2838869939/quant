"""左侧板块状态、五日量价形态与收盘决策；不修改固定执行引擎。"""
from functools import lru_cache
import numpy as np
import pandas as pd
from initialized_policy import PolicyAction


def membership_matrix(members, catalog, dates, codes, cfg):
    """按已记录进出区间构造归属；未知为 -1，冲突为 -2，不回填。"""
    sectors = sorted(catalog.index_code.unique())
    lookup = {s: i for i, s in enumerate(sectors)}
    stock_idx = {s: i for i, s in enumerate(codes)}
    result = np.full((len(dates), len(codes)), -1, dtype=np.int16)
    change = pd.Timestamp(cfg['version_change'])
    old = set(catalog.loc[catalog.src.eq(cfg['older_version']), 'index_code'])
    new = set(catalog.loc[catalog.src.eq(cfg['newer_version']), 'index_code'])
    for r in members.itertuples():
        if r.con_code not in stock_idx or r.index_code not in lookup or pd.isna(r.in_date):
            continue
        start = pd.to_datetime(str(r.in_date), format='%Y%m%d', errors='coerce')
        end = pd.to_datetime(str(r.out_date), format='%Y%m%d', errors='coerce')
        if pd.isna(start):
            continue
        active = (dates >= start) & ((dates < end) if pd.notna(end) else True)
        active &= ((dates < change) & (r.index_code in old)) | ((dates >= change) & (r.index_code in new))
        j, value = stock_idx[r.con_code], lookup[r.index_code]
        existing = result[:, j]
        collision = active & (existing != -1) & (existing != value)
        existing[active & (existing == -1)] = value
        existing[collision] = -2
    return result, sectors


def capped_weights(values, cap):
    """将昨日总市值权重迭代限制在单股上限，避免大股票独占代理指数。"""
    weights = np.zeros(len(values), dtype=float)
    active = np.ones(len(values), dtype=bool)
    remaining = 1.
    if len(values) * cap < 1 - 1e-10:
        raise ValueError('成员数量不足以满足权重上限')
    while active.any():
        trial = values[active] / values[active].sum() * remaining
        over = trial > cap + 1e-12
        idx = np.flatnonzero(active)
        if not over.any():
            weights[idx] = trial
            break
        weights[idx[over]] = cap
        active[idx[over]] = False
        remaining = 1 - weights.sum()
    return weights


def sector_panel(close, cap, mapping, sectors, cfg):
    """以昨日历史成员及市值权重自算指数；缺口中断序列，重启为新段。"""
    prices, caps = close.to_numpy(float), cap.to_numpy(float)
    returns = np.full_like(prices, np.nan)
    returns[1:] = prices[1:] / prices[:-1] - 1
    output = []
    levels = np.full(len(sectors), cfg['base_value'], dtype=float)
    continuous = np.zeros(len(sectors), dtype=bool)
    for i, date in enumerate(close.index):
        previous = mapping[i - 1] if i else np.full(len(close.columns), -1)
        for s, code in enumerate(sectors):
            mask = previous == s
            n = int(mask.sum())
            valid = mask & np.isfinite(returns[i])
            if i:
                valid &= np.isfinite(caps[i - 1]) & (caps[i - 1] > 0)
            else:
                valid[:] = False
            used = int(valid.sum())
            coverage = used / n if n else 0.
            ready = used >= cfg['minimum_members'] and coverage >= cfg['minimum_quote_coverage']
            ret = np.nan
            if ready:
                weights = capped_weights(caps[i - 1, valid], cfg['maximum_member_weight'])
                ret = float(np.dot(weights, returns[i, valid]))
                if continuous[s]:
                    levels[s] *= 1 + ret
                else:
                    levels[s] = cfg['base_value']
            continuous[s] = ready
            current = mapping[i] == s
            breadth_ok = current & np.isfinite(returns[i])
            output.append(dict(date=date, sector=code, sector_id=s,
                               close=levels[s] if ready else np.nan, ret=ret,
                               known_members=n, quoted_members=used, coverage=coverage,
                               breadth=float((returns[i, breadth_ok] > 0).mean()) if breadth_ok.any() else np.nan,
                               observed_total_cap=float(np.nansum(caps[i, current]))))
    return pd.DataFrame(output)


def sector_states(frame, cfg):
    """逐日建立平台—上涨—急跌—减速状态，不在未来反弹后倒选平台。"""
    out = []
    for sector, part in frame.groupby('sector', sort=False):
        part = part.sort_values('date').reset_index(drop=True)
        p = part.close
        r = p.pct_change(fill_method=None)
        high = p.shift().rolling(cfg['platform_window']).max()
        low = p.shift().rolling(cfg['platform_window']).min()
        support = p.shift().rolling(cfg['support_window']).min()
        fast = p / p.shift(cfg['fast_fall_window']) - 1
        losses = (-r).clip(lower=0)
        recent = losses.rolling(cfg['recent_slow_days']).mean()
        before = losses.shift(cfg['recent_slow_days']).rolling(cfg['preceding_fall_days']).mean()
        state = None
        for i, row in part.iterrows():
            price = row.close
            if not np.isfinite(price):
                state = None
            if state is not None:
                age = i - state['anchor_i']
                expired = age > cfg['maximum_episode_days']
                expired |= not state['rallied'] and age > cfg['maximum_unconfirmed_days']
                if expired:
                    state = None
            if state is None and np.isfinite(price) and np.isfinite(high.iloc[i]) and low.iloc[i] > 0:
                if high.iloc[i] / low.iloc[i] - 1 <= cfg['platform_max_range'] and price > high.iloc[i] * (1 + cfg['breakout_margin']):
                    state = dict(anchor_i=i - 1, anchor_date=part.date.iloc[i - 1],
                                 platform_low=low.iloc[i], platform_high=high.iloc[i],
                                 value_high=price, peak=price, peak_i=i, peak_date=row.date,
                                 rallied=False, fast_fall=False, broken=False)
            rec = row.to_dict()
            rec.update(rallied=False, drawn_down=False, fast_fall=False, broken=False,
                       slowed=False, structural_gate=False, value_distance=np.nan,
                       above_lower_bound=False, anchor_i=-1, peak_i=-1,
                       anchor_date=pd.NaT, peak_date=pd.NaT)
            if state is not None:
                if i - state['anchor_i'] <= cfg['initial_rise_days']:
                    state['value_high'] = max(state['value_high'], price)
                if price > state['peak']:
                    state.update(peak=price, peak_i=i, peak_date=row.date, fast_fall=False, broken=False)
                state['rallied'] |= state['peak'] / state['platform_high'] - 1 >= cfg['minimum_rise']
                if state['rallied'] and i > state['peak_i']:
                    state['fast_fall'] |= fast.iloc[i] <= -cfg['fast_fall_threshold']
                    state['broken'] |= price < support.iloc[i]
                dd = price / state['peak'] - 1
                slowed = bool(before.iloc[i] > 0 and recent.iloc[i] <= before.iloc[i] * cfg['slowdown_ratio'])
                structural = state['rallied'] and dd <= -cfg['minimum_drawdown'] and state['fast_fall'] and state['broken'] and slowed
                rec.update(state)
                rec.update(drawn_down=dd <= -cfg['minimum_drawdown'], slowed=slowed,
                           structural_gate=bool(structural), drawdown=dd,
                           value_distance=price / state['value_high'] - 1,
                           above_lower_bound=bool(price >= state['platform_low'] * (1 - cfg['lower_value_tolerance'])))
            out.append(rec)
    return pd.DataFrame(out).sort_values(['date', 'sector']).reset_index(drop=True)


def stock_shape(wide, cfg, params):
    """先放量下跌，后缩量且负收益幅度收窄；不要求反转阳线。"""
    c, turnover = wide['hfq_close'], wide['turnover_rate']
    r = c.pct_change(fill_method=None)
    recent = cfg['recent_days']
    early = params['shape_window'] - recent
    early_turn = turnover.shift(recent).rolling(early).mean()
    late_turn = turnover.rolling(recent).mean()
    reference = turnover.shift(params['shape_window']).rolling(cfg['turnover_baseline_window']).mean()
    early_ret = c.shift(recent) / c.shift(params['shape_window']) - 1
    late_ret = c / c.shift(recent) - 1
    early_loss = (-r).clip(lower=0).shift(recent).rolling(early).mean()
    late_loss = (-r).clip(lower=0).rolling(recent).mean()
    expansion = early_turn / reference.where(reference > 0)
    contraction = late_turn / early_turn.where(early_turn > 0)
    slowdown = late_loss / early_loss.where(early_loss > 0)
    shape = ((early_ret <= -cfg['minimum_early_fall']) &
             (expansion >= params['turnover_expansion']) &
             (contraction <= params['turnover_contraction']) &
             (slowdown <= cfg['slowdown_ratio']) &
             (late_ret <= cfg['maximum_recent_gain']) &
             (r >= -cfg['maximum_current_daily_fall']))
    return dict(shape=shape, expansion=expansion, contraction=contraction,
                slowdown=slowdown, early_return=early_ret, late_return=late_ret)


def stock_sector_features(states, full_close, mapping, sectors, stocks, cfg):
    """历史龙头按同一已发生上涨段排名，随后下跌不会改写其历史身份。"""
    dates, all_codes = full_close.index, full_close.columns
    positions = all_codes.get_indexer(stocks)
    if (positions < 0).any():
        raise ValueError('工程股票未包含在全市场行情中')
    prices = full_close.to_numpy(float)
    sid = mapping[:, positions]
    rows = states.set_index(['date', 'sector_id'])
    fields = ['structural_gate', 'value_distance', 'above_lower_bound', 'breadth', 'coverage', 'anchor_i', 'peak_i']
    result = {f: np.full((len(dates), len(stocks)), np.nan) for f in fields}
    leader = np.full_like(result['coverage'], np.nan)

    @lru_cache(maxsize=None)
    def leaders(sector_id, a, p):
        valid = (mapping[a] == sector_id) & (mapping[p] == sector_id)
        valid &= np.isfinite(prices[a]) & np.isfinite(prices[p]) & (prices[a] > 0)
        ranks = np.full(len(all_codes), np.nan)
        if valid.sum() >= cfg['minimum_leader_peers']:
            ranks[valid] = pd.Series(prices[p, valid] / prices[a, valid] - 1).rank(pct=True).to_numpy()
        return ranks[positions]

    for i, date in enumerate(dates):
        for s in np.unique(sid[i]):
            if s < 0 or (date, int(s)) not in rows.index:
                continue
            row = rows.loc[(date, int(s))]
            selected = sid[i] == s
            for f in fields:
                result[f][i, selected] = float(row[f])
            a, p = int(row.anchor_i), int(row.peak_i)
            if a >= 0 and p > a:
                leader[i, selected] = leaders(int(s), a, p)[selected]
    result['leader_rank'] = leader
    result['sector_id'] = sid.astype(float)
    return {k: pd.DataFrame(v, index=dates, columns=stocks) for k, v in result.items()}


def entry_scores(wide, eligible, sector_features, cfg, params, mode='full'):
    """辅助广度和市值只记录，不擅自指定方向；龙头历史默认用于排序。"""
    shape = stock_shape(wide, cfg['stock'], params)
    f = sector_features
    sector_ok = (f['structural_gate'].eq(1) & f['above_lower_bound'].eq(1)
                 & f['value_distance'].le(params['value_tolerance']))
    gate = shape['shape'] if mode == 'stock_only' else sector_ok & shape['shape']
    allowed = eligible.reindex_like(gate).fillna(False).astype(bool)
    leader = f['leader_rank'] if mode not in ['no_leader', 'stock_only'] else f['leader_rank'] * 0 + .5
    # 缺历史身份保持中性排序，覆盖损失单独记录，不能误认定为历史龙头。
    score = cfg['stock']['leader_weight'] * leader.fillna(.5)
    score -= cfg['stock']['shape_weight'] * shape['contraction']
    return score.where(gate & allowed), shape, sector_ok


class LeftSidePolicy:
    """用真实成交初始化持仓状态；退出请求持续至成交，支持日线移动止盈。"""
    def __init__(self, scores, wide, cfg):
        self.scores, self.wide, self.cfg = scores, wide, cfg
        self.codes = scores.columns
        self.date_idx = {d: i for i, d in enumerate(scores.index)}
        self.support = wide['hfq_low'].rolling(cfg['exit']['support_window']).min()
        self.previous_support = self.support.shift()
        self.state, self.last_sold = {}, {}
        self.records, self.actions = [], []

    def __call__(self, snapshot):
        date, positions = snapshot['date'], snapshot['positions']
        i = self.date_idx[date]
        for code in list(self.state):
            if code not in positions:
                del self.state[code]
                self.last_sold[code] = i
        buys = {t['ts_code']: t for t in snapshot['trades'] if t['side'] == 'buy'}
        for code, trade in buys.items():
            if code in self.state:
                continue
            signal = pd.Timestamp(trade['signal_date'])
            ratio = self.wide['hfq_close'].at[date, code] / self.wide['close'].at[date, code]
            stop = self.support.at[signal, code] * (1 - self.cfg['exit']['support_buffer'])
            self.state[code] = dict(entry=trade['price'] * ratio, stop=stop, trailing=False, exit=False)
        exits, keep = set(), set()
        for code in positions:
            state = self.state[code]
            price = self.wide['hfq_close'].at[date, code]
            if np.isfinite(price):
                state['trailing'] |= price / state['entry'] - 1 >= self.cfg['exit']['trailing_activation']
                support = self.previous_support.at[date, code]
                if state['trailing'] and np.isfinite(support):
                    state['stop'] = max(state['stop'], support * (1 - self.cfg['exit']['support_buffer']))
                state['exit'] |= bool(price < state['stop'])
            state['exit'] |= code in snapshot['pending_sell']
            (exits if state['exit'] else keep).add(code)
            self.records.append(dict(date=date, ts_code=code, action='exit' if state['exit'] else 'hold',
                                     reason='trailing_support' if state['trailing'] else 'initial_support', **state))
        scores = self.scores.loc[date].copy()
        excluded = set(positions) | {c for c, day in self.last_sold.items() if i - day <= self.cfg['exit']['cooldown_days']}
        scores.loc[list(excluded)] = np.nan
        # 有限支撑是入场必需数据；不因其缺失建立没有退出定义的仓位。
        scores = scores.where(self.support.loc[date].notna())
        action = PolicyAction(scores=scores, top_n=self.cfg['stock']['top_n'], keep=keep, exit=exits)
        self.actions.append((date, action))
        return action

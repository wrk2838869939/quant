"""独立板块数据层：历史归属、前日权重指数、当日广度与市值分解。"""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

FIELDS = ['hfq_open', 'hfq_high', 'hfq_low', 'hfq_close', 'total_mv_yuan',
          'circ_mv_yuan', 'vol_share', 'amount_yuan', 'float_share']


def digest(path):
    """流式计算文件哈希。"""
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def capped_weights(values, cap):
    """对正市值迭代限权；无法满足上限时拒绝计算。"""
    values = np.asarray(values, float)
    if len(values) * cap < 1 - 1e-10 or not (np.isfinite(values) & (values > 0)).all():
        raise ValueError('市值或权重上限无效')
    w = np.zeros(len(values))
    active = np.ones(len(values), bool)
    while active.any():
        idx = np.flatnonzero(active)
        trial = values[idx] / values[idx].sum() * (1 - w.sum())
        over = trial > cap + 1e-12
        if not over.any():
            w[idx] = trial
            break
        w[idx[over]] = cap
        active[idx[over]] = False
    return w


def build_membership(members, catalog, dates, codes, basic, cfg):
    """区间采用 [in_date,out_date)，未知 -1、冲突 -2；不回填当前成员。"""
    sectors = sorted(catalog.index_code.unique())
    lookup = {s: i for i, s in enumerate(sectors)}
    stocks = {s: i for i, s in enumerate(codes)}
    mapping = np.full((len(dates), len(codes)), -1, np.int16)
    active = np.zeros((len(dates), len(sectors)), bool)
    change = pd.Timestamp(cfg['version_change'])
    for version, mask in [(cfg['older_version'], dates < change),
                          (cfg['newer_version'], dates >= change)]:
        for s in catalog.loc[catalog.src.eq(version), 'index_code']:
            active[mask, lookup[s]] = True
    for r in members.itertuples():
        if r.con_code not in stocks or r.index_code not in lookup:
            continue
        start = pd.to_datetime(r.in_date, format='%Y%m%d', errors='coerce')
        end = pd.to_datetime(r.out_date, format='%Y%m%d', errors='coerce')
        if pd.isna(start):
            continue
        s, j = lookup[r.index_code], stocks[r.con_code]
        mask = (dates >= start) & active[:, s]
        if pd.notna(end):
            mask &= dates < end
        col = mapping[:, j]
        conflict = mask & (col != -1) & (col != s)
        col[mask & (col == -1)] = s
        col[conflict] = -2
    # 只使用上市/退市日期限制存续，不以当前上市状态筛股。
    life = np.ones(mapping.shape, bool)
    for r in basic.itertuples():
        if r.ts_code not in stocks:
            continue
        start = pd.to_datetime(r.list_date, format='%Y%m%d', errors='coerce')
        end = pd.to_datetime(r.delist_date, format='%Y%m%d', errors='coerce')
        j = stocks[r.ts_code]
        if pd.notna(start):
            life[:, j] &= dates >= start
        if pd.notna(end):
            life[:, j] &= dates < end
    mapping[~life] = -1
    return mapping, sectors, active, life


def make_wide(raw, dates, codes):
    """按交易日对齐，缺报价不填充；显式拒绝重复记录。"""
    if raw.duplicated(['date', 'ts_code']).any():
        raise ValueError('重复个股日线')
    return {k: raw.pivot(index='date', columns='ts_code', values=k)
            .reindex(index=dates, columns=codes).to_numpy(float) for k in FIELDS}


def positive(x):
    return np.isfinite(x) & (x > 0)


def sum_known(x, valid=None):
    valid = np.isfinite(x) if valid is None else valid
    return float(x[valid].sum()) if valid.any() else np.nan


def aggregate(wide, dates, mapping, sectors, active, cfg, progress=False):
    """构建自算代理指数及观测统计；所有输入仅使用当前及过去交易日。

    指数每日按前日成员与前日市值重新限权。缺行情成员排除后归一化，
    因而是带覆盖限制的观测代理，不是停牌估值完备的可交易基金净值。
    """
    close = wide['hfq_close']
    cap = wide['total_mv_yuan']
    nsector = len(sectors)
    levels = np.full(nsector, cfg['base_value'], float)
    equal_levels = levels.copy()
    continuous = np.zeros(nsector, bool)
    segments = np.zeros(nsector, int)
    records = []
    for i, date in enumerate(dates):
        if progress and i % 250 == 0:
            print(f'AGGREGATE {i}/{len(dates)} {date.date()}', flush=True)
        current = mapping[i]
        previous = mapping[i - 1] if i else np.full(mapping.shape[1], -1)
        prev_close = close[i - 1] if i else np.full(mapping.shape[1], np.nan)
        prev_cap = cap[i - 1] if i else np.full(mapping.shape[1], np.nan)
        ret = np.full(mapping.shape[1], np.nan)
        valid_ret = positive(close[i]) & positive(prev_close)
        ret[valid_ret] = close[i, valid_ret] / prev_close[valid_ret] - 1
        version_break = i == 0 or ((dates[i - 1] < pd.Timestamp(cfg['version_change'])) !=
                                  (date < pd.Timestamp(cfg['version_change'])))
        for s, sector in enumerate(sectors):
            if not active[i, s]:
                continuous[s] = False
                continue
            cur = np.flatnonzero(current == s)
            prev = np.flatnonzero(previous == s)
            n = len(cur)
            r = dict(date=date, sector_code=sector,
                     classification_version=cfg['older_version'] if date < pd.Timestamp(cfg['version_change']) else cfg['newer_version'],
                     known_members=n, prior_known_members=len(prev))
            for field in ['total_mv_yuan', 'circ_mv_yuan', 'vol_share', 'amount_yuan']:
                x = wide[field][i, cur]
                valid = positive(x) if 'mv_' in field else np.isfinite(x) & (x >= 0)
                r[field] = sum_known(x, valid)
                r[field + '_members'] = int(valid.sum())
            float_shares = wide['float_share'][i, cur] * cfg['share_unit_multiplier']
            volume = wide['vol_share'][i, cur]
            turn_ok = positive(float_shares) & np.isfinite(volume) & (volume >= 0)
            r['turnover_members'] = int(turn_ok.sum())
            r['turnover_pct'] = float(volume[turn_ok].sum() / float_shares[turn_ok].sum() * 100) if turn_ok.any() else np.nan
            breadth = ret[cur]
            eps = cfg['flat_return_tolerance']
            up = int((breadth > eps).sum())
            down = int((breadth < -eps).sum())
            flat = int((np.abs(breadth) <= eps).sum())
            r.update(up_count=up, down_count=down, flat_count=flat,
                     breadth_members=up + down + flat, breadth_unknown=n - up - down - flat,
                     up_share=up / (up + down + flat) if up + down + flat else np.nan,
                     up_down_ratio=up / down if down else np.nan,
                     advancing_share=up / (up + down) if up + down else np.nan)
            common = np.flatnonzero((current == s) & (previous == s))
            matched = common[positive(cap[i, common]) & positive(prev_cap[common])]
            r['matched_cap_members'] = len(matched)
            r['matched_cap_change_yuan'] = float((cap[i, matched] - prev_cap[matched]).sum()) if len(matched) else np.nan
            r['matched_cap_return'] = float(cap[i, matched].sum() / prev_cap[matched].sum() - 1) if len(matched) else np.nan
            added = np.setdiff1d(cur, prev, assume_unique=True)
            removed = np.setdiff1d(prev, cur, assume_unique=True)
            r.update(added_members=len(added), removed_members=len(removed))
            # 按可观测市值恒等分解；缺失报价带来的变化单列，不伪装为成员变化。
            now_known = cur[positive(cap[i, cur])]
            old_known = prev[positive(prev_cap[prev])]
            entering = np.setdiff1d(now_known, old_known, assume_unique=True)
            leaving = np.setdiff1d(old_known, now_known, assume_unique=True)
            r['member_add_cap_yuan'] = float(cap[i, np.intersect1d(entering, added)].sum())
            r['member_remove_cap_yuan'] = float(prev_cap[np.intersect1d(leaving, removed)].sum())
            r['missing_cap_effect_yuan'] = float(cap[i, np.setdiff1d(entering, added)].sum() - prev_cap[np.setdiff1d(leaving, removed)].sum())
            r['observed_cap_change_yuan'] = (float(cap[i, now_known].sum() - prev_cap[old_known].sum())
                                             if i and len(now_known) and len(old_known) else np.nan)
            valid = valid_ret[prev] & positive(prev_cap[prev])
            for field in ['hfq_open', 'hfq_high', 'hfq_low']:
                valid &= positive(wide[field][i, prev])
            use = prev[valid]
            coverage = len(use) / len(prev) if len(prev) else 0.
            enough = len(use) >= max(cfg['minimum_members'], int(np.ceil(1 / cfg['maximum_member_weight'])))
            ready = enough and coverage >= cfg['minimum_quote_coverage'] and not version_break
            r.update(index_members=len(use), index_member_coverage=coverage,
                     index_valid=ready, index_status=('version_or_start' if version_break else
                     'ok' if ready else 'few_members' if not enough else 'low_quote_coverage'))
            r.update(index_close=np.nan, index_open=np.nan, index_high_bound=np.nan,
                     index_low_bound=np.nan, equal_index_close=np.nan, index_return=np.nan,
                     equal_return=np.nan, chain_return=np.nan, max_weight=np.nan, weight_hhi=np.nan)
            if ready:
                weights = capped_weights(prev_cap[use], cfg['maximum_member_weight'])
                factors = {k: float(np.dot(weights, wide['hfq_' + k][i, use] / prev_close[use]))
                           for k in ['open', 'high', 'low', 'close']}
                eq_factor = float(np.mean(close[i, use] / prev_close[use]))
                restart = not continuous[s]
                if restart:
                    segments[s] += 1
                    # 首日收盘固定基点，开盘及边界同步缩放，首日不提供跨段链收益。
                    prior_level = cfg['base_value'] / factors['close']
                    equal_levels[s] = cfg['base_value']
                else:
                    prior_level = levels[s]
                    equal_levels[s] *= eq_factor
                levels[s] = prior_level * factors['close']
                r.update(index_close=levels[s], index_open=prior_level * factors['open'],
                         index_high_bound=prior_level * factors['high'], index_low_bound=prior_level * factors['low'],
                         equal_index_close=equal_levels[s], index_return=factors['close'] - 1,
                         equal_return=eq_factor - 1, chain_return=np.nan if restart else factors['close'] - 1,
                         max_weight=float(weights.max()), weight_hhi=float(np.square(weights).sum()))
            continuous[s] = ready
            r['segment_id'] = segments[s] if ready else 0
            records.append(r)
    return pd.DataFrame(records)


class SectorDataset:
    """读取一个明确的数据版本；默认核验全部产物哈希。"""
    def __init__(self, root, verify=True):
        self.root = Path(root)
        self.manifest = json.loads((self.root / 'manifest.json').read_text())
        if self.manifest['status'] != 'complete':
            raise ValueError('数据集未完成')
        if verify:
            for name, expected in self.manifest['artifacts'].items():
                if digest(self.root / name) != expected:
                    raise ValueError(f'文件哈希不匹配: {name}')

    def history(self, sector_code, start=None, end=None):
        filters = [('sector_code', '=', sector_code)]
        if start:
            filters.append(('date', '>=', pd.Timestamp(start)))
        if end:
            filters.append(('date', '<=', pd.Timestamp(end)))
        return pd.read_parquet(self.root / 'daily.parquet', filters=filters).sort_values('date')

    def members_at(self, sector_code, date):
        frame = pd.read_parquet(self.root / 'membership_daily.parquet')
        date = pd.Timestamp(date)
        if date not in frame.index:
            raise ValueError('要求数据范围内的交易日')
        codes = json.loads((self.root / 'sector_codes.json').read_text())
        if sector_code not in codes:
            raise KeyError(sector_code)
        return frame.columns[frame.loc[date].eq(codes.index(sector_code))].tolist()

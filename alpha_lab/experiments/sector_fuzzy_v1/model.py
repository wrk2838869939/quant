"""仅依赖当时数据的特征、模糊规则和Takagi–Sugeno网络。"""
import copy
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.cluster import KMeans


def features_at_events(signals, panel, cfg):
    """只读取逐日快照及截至确认日的滚动行情，显式列白名单。"""
    pieces = []
    for _, g in panel[panel.index_valid].groupby(['sector_code', 'segment_id']):
        g = g.sort_values('date').copy()
        breadth = g.up_share.where(g.breadth_members / g.known_members >= cfg['minimum_breadth_coverage'])
        recent, previous = cfg['breadth_recent_days'], cfg['breadth_previous_days']
        g['breadth_change'] = breadth.rolling(recent).mean() - breadth.shift(recent).rolling(previous).mean()
        g['volatility'] = g.index_close.pct_change(fill_method=None).rolling(cfg['volatility_window']).std()
        pieces.append(g[['sector_code', 'date', 'breadth_change', 'volatility']])
    history = pd.concat(pieces, ignore_index=True)
    s = signals.merge(history, on=['sector_code', 'date'], how='left', validate='one_to_one')
    f = pd.DataFrame(dict(event_id=s.event_id, date=s.date, sector_code=s.sector_code,
        rally=s.rally_return, pullback=-s.drawdown, reference_distance=s.reference_distance,
        slowdown=s.recent_loss_mean / s.previous_loss_mean.where(s.previous_loss_mean > 0),
        log_turnover=np.log(s.turnover_vs_previous20.where(s.turnover_vs_previous20 > 0)),
        up_share=s.up_share.where(s.breadth_members / s.known_members >= cfg['minimum_breadth_coverage']),
        breadth_change=s.breadth_change, peak_age=s.pullback_days,
        platform_range=s.platform_high / s.platform_low - 1,
        trend_return=s.trend_return_at_peak, trend_slope=s.trend_slope_at_peak,
        volatility=s.volatility, quote_coverage=s.index_member_coverage))
    f[cfg['features']] = f[cfg['features']].replace([np.inf, -np.inf], np.nan)
    return f


def purged_history(frame, start, calendar, cfg):
    """所有训练标签必须在隔离区前成熟，且确认日落在训练窗口内。"""
    i = calendar.searchsorted(pd.Timestamp(start))
    cutoff = calendar[max(0, i - cfg['embargo_sessions'])]
    lower = pd.Timestamp(start) - pd.DateOffset(years=cfg['training_years'])
    mask = frame.date.ge(lower) & frame.date.lt(start) & frame.end_date.lt(cutoff) & frame.complete & frame.target.notna()
    return frame[mask].copy(), cutoff


def inner_split(train, calendar, cfg):
    dates = np.sort(train.date.unique())
    n = max(cfg['minimum_validation_dates'], int(np.ceil(len(dates) * cfg['validation_fraction'])))
    if n >= len(dates):
        raise ValueError('内层日期不足')
    start = pd.Timestamp(dates[-n])
    fit, cutoff = purged_history(train, start, calendar, cfg)
    valid = train[train.date >= start].copy()
    if len(fit) < cfg['minimum_train_events'] or len(valid) < cfg['minimum_validation_events']:
        raise ValueError(f'内层样本不足: {len(fit)}/{len(valid)}')
    return fit, valid, cutoff


def sample_weights(dates, asof, half_life):
    dates = pd.Series(dates)
    weights = 1 / dates.map(dates.value_counts()).to_numpy(float)
    if half_life is not None:
        weights *= np.exp2(-(pd.Timestamp(asof) - dates).dt.days.to_numpy() / half_life)
    return weights / weights.mean()


class TrainScaler:
    """缺失填充、位置和尺度均只在拟合样本估计。"""
    def __init__(self, cfg):
        self.floor, self.clip = cfg['scaler_floor'], cfg['feature_clip']

    def fit(self, x):
        x = np.asarray(x, float)
        self.median = pd.DataFrame(x).median().fillna(0).to_numpy()
        filled = np.where(np.isfinite(x), x, self.median)
        self.mean = filled.mean(axis=0)
        self.scale = np.maximum(filled.std(axis=0), self.floor)
        return self

    def transform(self, x):
        x = np.asarray(x, float)
        x = np.where(np.isfinite(x), x, self.median)
        return np.clip((x - self.mean) / self.scale, -self.clip, self.clip).astype('float32')


def fuzzy_score(frame, medians, cfg):
    """固定软规则。多条规则允许部分满足，不重复设硬筛选条件。"""
    c = cfg['fuzzy']
    f = frame[cfg['features']].fillna(medians)
    value = np.exp(-.5 * ((f.reference_distance - c['reference_center']) / c['reference_width']) ** 2)
    strength = np.clip((f.rally - c['rally_floor']) / c['rally_span'], 0, 1)
    slow = np.clip(1 - f.slowdown / c['slowdown_ceiling'], 0, 1)
    turnover = np.exp(-.5 * ((f.log_turnover - np.log(c['turnover_center'])) / c['log_turnover_width']) ** 2)
    breadth = np.clip((f.breadth_change - c['breadth_floor']) / c['breadth_span'], 0, 1)
    rules = np.stack([np.sqrt(value * slow), np.sqrt(value * strength),
                      np.sqrt(turnover * slow), np.sqrt(value * breadth)], axis=1)
    return rules @ np.asarray(c['weights'])


class FuzzyNetwork(nn.Module):
    """高斯前件＋线性后件，归一化规则激活在对数域计算。"""
    def __init__(self, x, rules, seed, cfg):
        super().__init__()
        torch.manual_seed(seed)
        centers = KMeans(n_clusters=rules, random_state=seed, n_init=10).fit(x).cluster_centers_
        self.centers = nn.Parameter(torch.tensor(centers, dtype=torch.float32))
        width = np.log(np.expm1(cfg['width_initial'] - cfg['width_floor']))
        self.raw_width = nn.Parameter(torch.full_like(self.centers, float(width)))
        self.consequents = nn.Parameter(torch.zeros(rules, x.shape[1] + 1))
        self.width_floor = cfg['width_floor']

    def forward(self, x):
        width = torch.nn.functional.softplus(self.raw_width) + self.width_floor
        log_activation = -.5 * (((x[:, None, :] - self.centers) / width) ** 2).sum(dim=2)
        activation = torch.softmax(log_activation, dim=1)
        augmented = torch.cat([x, torch.ones(len(x), 1, device=x.device)], dim=1)
        local = augmented @ self.consequents.T
        return (activation * local).sum(dim=1)


def fit_network(x, y, weights, rules, lr, seed, cfg, validation=None, epochs=None):
    """验证集只用于内层早停；最终训练轮数在测试前冻结。"""
    model = FuzzyNetwork(x, rules, seed, cfg)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=cfg['weight_decay'])
    x, y, weights = [torch.tensor(a, dtype=torch.float32) for a in (x, y, weights)]
    if validation is not None:
        vx, vy, vw = [torch.tensor(a, dtype=torch.float32) for a in validation]
    maximum = epochs or cfg['maximum_epochs']
    best_loss, best_epoch, best_state, log = np.inf, 0, None, []
    for epoch in range(1, maximum + 1):
        optimizer.zero_grad()
        losses = torch.nn.functional.smooth_l1_loss(model(x), y, reduction='none')
        loss = (losses * weights).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['gradient_clip'])
        optimizer.step()
        if epoch % cfg['check_interval'] == 0 or epoch == maximum:
            with torch.no_grad():
                vloss = float((torch.nn.functional.smooth_l1_loss(model(vx), vy, reduction='none') * vw).mean()) if validation is not None else np.nan
            log.append(dict(epoch=epoch, train_loss=float(loss.detach()), validation_loss=vloss))
            if validation is not None:
                if epoch >= cfg['minimum_epochs'] and vloss < best_loss - cfg['improvement_tolerance']:
                    best_loss, best_epoch, best_state = vloss, epoch, copy.deepcopy(model.state_dict())
                if epoch >= cfg['minimum_epochs'] and best_epoch and epoch - best_epoch >= cfg['patience_epochs']:
                    break
    if validation is not None:
        model.load_state_dict(best_state)
    else:
        best_epoch = maximum
    return model, best_epoch, best_loss, log


def predict_network(model, x):
    with torch.no_grad():
        return model(torch.tensor(x, dtype=torch.float32)).numpy()

"""基线可精确初始化的完整神经策略；所有网络参数均参与账户回报优化。"""
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from learning_search import build_features
from baseline_combination import baseline_scores, blend_scores
from rule_search import decision_dates


@dataclass
class PolicyAction:
    """收盘提交的次日动作；None 表示不调仓，空 scores 表示退出排名持仓。"""
    scores: pd.Series
    top_n: int
    cash_fraction: float = 1.
    keep: set = field(default_factory=set)
    exit: set = field(default_factory=set)
    reduce: dict = field(default_factory=dict)
    weights: dict = field(default_factory=dict)

    def validate(self, codes):
        if (self.top_n != int(self.top_n) or not 1 <= self.top_n <= 5
                or not np.isfinite(self.cash_fraction) or not 0 <= self.cash_fraction <= 1
                or self.scores.index.has_duplicates or not self.scores.index.equals(codes)
                or np.isinf(self.scores.to_numpy()).any()
                or not (self.keep | self.exit | set(self.reduce) | set(self.weights)) <= set(codes)
                or any(not np.isfinite(x) or not 0 <= x <= 1 for x in self.reduce.values())
                or any(not np.isfinite(x) or x <= 0 for x in self.weights.values())):
            raise ValueError('非法策略动作')


def make_package(wide, eligible, calendar, specs, baseline_params, equal_params, start, end):
    """市场数据调用前应已物理截断；本函数只做因果滚动与当日截面变换。"""
    dates = eligible.loc[start:end].index
    factors = build_features(wide, eligible, calendar, specs)
    raw = baseline_scores(wide, eligible, calendar, baseline_params)
    raw.append(blend_scores(raw, equal_params['weights']))
    ranked = [r.rank(axis=1, pct=True) - .5 for r in raw]
    arrays = [f.reindex(index=dates, columns=eligible.columns).to_numpy() for f in factors.values()]
    z = np.stack(arrays, axis=2)
    seed_scores = np.stack([r.loc[dates].to_numpy() for r in ranked], axis=2)
    # 模型可以重新开放原基线的研究性过滤；仅冻结共同的数据资格边界。
    features = np.concatenate([np.nan_to_num(z), np.isnan(z).astype(float),
                               np.nan_to_num(seed_scores), np.where(np.isfinite(seed_scores), 1., -1.)], axis=2)
    schedules = np.stack([np.where(dates.isin(decision_dates(dates, f, calendar)), 1., -1.)
                          for f in ['daily', 'weekly', 'monthly']], axis=1)
    return dict(dates=dates, codes=eligible.columns, features=features,
                eligible=eligible.loc[dates].to_numpy(bool), schedules=schedules,
                policies=[*baseline_params, equal_params],
                seed_score_start=2 * len(specs), seed_valid_start=2 * len(specs) + 4,
                feature_names=[*factors, *[k + '_missing' for k in factors],
                               *['seed_rank_' + str(i) for i in range(4)],
                               *['seed_valid_' + str(i) for i in range(4)]])


class NeuralPolicy:
    """共享股票网络 + 独立全局动作头；可变线性通路用于无误差基线编译。"""
    STOCK_STATE = ['held', 'weight', 'sellable_fraction', 'holding_age', 'holding_return',
                   'pending_exit', 'last_fill', 'last_rejection']
    GLOBAL_STATE = ['bias', 'daily', 'weekly', 'monthly', 'cash_fraction', 'drawdown',
                    'equity_return', 'n_hold', 'daily_return', 'rejections']
    HEADS = ['score', 'entry', 'hold_exit', 'reduce', 'allocation']

    @staticmethod
    def shapes(feature_dim, hidden):
        d = feature_dim + len(NeuralPolicy.STOCK_STATE) + len(NeuralPolicy.GLOBAL_STATE)
        return [('hidden_weight', (d, hidden)), ('hidden_bias', (hidden,)),
                ('stock_weight', (hidden, 5)), ('stock_linear', (d, 5)),
                ('global_weight', (hidden + len(NeuralPolicy.GLOBAL_STATE), 3))]

    @classmethod
    def initialize(cls, package, hidden, host, seed, random_start=False):
        rng = np.random.default_rng(seed)
        blocks = {}
        for name, shape in cls.shapes(package['features'].shape[2], hidden):
            blocks[name] = np.zeros(shape)
        blocks['hidden_weight'][:] = rng.normal(0, 1 / np.sqrt(blocks['hidden_weight'].shape[0]), blocks['hidden_weight'].shape)
        g0 = package['features'].shape[2] + len(cls.STOCK_STATE)
        if random_start:
            for name in ['stock_weight', 'stock_linear', 'global_weight']:
                blocks[name][:] = rng.normal(0, 1 / np.sqrt(blocks[name].shape[0]), blocks[name].shape)
            # 随机对照共用可交易的输出尺度，完全不读取宿主评分/频率/持股数参数。
            blocks['global_weight'][hidden, 1:] = [3., .75]
        else:
            p = package['policies'][host]
            blocks['stock_linear'][package['seed_score_start'] + host, 0] = 1.
            blocks['stock_linear'][package['seed_valid_start'] + host, 1] = 1.
            freq = ['daily', 'weekly', 'monthly'].index(p['frequency'])
            blocks['global_weight'][hidden + 1 + freq, 0] = 1.
            blocks['global_weight'][hidden, 1:] = [p['top_n'], 1.]
        theta = np.concatenate([blocks[name].ravel() for name, _ in cls.shapes(package['features'].shape[2], hidden)])
        return theta

    @classmethod
    def scales(cls, feature_dim, hidden):
        # 各层按输入维度预条件；不限制参数到初始基线的距离。
        return np.concatenate([np.full(np.prod(shape), 1 / np.sqrt(shape[0]) if len(shape) == 2 else 1.)
                               for _, shape in cls.shapes(feature_dim, hidden)])

    def __init__(self, package, theta, hidden, capital, trace=True):
        self.package, self.hidden, self.capital, self.trace = package, hidden, capital, trace
        self.lookup = {d: i for i, d in enumerate(package['dates'])}
        self.blocks, at = {}, 0
        for name, shape in self.shapes(package['features'].shape[2], hidden):
            size = int(np.prod(shape))
            self.blocks[name] = np.array(theta[at:at + size]).reshape(shape)
            at += size
        if at != len(theta) or not np.isfinite(theta).all():
            raise ValueError('策略参数维度或数值无效')
        self.peak, self.previous_equity = capital, capital
        self.held_since, self.entry_price = {}, {}
        self.records, self.actions = [], []

    def __call__(self, snapshot):
        date, positions = snapshot['date'], snapshot['positions']
        i, codes = self.lookup[date], self.package['codes']
        eq = snapshot['equity']
        self.peak = max(self.peak, eq)
        rejected = {e['ts_code'] for e in snapshot['events'] if e['kind'] in ['buy_skipped', 'sell_deferred']}
        fills = {t['ts_code']: 1. if t['side'] == 'buy' else -1. for t in snapshot['trades']}
        glob = np.array([1., *self.package['schedules'][i], snapshot['cash'] / eq,
                         eq / self.peak - 1., np.clip(eq / self.capital - 1, -1, 2),
                         len(positions) / 5., eq / self.previous_equity - 1., len(rejected) / 5.])
        self.previous_equity = eq
        self.held_since = {c: v for c, v in self.held_since.items() if c in positions}
        self.entry_price = {c: v for c, v in self.entry_price.items() if c in positions}
        states = np.zeros((len(codes), len(self.STOCK_STATE)))
        for j, code in enumerate(codes):
            pos = positions.get(code)
            if pos:
                self.held_since.setdefault(code, i)
                self.entry_price.setdefault(code, pos['price'] * pos['adj_factor'])
                states[j, :6] = [1., pos['value'] / eq, pos['available'] / pos['shares'],
                                 min(1., (i - self.held_since[code]) / 244.),
                                 np.clip(pos['price'] * pos['adj_factor'] / self.entry_price[code] - 1, -1, 2),
                                 float(code in snapshot['pending_sell'])]
            states[j, 6:] = [fills.get(code, 0.), float(code in rejected)]
        x = np.concatenate([self.package['features'][i], states, np.tile(glob, (len(codes), 1))], axis=1)
        b = self.blocks
        h = np.tanh(x @ b['hidden_weight'] + b['hidden_bias'])
        stock = x @ b['stock_linear'] + h @ b['stock_weight']
        global_output = np.r_[h.mean(axis=0), glob] @ b['global_weight']
        rebalance = bool(global_output[0] > 0)
        top_n = int(np.clip(np.floor(global_output[1] + .5), 1, 5))
        cash_fraction = float(np.clip(global_output[2], 0, 1))
        score = stock[:, 0].copy()
        score[~self.package['eligible'][i] | (stock[:, 1] <= 0)] = np.nan
        if self.trace:
            self.records.append(dict(date=date, rebalance=rebalance, top_n=top_n,
                                     cash_fraction=cash_fraction, cash=snapshot['cash'], equity=eq,
                                     drawdown=glob[5], n_hold=len(positions), rejections=len(rejected),
                                     candidates=int(np.isfinite(score).sum())))
        action = None
        if rebalance:
            action = PolicyAction(pd.Series(score, index=codes), top_n, cash_fraction,
                keep={c for c, y in zip(codes, stock[:, 2]) if c in positions and y > .5},
                exit={c for c, y in zip(codes, stock[:, 2]) if c in positions and y < -.5},
                reduce={c: float(np.clip(y - .5, 0, 1)) for c, y in zip(codes, stock[:, 3]) if c in positions},
                weights={c: float(np.exp(np.clip(y, -3, 3))) for c, y in zip(codes, stock[:, 4])})
        if self.trace and action is not None:
            self.actions.append((date, action))
        return action


class AnnualPolicy:
    """年度模型只按已锁定参数切换；保留跨年真实持仓记忆与净值峰值。"""
    def __init__(self, policies):
        self.policies, self.previous = policies, None

    def __call__(self, snapshot):
        policy = self.policies[snapshot['date'].year]
        if self.previous is not None and policy is not self.previous:
            old = self.previous
            policy.peak, policy.previous_equity = old.peak, old.previous_equity
            # 各年度包共用从 2019 年开始的完整因果日期轴。
            policy.held_since, policy.entry_price = dict(old.held_since), dict(old.entry_price)
        self.previous = policy
        return policy(snapshot)

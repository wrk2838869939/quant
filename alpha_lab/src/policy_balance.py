"""训练窗、迭代预算、重训间隔的历史可见选择与连续账户调度。"""
from bisect import bisect_right
import numpy as np
import pandas as pd
from policy_evolution import update, fitness_key


def stable_update(evaluated, theta, champion, multiplier, cfg):
    """同适应度保留原冠军，中心仍可探索；避免平台期冠军无收益漂移。"""
    center, new_theta, best, scale, success, elite = update(evaluated, theta, champion, multiplier, cfg)
    if cfg['keep_incumbent_on_equal_fitness'] and fitness_key(best, cfg['max_drawdown']) == fitness_key(champion, cfg['max_drawdown']):
        new_theta, best = theta.copy(), champion
    return center, new_theta, best, scale, success, elite


def select_balance(rows, asof, cfg):
    """仅接受决策时已经结束的完整历史记录；不对网格作单调插值。"""
    asof = pd.Timestamp(asof)
    if any(pd.Timestamp(r['evaluation_end']) >= asof for r in rows):
        raise ValueError('选择器收到尚未结束的评估数据')
    valid = [r for r in rows if r['quarters'] >= cfg['minimum_completed_quarters']
             and r['metrics']['feasible'] and r['metrics']['total'] > 0
             and (not cfg['require_nonnegative_excess'] or r['log_excess'] >= 0)]
    if not valid:
        return None
    best = max(r['score'] for r in valid)
    near = [r for r in valid if r['score'] >= best - cfg['score_tie_band']]
    # 近似同分时先降低年度训练次数与迭代预算，再看实测稳定得分。
    return min(near, key=lambda r: (r['generation'] * 12 / r['retrain_months'],
                                    -r['score'], r['window_months'], r['id']))


class RollingPolicy:
    """按锁定日期切换参数；继承持仓年龄、成本、峰值与昨日净值。"""
    def __init__(self, policies):
        self.policies = {pd.Timestamp(d): p for d, p in policies.items()}
        self.dates = sorted(self.policies)
        self.previous = None

    def __call__(self, snapshot):
        date = pd.Timestamp(snapshot['date'])
        at = bisect_right(self.dates, date) - 1
        if at < 0:
            raise ValueError('账户日期早于首个锁定策略')
        policy = self.policies[self.dates[at]]
        if self.previous is not None and policy is not self.previous:
            old = self.previous
            policy.peak, policy.previous_equity = old.peak, old.previous_equity
            # 对不同起点的日期轴显式平移，不能把局部下标直接复制。
            old_i = old.lookup[date]
            new_i = policy.lookup[date]
            policy.held_since = {c: new_i - (old_i - i) for c, i in old.held_since.items()}
            policy.entry_price = dict(old.entry_price)
        self.previous = policy
        return policy(snapshot)


def quarter_excess(daily, baseline):
    """同一日历与同一初始账户的季度对数相对收益，不剔除亏损季度。"""
    pd.testing.assert_index_equal(daily.index, baseline.index)
    logs = np.log1p(daily['ret']) - np.log1p(baseline['ret'])
    return logs.groupby(logs.index.to_period('Q')).sum()

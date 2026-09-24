"""延长既有演化轨迹；生成候选、更新与停止判据不接触验证或外层结果。"""
import copy
import numpy as np


def fitness_key(row, max_drawdown=.30):
    """与首轮同一可行性、净收益、换手排序。"""
    m = row['metrics']
    violation = max(0., -min(m['mdd'], m['worst_block_mdd']) - max_drawdown)
    return (int(m['feasible']), -violation, m['total'], -m['turnover'])


def candidates(center, champion, scales, multiplier, rng, cfg):
    """保留旧版随机数调用顺序、正负方向和三尺度，不增加新搜索机制。"""
    result = []
    for pair in range(cfg['population'] // 2):
        sigma = cfg['scales'][pair % len(cfg['scales'])] * multiplier
        anchor = center if pair % 2 == 0 else champion
        noise = rng.normal(size=len(center)) * scales * sigma
        for direction in [1, -1]:
            result.append((anchor + direction * noise, sigma, pair, direction))
    return result


def update(evaluated, champion_theta, champion, multiplier, cfg):
    """逐表达式沿用首轮更新；返回可保存的搜索状态。"""
    key = lambda row: fitness_key(row, cfg['max_drawdown'])
    success = np.mean([key(row) > key(champion) for _, row in evaluated])
    elite = sorted(evaluated + [(champion_theta, champion)], key=lambda pair: key(pair[1]), reverse=True)[:cfg['elites']]
    weights = np.log(cfg['elites'] + .5) - np.log(np.arange(1, cfg['elites'] + 1))
    weights /= weights.sum()
    center = sum(w * theta for w, (theta, _) in zip(weights, elite))
    theta, best = elite[0][0].copy(), elite[0][1]
    multiplier = float(np.clip(multiplier * (cfg['adaptation_up'] if success > cfg['success_target'] else cfg['adaptation_down']),
                               cfg['min_scale_multiplier'], cfg['max_scale_multiplier']))
    return center, theta, best, multiplier, float(success), [row for _, row in elite]


def plateau(archive, generation, cfg):
    """只看已完成训练轨迹；三个连续8代区间均低于阈值才登记平台。"""
    width, blocks = cfg['block_generations'], cfg['required_flat_blocks']
    if generation < width * blocks:
        return dict(plateau=False, reason='insufficient_history', windows=[])
    windows = []
    for end in range(generation - width * (blocks - 1), generation + 1, width):
        a, b = archive[end-width]['metrics'], archive[end]['metrics']
        gain = float(np.log1p(b['total']) - np.log1p(a['total'])) if min(a['total'], b['total']) > -1 else float('inf')
        feasible_transition = bool(b['feasible'] and not a['feasible'])
        flat = bool(a['feasible'] and b['feasible'] and not feasible_transition and gain < cfg['min_log_wealth_improvement'])
        windows.append(dict(start=end-width, end=end, log_wealth_gain=gain, flat=flat))
    return dict(plateau=all(w['flat'] for w in windows), reason='three_flat_training_blocks' if all(w['flat'] for w in windows) else 'training_still_active_or_infeasible', windows=windows)


def restore_rng(state):
    """精确保留续训随机流，避免重启变成新随机种子实验。"""
    rng = np.random.default_rng()
    rng.bit_generator.state = copy.deepcopy(state)
    return rng

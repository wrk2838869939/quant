"""固定协议后重建全A逐日排名、平滑候选、排名退出及原年度风险折扣。"""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import shutil
import sys

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
from trend_entry import entry_features
from patient_trend import trend_state
from selection_stability import smooth_scores, buffered_holds
from lifecycle_risk import risk_discount


def digest(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p, value): p.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str))


def main():
    cfg = yaml.safe_load((ROOT / 'configs/selection_stability_research.yaml').read_text())
    parent, risk_parent = ROOT / cfg['parent'], ROOT / cfg['risk_parent']
    previous = ROOT / cfg['startup_parent']
    out = ROOT / cfg['output_root'] / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    out.mkdir(parents=True); print(out, flush=True)
    shutil.copy2(ROOT / 'configs/selection_stability_research.yaml', out / 'config.yaml')
    source_names = ['scripts/build_selection_stability.py', 'src/selection_stability.py',
                    'src/trend_entry.py', 'src/patient_trend.py', 'src/lifecycle_risk.py',
                    'src/risk_budget_portfolio.py', 'src/selection_first.py', 'src/retail_rebalance.py',
                    'src/startup_timing.py', 'tests/test_selection_stability.py']
    for name in source_names:
        p = out / 'sources' / name; p.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(ROOT / name, p)
    parents = {}
    for p in [parent, risk_parent, previous]:
        receipt = json.loads((p / 'completion.json').read_text())
        for name, expected in receipt['artifact_sha256'].items(): assert digest(p / name) == expected, name
        parents[str(p)] = dict(receipt_sha256=digest(p / 'completion.json'), artifacts=len(receipt['artifact_sha256']))
    (out / 'PROTOCOL.md').write_text('''# 选择持续性与排名缓冲：事前固定协议

主参数：20个交易日每日合格截面分位分数的均值，原分数+.5后无资格/未知日记0，当前日仍须有资格。保留原60日换手均值、60/120日趋势、当时全A横截面、原常数收益预测与年度风险模型。
排名缓冲：当前仍合格但连续两次正常月度复核排名在10名之外，原Hybrid持有许可才额外置0；未知中断计数，并列取最优名次。原退出和未知不会改为持有。这是排名退出代理，未把分位差当收益差，不保证替代股成交。
四主政策baseline/smooth20/rank_buffer/combined；10日与40日平滑只做长期8账户邻域验证，不按收益重新选参数。
首次启动：沿用父实验15个季度窗口×8/15/22日、同窗244交易日，每组8万元。首次用当日新信号，随后回原月初。原45个baseline复用封存结果；逐一重放三个预定baseline后运行135个候选账户。排名退出的历史确认只在原月初累计，首次插入行不增加/清零计数。
长期检验：2022/2024空仓起始×1/8/15/22日历×四政策共32账户，及10/40日邻域16账户；起始资本同为8万元。候选池须全A重建，新增股票补完整执行数据，不能因为缺数据删股。
原引擎不改：5股、整手、原风险折扣、原费用、收盘冻结次日DAY限价、卖出成交后才能补位。不主动给出实盘指令。
检验门槛：首次窗口收益极差中位数比原版下降至少20%；长期8配对年化中位差不低于-2个百分点、最大回撤中位差不低于-1个百分点、成交数增幅中位数不超过25%。分别报告，不用加权总分掩盖退化。
报告所有账户与邻域、资金股数对账、初始股票重合、排名退出数、换手/费用、逐股损益集中。三个预定真实账户前缀截断；全A特征和每日平滑前缀验证；年度风险模型不重训，复算原风险在原代码日期上保持一致。
历史已观察、窗口重叠，不叫全新样本外。全部结果完成并封存才汇报，不修改生产或根AGENTS.md。
''')
    inputs = [ROOT / cfg['feature_projection'], parent / 'inputs/basics.parquet', parent / 'inputs/st.parquet',
              risk_parent / 'risk/verification.json', risk_parent / 'risk/annual_reference.csv', risk_parent / 'risk/discounts.parquet']
    inputs += list((parent / 'schedules').glob('calendar_*/*.parquet'))
    save(out / 'manifest.json', dict(config=cfg, source_sha256={n: digest(ROOT / n) for n in source_names},
         input_sha256={str(p): digest(p) for p in inputs}, parent_verification=parents,
         protocol_sha256=digest(out / 'PROTOCOL.md'), config_sha256=digest(out / 'config.yaml'),
         root_agents_sha256=digest(ROOT.parent / 'AGENTS.md')))
    raw = pd.read_parquet(ROOT / cfg['feature_projection'])
    wide = {k: raw.pivot(index='date', columns='ts_code', values=k).sort_index()
            for k in ['hfq_close', 'total_mv_yuan', 'turnover_rate']}
    del raw
    close = wide['hfq_close']; spec = yaml.safe_load((parent / 'config.yaml').read_text())
    features = entry_features(close, wide['total_mv_yuan'], wide['turnover_rate'],
                              medium=spec['trend']['medium_window'], long=spec['trend']['long_window'])
    _, trend = trend_state(close, spec['trend'])
    valid = pd.DataFrame(np.logical_and.reduce([np.isfinite(v.to_numpy()) for v in features.values()]),
                         index=close.index, columns=close.columns)
    # 只保存研究区间及足够平滑前史，但所有原特征使用2014起完整历史计算。
    dates = close.index[(close.index >= '2021-10-01') & (close.index <= cfg['end'])]
    daily = pd.DataFrame(np.nan, index=dates, columns=close.columns)
    basics = pd.read_parquet(parent / 'inputs/basics.parquet').set_index('ts_code')
    listed = pd.to_datetime(basics.list_date).reindex(close.columns)
    delisted = pd.to_datetime(basics.delist_date).reindex(close.columns)
    st = pd.read_parquet(parent / 'inputs/st.parquet')
    assert (st.known_date <= st.start).all()
    main_board = close.columns.str.match(r'^(?:(?:000|001|002|003)\d{3}\.SZ|(?:600|601|603|605)\d{3}\.SH)$')
    for date in dates:
        eligible = valid.loc[date] & (date - listed).dt.days.ge(spec['min_listing_days']) & (delisted.isna() | delisted.gt(date))
        active = st.loc[st.known_date.le(date) & st.start.le(date) & (st.end.isna() | st.end.ge(date)), 'ts_code']
        eligible.loc[eligible.index.isin(active)] = False
        # 与旧九特征DataFrame的turnover列rank完全同义；排名仍在全A合格截面内算。
        rank = features['turnover'].loc[date, eligible].rank(pct=True) - .5
        chosen = rank.index[rank.index.isin(close.columns[main_board])]
        daily.loc[date, chosen] = (-rank.loc[chosen]).where(trend.loc[date, chosen])
    (out / 'signals').mkdir(); daily.to_parquet(out / 'signals/daily_original.parquet')
    smooth = {w: smooth_scores(daily, w) for w in cfg['smoothing_windows']}
    for window, values in smooth.items(): values.to_parquet(out / f'signals/daily_smooth{window}.parquet')
    needed = set(); parity = []; hold_stats = []
    for day in cfg['calendar_days']:
        old = {k: pd.read_parquet(parent / f'schedules/calendar_{day:02d}/{k}.parquet') for k in ['signals', 'forecasts', 'hold']}
        pd.testing.assert_frame_equal(daily.loc[old['signals'].index], old['signals'], check_exact=True)
        parity.append(dict(calendar=day, original_signals_exact=True))
        for policy in cfg['policies'] + cfg['neighbors']:
            if policy in ['baseline', 'rank_buffer']: score = old['signals']
            else:
                window = cfg['primary_window'] if policy == 'combined' else int(policy.removeprefix('smooth'))
                score = smooth[window].loc[old['signals'].index]
            hold = old['hold']
            if policy in ['rank_buffer', 'combined']:
                hold, ranks, exits = buffered_holds(score, hold, cfg['rank_buffer'], cfg['rank_confirmations'], cfg['max_positions'])
                hold_stats.append(dict(calendar=day, policy=policy, additional_false_cells=int(((hold == 0) & (old['hold'] == 1)).sum().sum())))
            folder = out / f'schedules/calendar_{day:02d}/{policy}'; folder.mkdir(parents=True)
            for key, value in [('signals', score), ('hold', hold), ('forecasts', old['forecasts'])]: value.to_parquet(folder / f'{key}.parquet')
            needed.update(set().union(*(set(row.dropna().nlargest(20).index) for _, row in score.iterrows())))
    old_risk = pd.read_parquet(risk_parent / 'risk/discounts.parquet')
    needed.update(old_risk.columns)
    codes = sorted(needed); pd.DataFrame({'ts_code': codes}).to_csv(out / 'execution_codes.csv', index=False)
    price = close[codes].where(close[codes] > 0)
    vol = price.pct_change(fill_method=None).rolling(20, min_periods=20).std()
    dd = price / price.rolling(120, min_periods=120).max() - 1
    models = json.loads((risk_parent / 'risk/verification.json').read_text())['annual_checks']
    ref = pd.read_csv(risk_parent / 'risk/annual_reference.csv').set_index('year')
    all_reviews = sorted(set().union(*(set(pd.read_parquet(parent / f'schedules/calendar_{day:02d}/signals.parquet').index) for day in cfg['calendar_days'])))
    risk = pd.DataFrame(np.nan, index=pd.DatetimeIndex(all_reviews, name='date'), columns=close[codes].columns)
    for model in models:
        ds = risk.index[risk.index.year == model['year']]
        x = np.stack([vol.loc[ds].to_numpy(), dd.loc[ds].to_numpy()], axis=-1)
        known = np.isfinite(x).all(axis=-1)
        pred = ((x - np.array(model['center'])) / np.array(model['scale'])) @ np.array(model['coefficients']) + model['intercept']
        values = np.full(known.shape, model['uniform_discount'])
        values[known] = risk_discount(pred[known], model['reference_risk'])
        risk.loc[ds] = values
        assert pd.Timestamp(ref.loc[model['year'], 'max_label_end']) < pd.Timestamp(f"{model['year']}-01-01")
    error = float(np.max(np.abs(risk.loc[:, old_risk.columns].to_numpy() - old_risk.loc[risk.index].to_numpy())))
    assert error < 1e-12, error
    # 旧股票保持原浮点真值，避免最后一位误差改变整手边界。
    risk.loc[:, old_risk.columns] = old_risk.loc[risk.index]
    risk.to_parquet(out / 'risk.parquet'); close[codes].to_parquet(out / 'signals/close.parquet')
    cut = pd.Timestamp('2024-06-28')
    prefix = entry_features(close.loc[:cut], wide['total_mv_yuan'].loc[:cut], wide['turnover_rate'].loc[:cut],
                            medium=spec['trend']['medium_window'], long=spec['trend']['long_window'])
    for key in features: pd.testing.assert_frame_equal(prefix[key], features[key].loc[:cut], check_exact=True)
    for w in smooth: pd.testing.assert_frame_equal(smooth_scores(daily.loc[:cut], w), smooth[w].loc[:cut], check_exact=True)
    cache = ROOT / 'data/raw/execution_audit/20220101_20260908'
    missing = [c for c in codes if any(not (cache / api / f'{c}.parquet').exists() for api in ['stk_limit', 'dividend'])]
    save(out / 'signal_completion.json', dict(codes=len(codes), new_execution_codes=missing, original_parities=parity,
         nine_full_feature_prefixes_exact=True, three_smoothing_prefixes_exact=True, risk_old_max_error=error,
         annual_labels_mature=True, additional_hold_false_cells=hold_stats))
    save(out / 'signal_hashes.json', {str(p.relative_to(out)): digest(p) for p in out.rglob('*.parquet')})
    print('signals complete', len(codes), 'codes; new', missing, 'risk error', error, flush=True)


if __name__ == '__main__': main()

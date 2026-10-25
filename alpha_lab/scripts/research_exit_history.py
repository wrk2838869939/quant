"""重建历史候选退出，检验收益与路径风险的预测及真实账户迁移。"""
from datetime import datetime, timezone
from pathlib import Path
import argparse
import json
import shutil
import sys
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
from research_profit_defense import digest, save
from patient_trend import trend_state, review_dates
from trend_entry import entry_features
from conditional_hold import build_conditional_holds
from exit_evidence import state_features, forward_marks, nonoverlapping_events
from exit_history import candidate_spells, annual_fit, predict_frame


def prepare():
    config = ROOT / 'configs/exit_history_research.yaml'
    cfg = yaml.safe_load(config.read_text())
    run = ROOT / cfg['output_root'] / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    run.mkdir(parents=True)
    shutil.copy2(config, run / 'config.yaml')
    sources = ['scripts/research_exit_history.py', 'src/exit_history.py', 'tests/test_exit_history.py',
               'src/exit_evidence.py', 'src/patient_trend.py', 'src/trend_entry.py',
               'src/conditional_hold.py', 'src/hold_sensitivity.py', 'configs/conditional_hold_research.yaml']
    for name in sources:
        target = run / 'sources' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
    inp = ROOT / cfg['execution_parent'] / 'inputs'
    inputs = [ROOT / cfg['feature_projection'], inp / 'basics.parquet', inp / 'st.parquet',
              ROOT / cfg['parent'] / 'completion.json', ROOT / cfg['rolling_parent'] / 'completion.json']
    (run / 'PROTOCOL.md').write_text('''# 历史退出与防守证据：事前协议

2015年开始，逐月首个交易日按当时完整全A合格池重算低换手排名，取主板趋势合格Top20候选。
每个候选单独启动研究生命周期，沿用Hybrid持有许可，首次许可为0即退出；未知不算退出，不因排名下降退出。
生命周期无资金与持仓总数限制，不是5股账户、交易回测或真实订单。只用于补充状态样本；不将2022以后281只候选倒推成历史股票池。
同股退出间隔严格超过60交易日才进入主样本；未来结果不得决定事件取舍。所有原事件、未退出和缺失标签保留。
9项状态特征与Ridge alpha=10保持上轮不变，新增同结构的MAE路径风险目标；期限20/60日，标签仍下一交易日收盘起算。
2018—2026各年初只用此前完全成熟标签训练，至少30事件、15日期。标准化仅拟合训练集。不调参、不选择最好起点。
分别报告历史2018—2023、近期2024+的研究事件预测，并将各年模型迁移至上轮真实账户退出主事件2024+。
迁移样本不得追加进训练。两类人群不能合并为真实账户样本量。并列历史训练均值基准、逐年结果、按退出日期等权误差及同日去均值相关。
同日去均值只作事后归因，不能把未来同日收益均值放入预测特征，也不称为完整市场/行业中性化。
路径风险按MAE<=-10%描述；高低预测组用年度训练预测中位数。缺失后续行情不填充；另外报告非右截断缺失标签全部-100%的固定预测压力情形，不称其为真实收益。
本轮只检验证据迁移，不自动准入策略或实盘。过去已被项目观察，不称为新鲜留出样本。停止本轮阈值搜索。
检查：2022+候选信号精确对齐旧月初、真实历史前缀事件一致、特征截断、标签成熟/未来扰动、独立矩阵解复算、原封存哈希与源文件快照。
''')
    save(run / 'manifest.json', dict(config_sha256=digest(run / 'config.yaml'),
        protocol_sha256=digest(run / 'PROTOCOL.md'),
        source_hashes={name: digest(ROOT / name) for name in sources},
        input_hashes={str(p): digest(p) for p in inputs},
        root_agents_sha256=digest(ROOT.parent / 'AGENTS.md')))
    print(run, flush=True)


def verify(run):
    m = json.loads((run / 'manifest.json').read_text())
    assert digest(run / 'config.yaml') == m['config_sha256']
    assert digest(run / 'PROTOCOL.md') == m['protocol_sha256']
    for name, value in m['source_hashes'].items():
        assert digest(ROOT / name) == value, name
    return yaml.safe_load((run / 'config.yaml').read_text())


def build(run):
    cfg = verify(run)
    for name, expected in json.loads((run / 'manifest.json').read_text())['input_hashes'].items():
        assert digest(name) == expected, name
    out = run / 'data'
    out.mkdir()
    raw = pd.read_parquet(ROOT / cfg['feature_projection'])
    wide = {key: raw.pivot(index='date', columns='ts_code', values=key).sort_index()
            for key in ['hfq_close', 'total_mv_yuan', 'turnover_rate']}
    del raw
    close = wide['hfq_close']
    features = entry_features(close, wide['total_mv_yuan'], wide['turnover_rate'])
    _, trend = trend_state(close, cfg['trend'])
    observable = (close.gt(0) & close.shift(60).gt(0) & close.shift(120).gt(0)
                  & close.where(close > 0).rolling(120, min_periods=120).mean().notna())
    calendar = close.index[(close.index >= cfg['start']) & (close.index <= cfg['end'])]
    reviews = review_dates(calendar, 'monthly')
    inp = ROOT / cfg['execution_parent'] / 'inputs'
    basics = pd.read_parquet(inp / 'basics.parquet').set_index('ts_code')
    listing = pd.to_datetime(basics.list_date).reindex(close.columns)
    delisting = pd.to_datetime(basics.delist_date).reindex(close.columns)
    st = pd.read_parquet(inp / 'st.parquet')
    main = close.columns.str.match(r'^(?:(?:000|001|002|003)\d{3}\.SZ|(?:600|601|603|605)\d{3}\.SH)$')
    signals = pd.DataFrame(np.nan, index=pd.DatetimeIndex(reviews, name='date'), columns=close.columns)
    coverage = []
    for date in reviews:
        frame = pd.DataFrame({key: value.loc[date] for key, value in features.items()})
        eligible = ((date - listing).dt.days.ge(cfg['min_listing_days'])
                    & (delisting.isna() | delisting.gt(date)) & np.isfinite(frame).all(axis=1))
        active_st = st.loc[st.start.le(date) & (st.end.isna() | st.end.ge(date)), 'ts_code']
        eligible.loc[eligible.index.isin(active_st)] = False
        ranks = frame.loc[eligible].rank(pct=True) - .5
        codes = ranks.index[ranks.index.isin(close.columns[main])]
        signals.loc[date, codes] = (-ranks.loc[codes, 'turnover']).where(trend.loc[date, codes])
        coverage.append(dict(date=date, all_eligible=int(eligible.sum()), main_eligible=len(codes),
                             admitted=int(signals.loc[date].notna().sum())))
    print('Reconstructed', len(reviews), 'historical full-universe reviews', flush=True)
    old = pd.read_parquet(ROOT / cfg['rolling_parent'] / 'schedules/calendar_01/signals.parquet')
    pd.testing.assert_frame_equal(signals.loc[old.index], old, check_exact=True)
    union = sorted(set().union(*(set(row.dropna().sort_values(ascending=False, kind='stable').head(cfg['top_n']).index)
                                 for _, row in signals.iterrows())))
    # 当时全市场选完候选才缩减计算列；不能先按未来候选缩小历史排名池。
    del features, wide
    price, t, o = close[union], trend[union], observable[union]
    spec = yaml.safe_load((ROOT / 'configs/conditional_hold_research.yaml').read_text())
    spec['variants'] = {'hybrid': spec['variants']['hybrid']}
    variants, _ = build_conditional_holds(price, t, o, reviews, spec)
    hold = variants['hybrid']
    sig = signals[union]
    events, entries, active = candidate_spells(sig, hold, cfg['top_n'])
    f = state_features(price, t, o)
    prefix_checks = []
    for cutoff in pd.to_datetime(cfg['prefix_cutoffs']):
        dates = reviews[reviews <= cutoff]
        short_h, _ = build_conditional_holds(price.loc[:cutoff], t.loc[:cutoff], o.loc[:cutoff], dates, spec)
        pd.testing.assert_frame_equal(short_h['hybrid'], hold.loc[:cutoff], check_exact=True)
        e, en, _ = candidate_spells(sig.loc[:cutoff], short_h['hybrid'], cfg['top_n'])
        pd.testing.assert_frame_equal(e, events[events.date <= cutoff].reset_index(drop=True), check_exact=True)
        pd.testing.assert_frame_equal(en, entries[entries.date <= cutoff].reset_index(drop=True), check_exact=True)
        sf = state_features(price.loc[:cutoff], t.loc[:cutoff], o.loc[:cutoff])
        for key in f:
            pd.testing.assert_frame_equal(sf[key], f[key].loc[:cutoff], check_exact=True)
        prefix_checks.append(dict(cutoff=cutoff, events_exact=True, hold_exact=True, features_exact=len(f)))
    events = nonoverlapping_events(events, price.index, cfg['nonoverlap_days'])
    rows = []
    for item in events.to_dict('records'):
        date, code = item['date'], item['ts_code']
        item.update({key: value.at[date, code] for key, value in f.items()})
        item['features_complete'] = bool(np.isfinite([item[key] for key in cfg['features']]).all())
        for h in cfg['horizons']:
            item.update({f'{key}_{h}': value for key, value in forward_marks(price, date, code, h).items()})
        rows.append(item)
    pd.DataFrame(rows).to_parquet(out / 'events.parquet', index=False)
    pd.DataFrame(rows).to_csv(out / 'events.csv', index=False)
    entries.to_csv(out / 'entries.csv', index=False)
    active.to_csv(out / 'open_spells.csv', index=False)
    pd.DataFrame(coverage).to_csv(out / 'universe_coverage.csv', index=False)
    for name, frame in [('close', price), ('trend', t), ('observable', o), ('signals', sig), ('hold', hold)]:
        frame.to_parquet(out / f'{name}.parquet')
    save(run / 'data_completion.json', dict(status='complete', old_monthly_signals_exact=True,
        stocks=len(union), entries=len(entries), exits=len(events), primary_events=int(events.nonoverlap_primary.sum()),
        prefix_checks=prefix_checks, hashes={p.name: digest(p) for p in out.iterdir()}))
    print('Built', len(events), 'shadow exits;', int(events.nonoverlap_primary.sum()), 'primary;', len(union), 'candidate stocks', flush=True)


def fit(run):
    cfg = verify(run)
    out = run / 'models'
    out.mkdir()
    all_events = pd.read_parquet(run / 'data/events.parquet')
    frame = all_events[all_events.nonoverlap_primary].copy()
    original = pd.read_parquet(ROOT / cfg['parent'] / 'data/exit_events.parquet')
    original = original[original.nonoverlap_primary].copy()
    predictions, audits, coefficients = [], [], []
    for h in cfg['horizons']:
        for target in cfg['targets']:
            for year in cfg['prediction_years']:
                train, model, mean, threshold = annual_fit(frame, year, h, target, cfg)
                altered = frame.copy()
                unknown = altered[f'label_end_{h}'].ge(pd.Timestamp(f'{year}-01-01')) | altered[f'label_end_{h}'].isna()
                altered.loc[unknown, f'{target}_{h}'] = 1e6
                train2, model2, mean2, threshold2 = annual_fit(altered, year, h, target, cfg)
                assert train.index.equals(train2.index) and mean == mean2 and threshold == threshold2
                if model is not None:
                    np.testing.assert_array_equal(model[-1].coef_, model2[-1].coef_)
                    for key, value in zip(cfg['features'], model[-1].coef_):
                        coefficients.append(dict(horizon=h, target=target, year=year, feature=key, standardized_coefficient=value))
                train[['date', 'ts_code', f'label_end_{h}', f'{target}_{h}']].to_csv(out / f'{target}_h{h}_{year}_train.csv', index=False)
                audits.append(dict(horizon=h, target=target, year=year, train_events=len(train), train_dates=train.date.nunique(),
                                   max_label_end=train[f'label_end_{h}'].max(), model_fitted=model is not None, future_perturbation_exact=True))
                cohorts = [('shadow', frame)] + ([('original_transfer', original)] if year >= 2024 else [])
                for cohort, data in cohorts:
                    test = data[data.date.dt.year.eq(year)].copy()
                    values = predict_frame(test, model, mean, cfg['features'])
                    np.testing.assert_array_equal(values, predict_frame(test, model2, mean2, cfg['features']))
                    for (_, row), value in zip(test.iterrows(), values):
                        predictions.append(dict(cohort=cohort, horizon=h, target=target, year=year, date=row.date, ts_code=row.ts_code,
                            prediction=value, baseline=mean, threshold=threshold, high_group=value > threshold,
                            actual=row[f'{target}_{h}'], terminal=row[f'ret_{h}'], mae=row[f'mae_{h}'],
                            label_end=row[f'label_end_{h}'], right_censored=row[f'right_censored_{h}'],
                            missing=row[f'missing_{h}'], model_fitted=model is not None))
    pd.DataFrame(predictions).to_csv(out / 'predictions.csv', index=False)
    pd.DataFrame(audits).to_csv(out / 'training_audit.csv', index=False)
    pd.DataFrame(coefficients).to_csv(out / 'coefficients.csv', index=False)
    save(out / 'completion.json', dict(status='complete', hashes={p.name: digest(p) for p in out.iterdir()}))
    print('Fitted', len(audits), 'annual target slots;', len(predictions), 'predictions', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['prepare', 'build', 'fit'], required=True)
    parser.add_argument('--run', type=Path)
    args = parser.parse_args()
    if args.stage == 'prepare':
        prepare()
    else:
        {'build': build, 'fit': fit}[args.stage](args.run.resolve())

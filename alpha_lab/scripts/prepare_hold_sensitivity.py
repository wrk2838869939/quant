"""冻结退出灵敏性实验：只从历史原价派生持有矩阵，不读取候选收益结果。"""
import argparse
import json
from pathlib import Path
import shutil
import sys

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
from patient_trend import trend_state
from hold_sensitivity import build_hold_variants
from run_patient_value_execution import digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--parent', type=Path, required=True)
    args = parser.parse_args()
    run, parent = args.run.resolve(), args.parent.resolve()
    run.mkdir(parents=True, exist_ok=False)
    out = run / 'execution'; out.mkdir()
    (run / 'hold_variants').mkdir()
    parent_manifest = json.loads((parent / 'manifest.json').read_text())
    for name, expected in parent_manifest['source_hashes'].items():
        assert digest(ROOT / name) == expected, '原基线源码变化: ' + name
    for p in (parent / 'execution').iterdir():
        if p.is_file() and (p.suffix in ['.parquet', '.yaml', '.txt'] or p.name == 'data_manifest.json'):
            shutil.copy2(p, out / p.name)
    shutil.copy2(parent / 'execution_predictions.parquet', run / 'execution_predictions.parquet')
    signals = pd.read_parquet(out / 'low_turnover_signals.parquet')
    codes = sorted(set().union(*(set(row.dropna().sort_values(ascending=False, kind='stable').head(20).index)
                                 for _, row in signals.iterrows())))
    pd.DataFrame({'ts_code': codes}).to_csv(out / 'execution_codes.csv', index=False)
    spec = yaml.safe_load((ROOT / 'configs/hold_sensitivity_research.yaml').read_text())
    projection = parent / 'data_repair/repaired_feature_projection.parquet'
    raw = pd.read_parquet(projection, columns=['date', 'ts_code', 'hfq_close'],
                          filters=[('ts_code', 'in', codes)])
    assert not raw.duplicated(['date', 'ts_code']).any()
    close = raw.pivot(index='date', columns='ts_code', values='hfq_close').sort_index()
    assert close.index.max() == pd.Timestamp(parent_manifest['end'])
    _, trend = trend_state(close, spec['trend'])
    clean = close.where(close > 0)
    observable = (clean.notna() & clean.shift(60).notna() & clean.shift(120).notna()
                  & clean.rolling(120, min_periods=120).mean().notna())
    variants, evidence = build_hold_variants(trend, observable, signals.index, spec)
    reference = pd.read_parquet(out / 'monthly_hold_allowed.parquet').reindex(columns=close.columns)
    pd.testing.assert_frame_equal(variants['baseline'], reference, check_exact=True)
    for name, frame in dict(daily_close=close, daily_trend=trend, daily_observable=observable, **evidence).items():
        frame.to_parquet(run / (name + '.parquet'))
    for name, frame in variants.items():
        frame.to_parquet(run / 'hold_variants' / (name + '.parquet'))
    kinds = [('cold',50000,'2024-01-01',1,False), ('continued',50000,'2022-01-01',1,False),
             ('small',20000,'2024-01-01',1,False), ('large',100000,'2024-01-01',1,False),
             ('cost2',50000,'2024-01-01',2,False), ('reverse',50000,'2024-01-01',1,True)]
    cases = [dict(name=kind+'_'+variant, variant=variant, capital=capital, start=start,
                  cost_multiplier=cost, reverse_input=reverse, policy='low_turnover')
             for variant in spec['variants'] for kind,capital,start,cost,reverse in kinds]
    sources = list(parent_manifest['source_hashes']) + ['src/hold_sensitivity.py',
        'configs/hold_sensitivity_research.yaml', 'scripts/run_hold_sensitivity.py',
        'scripts/prepare_hold_sensitivity.py']
    hashes = {}
    for name in sources:
        p = ROOT / name; dest = run / 'sources' / name
        dest.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(p, dest)
        hashes[name] = digest(p)
    inputs = [p for p in run.rglob('*.parquet') if 'sources' not in p.parts]
    protocol = '''# 灵敏性与鲁棒性：事前协议

所有年份均为已观察的开发数据；不训练、不改变候选Top20、分数、预测、入场或恢复规则。
只替换持有许可矩阵，月度第一个交易日收盘判断，下一交易日开盘尝试成交。
原股仍弱时不能重新入场；pending退出不可撤销；不新增日频交易。

## 主假设与对照
- baseline：原月度2次失败；fast：1次；slow：3次。
- persist10为主假设：原2次兜底；当月当前趋势弱、过去20交易日全部可观测，且至少10日弱，允许提前退出。
- persist5与persist15为事前邻域检查，不按结果挑一个替代主假设。
- “10日弱”是窗口内累计天数，允许不连续；含复核日收盘，不是10个独立预测样本。
- 未知输出NaN并打断月度计数；缺完整20日不触发日线加速。完整月度计数从2022原始起点递推，2024冷启动不重置矩阵。

## 全部检验
6政策×6情景：近期5万元冷启动、2022起连续账户、2万/10万、双倍成本、实际倒序，共36组。
评估收益/回撤/Sharpe、年别和窗口一致性、成交频率、持有期、提前退出与恢复/反弹、基线固定持仓上的配对诊断。
接受风险收益权衡的存在，不要求一切指标单向占优；但不把單一年份/单邻域的改善宣称稳健。
主假设只有在跨窗口风险调整表现与相邻门槛支持、成本/本金未推翻时才可称进一步验证候选；本轮均不实盘准入。
原价、现金、股份、公司行动、T+1/整手/限价、原基线完整复现、输入倒序不变性、未来截断和未知边界全部核验。
不依照本轮结果增添门槛、改模型、挑选年份。全部失败与中间产物保留。
'''
    (run / 'PROTOCOL.md').write_text(protocol)
    manifest = dict(status='prepared', parent=str(parent), start='2024-01-01',
        continuation_start='2022-01-01', end=parent_manifest['end'], cases=cases, parameters=spec,
        primary_variant='persist10', refit=False, fresh_holdout=False, no_result_driven_changes=True,
        source_hashes=hashes, input_hashes={str(p.relative_to(run)):digest(p) for p in inputs},
        projection_source=str(projection), projection_sha256=digest(projection),
        protocol_sha256=digest(run / 'PROTOCOL.md'), baseline_hold_exact=True,
        execution_universe=codes, hold_columns_scope='original_top20_union_only; all_A_entry_ranks_unchanged')
    (run / 'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(run=str(run),cases=len(cases),codes=len(codes),baseline_exact=True)))


if __name__ == '__main__':
    main()

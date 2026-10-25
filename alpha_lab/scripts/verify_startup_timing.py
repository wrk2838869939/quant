"""截断未来数据重新回放，核对研究账户的历史前缀与来源时间。"""
from pathlib import Path
import argparse
import hashlib
import json
import sys

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
from startup_timing import startup_matrices
from risk_budget_portfolio import run_backtest
from audit_monthly_portfolio import reconcile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    run = parser.parse_args().run
    manifest = json.loads((run / 'manifest.json').read_text())
    cfg = manifest['config']
    for name, expected in manifest['source_sha256'].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, name
    for name, expected in manifest['input_sha256'].items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == expected, name
    assert hashlib.sha256((ROOT.parent / 'AGENTS.md').read_bytes()).hexdigest() == manifest['root_agents_sha256']
    parent, risk_parent = ROOT / cfg['parent'], ROOT / cfg['risk_parent']
    inp = parent / 'inputs'
    frames = {name: pd.read_parquet(inp / (name + '.parquet'))
              for name in ['market', 'basics', 'st', 'limits', 'normalized_actions']}
    schedules = {offset: {name: pd.read_parquet(parent / f'schedules/calendar_{offset:02d}/{name}.parquet')
                         for name in ['signals', 'forecasts', 'hold']} for offset in [1, 8, 15, 22]}
    discounts = pd.read_parquet(risk_parent / 'risk/discounts.parquet')
    portfolio = yaml.safe_load((inp / 'execution_config.yaml').read_text())['portfolio']
    portfolio.update(capital=cfg['capital'], top_n=cfg['max_positions'])
    retail = yaml.safe_load((inp / 'symmetric_phase_research.yaml').read_text())['retail']
    rights = yaml.safe_load((inp / 'corporate_action_components.yaml').read_text()).get('rights_issues', [])
    causal_path = ROOT / 'outputs/causal_execution/20260913T130810_374654Z/config.yaml'
    causal = yaml.safe_load(causal_path.read_text())['causal']
    reference = pd.read_csv(risk_parent / 'risk/annual_reference.csv')
    results = []
    # 在观察完整结果前指定三个启动政策及不同年份，截断到第126个交易日。
    names = ['2022-04_15_fresh_once', '2023-01_15_stale_once', '2024-07_22_shifted_calendar']
    for name in names:
        case = next(c for c in manifest['cases'] if c['name'] == name)
        folder = run / 'accounts' / name
        original = {key: pd.read_parquet(folder / (key + '.parquet'))
                    for key in ['daily', 'trades', 'holdings', 'events']}
        cut = original['daily'].index[125]
        start = pd.Timestamp(case['start'])
        matrices, risk, lineage = startup_matrices(schedules[1], schedules[case['offset']], discounts,
                                                 case['boot_date'], start, cut, case['policy'])
        # 原始观测、信号、hold、风险矩阵均截掉未来；公司行为以公告可知日截断。
        needed = sorted(set().union(*(set(row.dropna().nlargest(20).index)
                                      for _, row in matrices['signals'].iterrows())))
        local = [frames[k][frames[k].ts_code.isin(needed)].copy()
                 for k in ['market', 'basics', 'st', 'limits', 'normalized_actions']]
        local[0] = local[0][local[0].date.between(start, cut)]
        local[3] = local[3][local[3].date.le(cut)]
        local[4] = local[4][local[4].known_date.le(cut)]
        result = run_backtest(matrices['signals'], *local, start, cut, portfolio, rights,
                              retail_forecasts=matrices['forecasts'], retail_config=retail,
                              hold_allowed=matrices['hold'], causal_config=causal,
                              risk_multipliers=risk, adjust_existing_risk=True)
        checks = reconcile(result, cfg['capital'])
        for key in ['daily', 'trades', 'holdings', 'events']:
            expected = original[key]
            actual = getattr(result, key)
            if key == 'daily':
                expected = expected.loc[:cut]
            else:
                # 截断最后一天无次日，原引擎省略收盘规划，故事件比较到前一日。
                if key == 'events':
                    expected = expected.loc[expected.date.lt(cut)]
                    actual = actual.loc[actual.date.lt(cut)]
                    # 仅未来事件出现的附加字段允许缺省；历史字段值必须逐项一致。
                    extra = expected.columns.difference(actual.columns)
                    assert expected[extra].isna().all().all()
                    actual = actual.reindex(columns=expected.columns)
                else:
                    expected = expected.loc[expected.date.le(cut)]
                expected, actual = expected.reset_index(drop=True), actual.reset_index(drop=True)
            pd.testing.assert_frame_equal(actual, expected, check_exact=True, check_dtype=False)
        results.append(dict(account=name, cutoff=str(cut.date()), exact_prefix=True, **checks))
        print(name, 'prefix passed', flush=True)
    for case in manifest['cases']:
        lineage = json.loads((run / 'accounts' / case['name'] / 'lineage.json').read_text())
        assert not lineage['future_source_used']
        if lineage['source_date']:
            assert lineage['source_date'] <= lineage['boot_date']
    # 年度风险模型的训练标签必须在该年开始前已经成熟。
    assert (pd.to_datetime(reference.max_label_end) < pd.to_datetime(reference.year.astype(str) + '-01-01')).all()
    receipt = dict(prefixes=results, source_and_input_hashes_unchanged=True,
                   root_agents_unchanged=True, all_boot_sources_not_future=True,
                   annual_risk_labels_mature=True,
                   causal_config_sha256=hashlib.sha256(causal_path.read_bytes()).hexdigest(),
                   scope='验证执行/拼接前缀，不重新训练或全量重建上游因子')
    (run / 'prefix_verification.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2, default=str))


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""固定机会上的财务可得性审计；未经版本认证，不拟合或回测。"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from financial_candidates import FIELDS, join_financial_candidates, require_verified_financial_history


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    run = parser.parse_args().run.resolve()
    out = run / 'opportunity_coverage'
    out.mkdir(exist_ok=True)
    if (out / 'financial_candidates.parquet').exists():
        raise ValueError('禁止覆盖已完成候选连接')
    parent = ROOT / 'outputs/policy_cash_alias/20260913T070201_664126Z'
    source = ROOT / 'data/raw/fina_indicator_all.parquet'
    protocol = dict(status='registered_before_join_no_return_evaluation',
                    candidate_features=['sales_growth_change_pp', 'cash_conversion_change_pp'],
                    definitions='当期单季度同比营收增速及现金流/营收比，分别减上一年同期当时可见值；单位百分点',
                    announcement_rule='ann_date严格早于复核日；同日公告次日后才可用',
                    report_rule='取已公布最大end_date；旧期晚到修订不替代新期，缺值不回填；非标准期末保留报告但不算季度特征',
                    missing_policy='保留原候选身份与数值缺失，不将缺财务变成股票不可投资',
                    stale_policy='仅报告财务期末年龄，不设事后优选阈值',
                    history_verification='必须另有原始版本证据；ann_date连接本身不构成PIT认证',
                    models_run=False, target_labels_read=False, production_changed=False,
                    no_test_market_data=True, future_comparison='若数据通过，同覆盖价量对照与两项增量比较，禁止把覆盖变化算alpha')
    (out / 'protocol.json').write_text(json.dumps(protocol, ensure_ascii=False, indent=2))
    # Parquet列投影排除全部收益/路径目标；不加载2024+财务或任何行情。
    opportunities = pd.read_parquet(parent / 'policy_features.parquet', columns=['date', 'ts_code', 'trend'])
    assert opportunities.date.max() < pd.Timestamp('2024-01-01')
    reports = pd.read_parquet(source, columns=['ts_code', 'ann_date', 'end_date', *FIELDS],
                              filters=[('ann_date', '<=', '20231231'), ('end_date', '<=', '20231231')])
    result = join_financial_candidates(opportunities, reports)
    result = result.merge(opportunities, on=['date', 'ts_code'], validate='one_to_one')
    assert len(result) == len(opportunities)
    for field in ['report_ann', 'prior_ann']:
        assert (result[field].isna() | result[field].lt(result.date)).all()
    result.to_parquet(out / 'financial_candidates.parquet', index=False)
    summaries = []
    for year, group in result.groupby(result.date.dt.year):
        for population, sample in [('all', group), ('trend', group[group.trend])]:
            known = sample[sample.report_end.notna()]
            summaries.append(dict(year=int(year), population=population, rows=len(sample),
                latest_report_known=int(sample.report_end.notna().sum()),
                same_quarter_last_year_known=int(sample.prior_end.notna().sum()),
                both_changes_finite=int(sample.numeric_available.sum()),
                numeric_coverage=float(sample.numeric_available.mean()),
                report_age_median=known.report_age_days.median(),
                report_age_p95=known.report_age_days.quantile(.95),
                report_age_over365=int(known.report_age_days.gt(365).sum()),
                history_verified=int(sample.source_history_verified.sum())))
    pd.DataFrame(summaries).to_csv(out / 'coverage_by_year.csv', index=False)
    # 固定原五只，不用价格、收益或后续表现决定检验样本。
    held = pd.DataFrame({'date': pd.Timestamp('2022-01-25'),
                         'ts_code': ['600828.SH', '601006.SH', '601018.SH', '601811.SH', '603868.SH']})
    join_financial_candidates(held, reports).to_csv(out / 'first_alarm_five_financial_state.csv', index=False)
    blocked = False
    try:
        require_verified_financial_history(result)
    except ValueError:
        blocked = True
    assert blocked
    # 截短未来财务事件、倒序重排源记录必须不改变已经发生的机会。
    early = opportunities[opportunities.date.eq(pd.Timestamp('2021-12-01'))]
    earlier_reports = reports[reports.ann_date.lt('20211201')].iloc[::-1]
    reproduced = join_financial_candidates(early, earlier_reports).sort_values('ts_code').reset_index(drop=True)
    original = result[result.date.eq(pd.Timestamp('2021-12-01'))].drop(columns='trend').sort_values('ts_code').reset_index(drop=True)
    pd.testing.assert_frame_equal(original, reproduced)
    manifest = dict(status='complete_data_audit_only', rows=len(result), reports_filtered=len(reports),
                    source_version_gate='unverified_blocked', models_run=0, target_labels_read=False,
                    first_alarm_fixed_five=True, no_future_event_and_order_check_rows=len(early),
                    sources={str(path): sha(path) for path in [Path(__file__).resolve(), ROOT / 'src/financial_candidates.py',
                             ROOT / 'tests/test_financial_candidates.py', source, parent / 'policy_features.parquet']},
                    files={p.name: sha(p) for p in out.iterdir() if p.is_file() and p.name != 'manifest.json'})
    (out / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(json.dumps({key: manifest[key] for key in ['status', 'rows', 'reports_filtered', 'models_run', 'no_future_event_and_order_check_rows']}))


if __name__ == '__main__':
    main()

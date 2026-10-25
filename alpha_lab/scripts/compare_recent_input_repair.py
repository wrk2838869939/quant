#!/usr/bin/env python3
"""修复前后输入身份与数值差异；不比较或读取账户收益。"""
import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
import numpy as np


def compare(old, new):
    old, new = Path(old), Path(new)
    checks = {}
    monthly = []
    score_changes = []
    selected_changes = []
    for name in ['pit_review_features', 'execution_predictions']:
        before = pd.read_parquet(old / (name + '.parquet')).set_index(['date', 'ts_code']).sort_index()
        after = pd.read_parquet(new / (name + '.parquet')).set_index(['date', 'ts_code']).sort_index()
        common = before.index.intersection(after.index)
        added, removed = after.index.difference(before.index), before.index.difference(after.index)
        left, right = before.loc[common], after.loc[common]
        changed = ~((left == right) | (left.isna() & right.isna()))
        differences = {}
        for col in changed.columns[changed.any()]:
            mask = changed[col]
            item = {'changed_cells': int(mask.sum()), 'first_date': str(left.index[mask][0][0].date())}
            if pd.api.types.is_numeric_dtype(left[col]) and not pd.api.types.is_bool_dtype(left[col]):
                item['max_absolute_difference'] = float(np.abs(left.loc[mask, col] - right.loc[mask, col]).max())
            differences[col] = item
        if name == 'execution_predictions':
            col = 'low_turnover__score'
            for identity in changed.index[changed[col]]:
                score_changes.append({'date': identity[0], 'ts_code': identity[1], 'before': left.loc[identity, col], 'after': right.loc[identity, col]})
            for date in sorted(set(before.index.get_level_values('date'))):
                for top_n in [5, 20]:
                    old_top = before.loc[date].dropna(subset=[col]).sort_values([col, 'ts_code'], ascending=[False, True]).head(top_n).index.tolist()
                    new_top = after.loc[date].dropna(subset=[col]).sort_values([col, 'ts_code'], ascending=[False, True]).head(top_n).index.tolist()
                    if old_top != new_top:
                        selected_changes.append({'top_n': top_n, 'date': str(date.date()), 'before': old_top, 'after': new_top})
        checks[name] = {'old_rows': len(before), 'new_rows': len(after), 'common_rows': len(common), 'common_values_exact': not bool(changed.any().any()), 'changed_columns': differences,
                        'added_rows': len(added), 'removed_rows': len(removed)}
        for date, count in pd.Series(1, index=added).groupby(level='date').sum().items():
            monthly.append({'artifact': name, 'date': date, 'added_rows': int(count)})
    bh = pd.read_parquet(old / 'execution/monthly_hold_allowed.parquet')
    ah = pd.read_parquet(new / 'execution/monthly_hold_allowed.parquet')
    pd.testing.assert_frame_equal(bh, ah, check_exact=True)
    checks['hold_all_dates_exact'] = True
    checks['common_date_top5_and_top20_order_changes'] = selected_changes
    checks['floating_difference_explanation'] = '补回缺失输入改变pandas滚动均值内部补偿求和状态；即使后续窗口不再含修复日，浮点尾差仍可传播。原值不舍入，rank保留真实执行结果。2022–23硬前缀仍逐值精确。'
    checks['source_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    output = new / 'input_coverage_audit'
    output.mkdir(exist_ok=True)
    pd.DataFrame(score_changes).to_csv(output / 'repair_common_score_changes.csv', index=False)
    pd.DataFrame(monthly).to_csv(output / 'repair_added_rows_by_month.csv', index=False)
    (output / 'repair_input_comparison.json').write_text(json.dumps(checks, ensure_ascii=False, indent=2))
    (output / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    print(json.dumps(checks, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--old-run', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    compare(args.old_run, args.run)

"""重放当前观察池的机会门槛；只诊断数量，不搜索收益或改变生产规则。"""
from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent / 'opportunity_audit'
sys.path.insert(0, str(ROOT / 'alpha_lab/src'))
from datahub import DataHub
from left_side_rules import stock_shape
from left_side_trend import trend_pullback


def main():
    """从冻结行情重算六项量价门槛，核对既有输出并记录单项移除诊断。"""
    cfg = yaml.safe_load((ROOT / 'alpha_lab/experiments/sector_live_v1/config_20261013.yaml').read_text())
    rules = yaml.safe_load((ROOT / cfg['stock_rule_config']).read_text())
    scan = ROOT / 'alpha_lab/outputs/sector_live_scan/20261013T053630_999606Z'
    candidates = ROOT / 'alpha_lab/outputs/sector_live_candidates/20261013T054152_793466Z'
    old = pd.read_csv(candidates / 'stock_diagnostics_all.csv').set_index('ts_code')
    assert old.index.is_unique
    hub = DataHub()
    assert hub.cfg.root.resolve() == (ROOT / cfg['release']).resolve()
    codes = sorted(old.index)
    dates = hub.calendar.cn_days(end=pd.Timestamp(cfg['asof']))[-cfg['stock_history_sessions']:]
    fields = ['hfq_close', 'turnover_rate', 'close', 'vol_share']
    raw = hub.cn.daily(codes=codes, start=str(dates[0].date()), end=cfg['asof'], fields=fields)
    wide = {f: raw.pivot(index='date', columns='ts_code', values=f).reindex(index=dates, columns=codes) for f in fields}
    stock = rules['stock']
    shape = stock_shape(wide, stock, stock)
    trend = trend_pullback(wide['hfq_close'], rules['trend'], rules['trend']['stock_minimum_advance'], rules['trend']['stock_minimum_pullback'])
    table = old.loc[codes, ['sector_code', 'sector_name', 'name']].copy()
    for key in ['expansion', 'contraction', 'slowdown', 'early_return', 'late_return']:
        table[key] = shape[key].iloc[-1]
    table['daily_return'] = wide['hfq_close'].pct_change(fill_method=None).iloc[-1]
    gates = pd.DataFrame({
        '早期三日下跌至少1%': table.early_return.le(-stock['minimum_early_fall']),
        '早期换手放大至少1.4倍': table.expansion.ge(stock['turnover_expansion']),
        '最近换手缩至0.7倍以下': table.contraction.le(stock['turnover_contraction']),
        '负收益幅度缩至0.65倍以下': table.slowdown.le(stock['slowdown_ratio']),
        '最近两日上涨不超2%': table.late_return.le(stock['maximum_recent_gain']),
        '当日跌幅不超2%': table.daily_return.ge(-stock['maximum_current_daily_fall']),
    })
    trend_ok = trend['passes'].iloc[-1].eq(1)
    qualified = (wide['close'].iloc[-1].gt(0) & wide['vol_share'].iloc[-1].gt(0)
                 & ~table['name'].str.upper().str.contains('ST|退', regex=True))
    np.testing.assert_array_equal(gates.all(axis=1), shape['shape'].iloc[-1])
    np.testing.assert_array_equal(gates.all(axis=1), old.loc[codes, 'shape_pass'])
    np.testing.assert_array_equal(trend_ok, old.loc[codes, 'trend_pass'])
    for key, saved in [('expansion', 'turnover_expansion'), ('contraction', 'turnover_contraction'), ('slowdown', 'loss_slowdown')]:
        np.testing.assert_allclose(table[key], old.loc[codes, saved], equal_nan=True)
    rows = []
    for key in gates:
        remaining = gates.drop(columns=key).all(axis=1)
        rows.append(dict(condition=key, passed_all=int(gates[key].sum()),
                         passed_with_trend=int((gates[key] & trend_ok).sum()),
                         shape_pass_if_removed=int(remaining.sum()),
                         stock_pass_if_removed=int((remaining & trend_ok & qualified).sum())))
    OUT.mkdir(exist_ok=True)
    pd.concat([table, gates, trend_ok.rename('trend_pass')], axis=1).to_csv(OUT / 'stock_gate_details.csv')
    pd.DataFrame(rows).to_csv(OUT / 'gate_counts.csv', index=False)
    events = pd.read_parquet(scan / 'all_confirmations.parquet')
    assert events.merged_event_id.is_unique
    events['year'] = pd.to_datetime(events.date).dt.year
    yearly = events.groupby('year').agg(events=('merged_event_id', 'size'), signal_dates=('date', 'nunique'))
    yearly.to_csv(OUT / 'event_counts_by_year.csv')
    watch = pd.read_csv(scan / 'watch_today.csv')
    watch[['sector_code', 'name', 'watch', 'trigger', 'support_recent', 'slowed', 'confirmed']].to_csv(OUT / 'sector_gates.csv', index=False)
    result = dict(asof=cfg['asof'], stocks=len(codes), qualified=int(qualified.sum()),
                  trend_pass=int(trend_ok.sum()), shape_pass=int(gates.all(axis=1).sum()),
                  maximum_shape_pass_after_one_gate_removed=max(r['shape_pass_if_removed'] for r in rows),
                  sector_watch=len(watch), sector_trigger=int(watch.trigger.sum()), sector_confirmed=int(watch.confirmed.sum()),
                  events=len(events), unique_signal_dates=int(events.date.nunique()),
                  latest_diagnostics_reproduced=True,
                  scope='仅2026-10-12观察池条件诊断；当前成员不用于历史选股回测；未检验修改后收益。')
    (OUT / 'summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(result, ensure_ascii=False), flush=True)
    print(pd.DataFrame(rows).to_string(index=False), flush=True)
    print(yearly.to_string(), flush=True)


if __name__ == '__main__':
    main()

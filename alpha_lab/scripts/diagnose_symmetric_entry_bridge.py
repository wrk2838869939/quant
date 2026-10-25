#!/usr/bin/env python3
"""事后解释桥接：同一2022—2023窗口的60日机会与实际入场信号代理。"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from trend_entry import marked_paths


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_manifest(out, manifest):
    manifest['output_hashes'] = {p.name: sha(p) for p in sorted(out.iterdir())
        if p.is_file() and p.name != 'manifest.json'}
    (out/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str))


def metrics(frame):
    """缺入场原标签为现金0；不成熟为NaN，保留各自计数。"""
    valid = frame[frame.mature]
    return dict(n=len(frame), mature_n=len(valid), immature_n=int((~frame.mature).sum()),
        no_entry_n=int(valid.entry_missing.fillna(False).sum()),
        exit_marked_n=int(valid.exit_marked.fillna(False).sum()),
        missing_mark_paths_n=int(valid.missing_marks.gt(0).sum()),
        missing_mark_observations=int(valid.missing_marks.sum()),
        delisted_n=int(valid.delisted.fillna(False).sum()),
        ret_60_mean=valid.ret.mean(), mae_60_mean=valid.mae.mean(),
        predicted_mean=valid.predicted.mean() if 'predicted' in valid else np.nan,
        optimism=(valid.predicted-valid.ret).mean() if 'predicted' in valid else np.nan)


def prepare(run, cfg, out):
    """全机会先标记再按冻结分数选Top5，不能按未来可观测性过滤候选。"""
    prediction_path = run/'execution_predictions.parquet'
    prediction = pd.read_parquet(prediction_path)
    if prediction.duplicated(['date','ts_code']).any() or not prediction.date.between('2022-01-01','2023-12-31').all():
        raise ValueError('预测日期或唯一性违规')
    path_config = (ROOT/cfg['source_panel']).parent/'config.yaml'
    original_cfg = yaml.safe_load(path_config.read_text())
    h = cfg['forecast']['horizon']
    if h != original_cfg['primary_horizon'] or h != 60:
        raise ValueError('桥接必须沿用原60日定义')
    columns = ['date','ts_code','hfq_open','hfq_close']
    filters = [('date','>=',pd.Timestamp('2022-01-01')),('date','<=',pd.Timestamp('2023-12-31'))]
    raw = pd.read_parquet(ROOT/'data/panel/daily_panel.parquet', columns=columns, filters=filters)
    if not raw.date.between('2022-01-01','2023-12-31').all():
        raise ValueError('行情过滤越界')
    input_hash = hashlib.sha256(pd.util.hash_pandas_object(raw,index=False).values.tobytes()).hexdigest()
    raw_count = len(raw)
    opening = raw.pivot(index='date',columns='ts_code',values='hfq_open').sort_index()
    closing = raw.pivot(index='date',columns='ts_code',values='hfq_close').reindex_like(opening)
    calendar = opening.index
    pd.DataFrame({'date':calendar}).to_csv(out/'market_calendar.csv',index=False)
    basic_path = ROOT/'data/raw/stock_basic_all.parquet'
    basic = pd.read_parquet(basic_path, columns=['ts_code','delist_date']).set_index('ts_code')
    delist = pd.to_datetime(basic.delist_date)
    # 截止后的元数据不参与任何路径，未发生退市的offset设为无穷。
    delist = delist.where(delist.le(calendar[-1]))
    blocks = []
    for date, group in prediction.groupby('date', sort=True):
        i = calendar.get_loc(date)
        codes = group.ts_code
        frame = group[['date','ts_code','phase','trend']].copy()
        frame['mature'] = i+h+1 < len(calendar)
        frame['label_end'] = pd.NaT
        frame['label_entry_date'] = calendar[i+1] if i+1 < len(calendar) else pd.NaT
        for key in ['ret','mae','mfe','mdd','giveback','late_return','sustained','roundtrip',
                    'state','entry_missing','missing_marks','exit_marked','delisted']:
            frame[key] = np.nan
        if i+h+1 < len(calendar):
            d = delist.reindex(codes).fillna(pd.Timestamp.max).to_numpy()
            offsets = np.searchsorted(calendar.to_numpy(), d, side='left')-i
            result = marked_paths(opening.reindex(columns=codes).iloc[i+1].to_numpy(),
                closing.reindex(columns=codes).iloc[i+1:i+h+1].to_numpy(),
                opening.reindex(columns=codes).iloc[i+h+1].to_numpy(), offsets, original_cfg['path_labels'])
            for key in result:
                frame[key] = result[key].to_numpy()
            frame['label_end'] = calendar[i+h+1]
        blocks.append(frame)
    labels = pd.concat(blocks,ignore_index=True)
    labels.to_parquet(out/'execution_signal_labels_60.parquet',index=False)
    merged = prediction.merge(labels.drop(columns=['phase','trend']),on=['date','ts_code'],validate='one_to_one')
    names = cfg['policies'] + cfg['execution_transfer_policies']
    selected_rows, date_rows, pool_rows = [], [], []
    for date, group in merged.groupby('date',sort=True):
        trend_pool = group[group.trend]
        pool_mean = trend_pool.loc[trend_pool.mature,'ret'].mean()
        phase_means = trend_pool[trend_pool.mature].groupby('phase').ret.mean()
        for phase, part in [('all',trend_pool),*list(trend_pool.groupby('phase'))]:
            pool_rows.append(dict(date=date, year=date.year,phase=phase,**metrics(part)))
        for name in names:
            admitted = group[group[name+'__score'].notna()]
            chosen = admitted.sort_values([name+'__score','ts_code'],ascending=[False,True]).head(5).copy()
            chosen['policy'] = name
            chosen['rank'] = np.arange(1,len(chosen)+1)
            chosen['predicted'] = chosen[name+'__expected_return']
            chosen['priority'] = chosen[name+'__score']
            chosen['trend_pool_return'] = pool_mean
            chosen['phase_pool_return'] = chosen.phase.map(phase_means)
            selected_rows.append(chosen[['date','ts_code','phase','policy','rank','priority','predicted',
                'mature','label_end','label_entry_date','ret','mae','entry_missing','exit_marked',
                'missing_marks','delisted','trend_pool_return','phase_pool_return']])
            row = dict(date=date,year=date.year,policy=name,admitted=len(admitted),
                maturity_available=bool(group.mature.all()),trend_pool_return=pool_mean,**metrics(chosen))
            row['between_phase'] = chosen.loc[chosen.mature,'phase_pool_return'].mean()-pool_mean
            row['within_phase'] = (chosen.loc[chosen.mature,'ret']-chosen.loc[chosen.mature,'phase_pool_return']).mean()
            date_rows.append(row)
    selected = pd.concat(selected_rows,ignore_index=True)
    dates = pd.DataFrame(date_rows)
    selected.to_csv(out/'monthly_top5_signal_labels.csv',index=False)
    dates.to_csv(out/'monthly_top5_dates.csv',index=False)
    pd.DataFrame(pool_rows).to_csv(out/'trend_pool_dates_phases.csv',index=False)
    summaries = []
    for (policy,year), group in dates.assign(year='all').groupby(['policy','year']):
        summaries.append(summarize_dates(group,policy,year))
    for (policy,year), group in dates.groupby(['policy','year']):
        summaries.append(summarize_dates(group,policy,year))
    summary = pd.DataFrame(summaries)
    summary.to_csv(out/'monthly_top5_summary.csv',index=False)
    old = pd.read_csv(run/'opportunities_summary.csv')
    old = old[old.k.eq(5)][['policy','realized','mae','predicted','optimism']].rename(
        columns={c:'development_2019_2021_'+c for c in ['realized','mae','predicted','optimism']})
    bridge = summary[summary.year.eq('all')].merge(old,on='policy',validate='one_to_one')
    bridge.to_csv(out/'same_horizon_window_bridge.csv',index=False)
    inputs = [prediction_path,path_config,basic_path,run/'config.yaml',run/'opportunities_summary.csv',
              ROOT/'src/trend_entry.py',ROOT/'scripts/audit_trend_entry.py',Path(__file__)]
    manifest = dict(status='labels_and_top5_complete',posthoc_not_selection=True,
        fresh_holdout=False,test_market_data_used=False,changes_to_registered_cases=False,
        no_training=True,no_new_trade_replay=True,horizon=h,path_spec=original_cfg['path_labels'],
        source_hashes={str(p.resolve()):sha(p) for p in inputs},market_columns=columns,
        market_filters=[[a,b,str(c)] for a,b,c in filters],filtered_market_rows=raw_count,
        filtered_market_sha256=input_hash,prediction_rows=len(prediction),review_dates=prediction.date.nunique(),
        immature_dates=int(labels.loc[~labels.mature,'date'].nunique()),
        immature_rows=int((~labels.mature).sum()),label_definition='signal T+1 adjusted open through T+61 adjusted open; missing marks carried; delist zero; no_entry cash zero',
        limitation='adjusted price proxy, not executable 60d trade or policy net return; 2019-2023 repeatedly inspected development')
    save_manifest(out,manifest)


def summarize_dates(group, policy, year):
    mature = group[group.maturity_available]
    return dict(policy=policy,year=year,total_dates=len(group),mature_dates=len(mature),
        immature_dates=int((~group.maturity_available).sum()),selected_n=int(group.n.sum()),
        mature_selected_n=int(group.mature_n.sum()),immature_selected_n=int(group.immature_n.sum()),
        no_entry_n=int(group.no_entry_n.sum()),exit_marked_n=int(group.exit_marked_n.sum()),
        missing_mark_paths_n=int(group.missing_mark_paths_n.sum()),
        missing_mark_observations=int(group.missing_mark_observations.sum()),
        delisted_n=int(group.delisted_n.sum()),date_equal_ret_60_mean=mature.ret_60_mean.mean(),
        date_equal_mae_60_mean=mature.mae_60_mean.mean(),date_equal_predicted_mean=mature.predicted_mean.mean(),
        date_equal_optimism=mature.optimism.mean(),trend_pool_return=mature.trend_pool_return.mean(),
        between_phase=mature.between_phase.mean(),within_phase=mature.within_phase.mean())


def finalize(run,out):
    """真实入场映射原信号标签，同时显式列示成交时滞和不成熟样本。"""
    manifest = json.loads((out/'manifest.json').read_text())
    for path, expected in manifest['source_hashes'].items():
        if Path(path).resolve() != Path(__file__).resolve() and sha(path) != expected:
            raise ValueError(f'桥接固定输入发生变化: {path}')
    if sha(out/'execution_signal_labels_60.parquet') != manifest['output_hashes']['execution_signal_labels_60.parquet']:
        raise ValueError('准备阶段冻结标签发生变化')
    labels = pd.read_parquet(out/'execution_signal_labels_60.parquet')
    calendar = pd.DatetimeIndex(pd.to_datetime(pd.read_csv(out/'market_calendar.csv').date))
    top = pd.read_csv(out/'monthly_top5_signal_labels.csv',parse_dates=['date'])
    cases = json.loads((run/'execution/manifest.json').read_text())['cases']
    missing = [c['name'] for c in cases if not (run/'execution'/c['name']/'summary.json').exists()]
    if missing:
        raise ValueError(f'等候完整登记案例输出，不选择已完成子集: {missing}')
    all_rows, summaries = [], []
    for case in cases:
        folder = run/'execution'/case['name']
        path = folder/'actual_entry_priorities.csv'
        actual = pd.read_csv(path,parse_dates=['date','signal_date'])
        actual = actual.rename(columns={'date':'actual_entry_date','signal_date':'date','expected_return':'predicted'})
        merged = actual.merge(labels,on=['date','ts_code'],how='left',validate='one_to_one',indicator=True)
        if not merged._merge.eq('both').all():
            raise ValueError('实际入场缺少原信号机会，禁止无声删除')
        merged['case'] = case['name']
        merged['capital'] = case['capital']
        merged['policy'] = case['policy']
        merged['actual_fill_lag_trading_intervals'] = [calendar.get_loc(fill)-calendar.get_loc(signal)
            for fill,signal in zip(merged.actual_entry_date,merged.date)]
        merged['fill_after_label_entry'] = merged.actual_entry_date.ne(merged.label_entry_date)
        merged['is_actual_trading_return'] = False
        same_top = top[top.policy.eq(case['policy']) & top.date.isin(merged.date)]
        keys = set(zip(top.loc[top.policy.eq(case['policy']),'date'],top.loc[top.policy.eq(case['policy']),'ts_code']))
        merged['in_monthly_paper_top5'] = [(d,c) in keys for d,c in zip(merged.date,merged.ts_code)]
        stats = metrics(merged)
        stats['actual_entry_60d_mean'] = stats.pop('ret_60_mean')
        summaries.append(dict(case=case['name'],capital=case['capital'],policy=case['policy'],
            **stats,actual_entry_dates=merged.date.nunique(),mature_entry_dates=merged.loc[merged.mature,'date'].nunique(),
            date_equal_signal_proxy_60d_mean=merged[merged.mature].groupby('date').ret.mean().mean(),
            same_actual_dates_paper_top5_60d_mean=same_top[same_top.mature].groupby('date').ret.mean().mean(),
            fill_after_label_entry_n=int(merged.fill_after_label_entry.sum()),
            max_signal_fill_lag=int(merged.actual_fill_lag_trading_intervals.max()) if len(merged) else 0,
            in_paper_top5_n=int(merged.in_monthly_paper_top5.sum()),
            actual_entry_60d_mean_is_signal_label_proxy=True))
        all_rows.append(merged.drop(columns=['_merge']))
        manifest['source_hashes'][str(path.resolve())] = sha(path)
    actual_summary = pd.DataFrame(summaries)
    actual_summary.to_csv(out/'actual_entry_signal_proxy_summary.csv',index=False)
    pd.concat(all_rows,ignore_index=True).to_csv(out/'actual_entry_signal_proxy_labels.csv',index=False)
    report = ('# 追加解释性入场桥接\n\n'
        '本分析在看到部分执行结果后追加（posthoc_not_selection），不修改已登记21案例、模型、排序或阈值。'
        '2019—2023均是已反复查看的开发窗口，不是新OOS。\n\n'
        '同一2022—2023预测按原score及股票代码取每月Top5；先保留候选再生成原60日marked_paths标签。'
        '起点为信号次日后复权开盘，终点为第61个交易日后复权开盘；终点缺价沿用旧标记，退市减记，'
        '缺入场保留为原现金零收益标签。末尾不成熟月仍列出，不用2024补足。'
        '同日期全部已知trend机会池与阶段均值另列，未重新匹配或训练。\n\n'
        'monthly_top5_summary以日期等权；same_horizon_window_bridge与原2019—2021同期限机会比较，'
        '只能描述窗口差异，不能把市场差异或训练截止变化归因给持有政策。\n\n'
        'actual_entry_60d_mean是按真实入场所对应signal_date+ts_code映射的60日信号标记代理，按入场次数等权，'
        '不是实际成交收益。另列按入场信号日期等权均值，以及仅相同入场日期的纸面Top5，'
        '帮助区分选股差异与不同时点的机会。实际买入滞后及是否偏离标签次日开盘逐笔记录；'
        '即使日期相同，后复权标记也不等于实际原价成交、费用、分红或持有政策净收益。\n\n'
        '覆盖、不成熟、no_entry、缺标记路径、终点沿用估值和退市计数均保留。'
        '机会窗口重叠、同股重复、多资本案例重复，不能视为独立样本；该桥接不能独立识别持有政策因果效果。\n')
    bridge = pd.read_csv(out/'same_horizon_window_bridge.csv')
    report += '\n## 同期限机会跨窗口结果\n\n'
    report += '| 方案 | 2019—2021 Top5 | 2022—2023 Top5 |\n|---|---:|---:|\n'
    for row in bridge.itertuples():
        report += (f'| {row.policy} | {row.development_2019_2021_realized:+.3%} | '
                   f'{row.date_equal_ret_60_mean:+.3%} |\n')
    report += (f'\n2022—2023共24个月，21个月成熟、期末3个月保留为未成熟；同期全趋势机会池日期等权均值'
               f'{bridge.trend_pool_return.iloc[0]:+.3%}。局部模型的改善在同样60日机会口径下已消失，'
               '因此不能把退化主要解释为更长持有政策造成；本比较仍同时改变市场窗口和训练截止，无法识别二者因果贡献。\n')
    report += '\n## 5万元主板真实入场信号代理\n\n'
    report += '| 方案 | 成熟／未成熟入场数 | 60日信号标记均值 |\n|---|---:|---:|\n'
    for row in actual_summary[actual_summary.case.str.startswith('base_50000_')].itertuples():
        report += f'| {row.policy} | {row.mature_n}／{row.immature_n} | {row.actual_entry_60d_mean:+.3%} |\n'
    report += (f'\n21案例按原信号映射后，偏离标签次日买入日期的入场共{actual_summary.fill_after_label_entry_n.sum()}次。'
               '真实入场是当月纸面Top5中在可用仓位时实际买入的子集，不是每个月将Top5全部重新买入；'
               '表中的均值仍是原信号价格标记代理，不能用来还原整个耐心持有账本。\n')
    (out/'REPORT.md').write_text(report)
    manifest.update(status='complete',actual_cases=len(cases),actual_signal_proxy_only=True)
    manifest['source_hashes'][str(Path(__file__).resolve())] = sha(__file__)
    save_manifest(out,manifest)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--prepare-only',action='store_true')
    args = parser.parse_args()
    run = args.run.resolve()
    out = run/'entry_bridge'
    out.mkdir(exist_ok=True)
    cfg = yaml.safe_load((run/'config.yaml').read_text())
    if not (out/'manifest.json').exists():
        prepare(run,cfg,out)
    if not args.prepare_only:
        finalize(run,out)
    print((out/'manifest.json').read_text(),flush=True)


if __name__ == '__main__':
    main()

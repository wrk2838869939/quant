#!/usr/bin/env python3
"""在当日板块观察池内应用既有个股规则，分别列出通过与未通过的证据。"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import sys
import json
import shutil
import numpy as np
import pandas as pd
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.backends.backend_pdf import PdfPages
HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(PROJECT / 'alpha_lab/src'))
from datahub import DataHub
from sector_dataset import SectorDataset, digest
from left_side_rules import stock_shape
from left_side_trend import trend_pullback
from update_market import query
from scan_sectors import save_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scan', type=Path, required=True)
    parser.add_argument('--membership-cache', type=Path)
    parser.add_argument('--config', type=Path, default=HERE / 'config.yaml')
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    rules = yaml.safe_load((PROJECT / cfg['stock_rule_config']).read_text())
    manifest = json.loads((args.scan / 'manifest.json').read_text())
    assert manifest['status'] == 'complete'
    for name, h in manifest['artifacts'].items():
        assert digest(args.scan / name) == h
    ds = SectorDataset(manifest['dataset'])
    hub = DataHub()
    assert hub.cfg.root.resolve() == (PROJECT / cfg['release']).resolve()
    asof = pd.Timestamp(cfg['asof'])
    out = PROJECT / 'alpha_lab/outputs/sector_live_candidates' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    out.mkdir(parents=True)
    save_json(out / 'manifest.json', dict(status='building'))
    print('OUTPUT', out, flush=True)
    for p in [Path(__file__), PROJECT / cfg['stock_rule_config']]:
        shutil.copy2(p, out / p.name)
    shutil.copy2(args.config, out / 'config.yaml')
    watch = pd.read_csv(args.scan / 'watch_today.csv')
    recent_path = args.scan / 'recent_confirmations_scored.csv'
    new = pd.read_csv(recent_path if recent_path.exists() else args.scan / 'new_confirmations_scored.csv')
    dates = hub.calendar.cn_days(end=asof)[-cfg['stock_history_sessions']:]
    latest_days = dates[-cfg['recent_signal_sessions']:]
    recent = new[pd.to_datetime(new.date).isin(latest_days)]
    recent.to_csv(out / 'recent_confirmations.csv', index=False)
    membership = pd.read_parquet(ds.root / 'membership_daily.parquet')
    sector_codes = json.loads((ds.root / 'sector_codes.json').read_text())
    members = {r.sector_code: membership.columns[membership.loc[asof].eq(sector_codes.index(r.sector_code))].tolist() for r in watch.itertuples()}
    frozen_members = {code: set(values) for code, values in members.items()}
    master = pd.read_parquet(hub.cfg.meta_dir / 'stock_basic.parquet').set_index('ts_code')
    membership_rows = []
    for r in watch.itertuples():
        # 在线成员只作当下交叉检查，不回填历史数据、不更改冻结指数。
        cache = args.membership_cache / f'current_members_{r.sector_code}.parquet' if args.membership_cache else None
        fresh = pd.read_parquet(cache) if cache and cache.exists() else query('index_member_all', cfg, l2_code=r.sector_code, is_new='Y')
        fresh.to_parquet(out / f'current_members_{r.sector_code}.parquet', index=False)
        # is_new=Y 仍可能包含已退市证券，必须结合当下主档和成员区间核验。
        if len(fresh):
            began = pd.to_datetime(fresh.in_date, format='%Y%m%d', errors='coerce')
            ended = pd.to_datetime(fresh.out_date, format='%Y%m%d', errors='coerce')
            active = fresh.ts_code.map(master.list_status).eq('L') & ~fresh['name'].astype(str).str.contains('退')
            active &= began.le(asof) & (ended.isna() | ended.gt(asof))
            current = set(fresh.loc[active, 'ts_code'])
        else:
            current = set()
        frozen = set(members[r.sector_code])
        # 当前新发现的成员可额外诊断，但不改写板块历史指数或模型输入。
        members[r.sector_code] = sorted(frozen | current)
        membership_rows.append(dict(sector_code=r.sector_code, name=r.name, frozen_members=len(frozen),
            current_raw_members=len(fresh), current_active_members=len(current), overlap=len(frozen & current), added='|'.join(sorted(current - frozen)),
            absent='|'.join(sorted(frozen - current))))
    pd.DataFrame(membership_rows).to_csv(out / 'membership_crosscheck.csv', index=False)
    codes = sorted({code for values in members.values() for code in values})
    rows = []
    if codes:
        fields = ['close', 'hfq_close', 'hfq_low', 'hfq_high', 'hfq_open', 'turnover_rate', 'amount_yuan', 'vol_share', 'total_mv_yuan', 'adj_factor']
        raw = hub.cn.daily(codes=codes, start=str(dates[0].date()), end=cfg['asof'], fields=fields)
        wide = {f: raw.pivot(index='date', columns='ts_code', values=f).reindex(index=dates, columns=codes) for f in fields}
        shape = stock_shape(wide, rules['stock'], rules['stock'])
        trend = trend_pullback(wide['hfq_close'], rules['trend'], rules['trend']['stock_minimum_advance'], rules['trend']['stock_minimum_pullback'])
        master = pd.read_parquet(hub.cfg.meta_dir / 'stock_basic.parquet').set_index('ts_code')
        five_return = wide['hfq_close'] / wide['hfq_close'].shift(rules['stock']['shape_window']) - 1
        low = wide['hfq_low'].rolling(rules['exit']['support_window']).min().iloc[-1] / wide['adj_factor'].iloc[-1]
        for sector in watch.itertuples():
            securities = members[sector.sector_code]
            start, peak = pd.Timestamp(sector.platform_end), pd.Timestamp(sector.peak_date)
            returns = wide['hfq_close'].loc[peak, securities] / wide['hfq_close'].loc[start, securities] - 1 if start in dates and peak in dates else pd.Series(np.nan, index=securities)
            rank = returns.rank(pct=True)
            live_signal = recent[recent.sector_code.eq(sector.sector_code)]
            model_pass = int(live_signal.model_pass_count.max()) if len(live_signal) and 'model_pass_count' in live_signal else 0
            signal_date = live_signal.date.max() if len(live_signal) else None
            for code in securities:
                name = master.at[code, 'name'] if code in master.index else code
                price = wide['close'].iloc[-1][code]
                liquid = bool(pd.notna(price) and price > 0 and wide['vol_share'].iloc[-1][code] > 0)
                normal_name = not ('ST' in str(name).upper() or '退' in str(name))
                shape_ok = bool(shape['shape'].iloc[-1][code])
                trend_ok = bool(trend['passes'].iloc[-1][code] == 1)
                current_sector = bool(sector.confirmed)
                ready = bool(liquid and normal_name and shape_ok and trend_ok and current_sector and model_pass > 0)
                blockers = []
                for okay, why in [(liquid, '缺当日可交易报价'), (normal_name, '名称含ST或退市标识'),
                    (shape_ok, '五日量价形态未通过'), (trend_ok, '上涨趋势及先涨后回落未通过'),
                    (current_sector, '板块今天未确认'), (model_pass > 0, '近五日无模型通过的新确认')]:
                    if not okay:
                        blockers.append(why)
                rows.append(dict(sector_code=sector.sector_code, sector_name=sector.name, ts_code=code, name=name,
                    membership_source='frozen_history' if code in frozen_members[sector.sector_code] else 'current_snapshot_supplement',
                    close=price, total_mv_yuan=wide['total_mv_yuan'].iloc[-1][code], turnover_today=wide['turnover_rate'].iloc[-1][code],
                    return_5d=five_return.iloc[-1][code], leader_rally_return=returns.get(code), leader_rank=rank.get(code),
                    turnover_expansion=shape['expansion'].iloc[-1][code], turnover_contraction=shape['contraction'].iloc[-1][code],
                    loss_slowdown=shape['slowdown'].iloc[-1][code], shape_pass=shape_ok, trend_pass=trend_ok,
                    stock_rules_pass=bool(liquid and normal_name and shape_ok and trend_ok),
                    sector_confirmed=current_sector, model_pass_count=model_pass, recent_signal_date=signal_date,
                    recent_low_reference=low[code], candidate=ready, blockers='；'.join(blockers)))
    stocks = pd.DataFrame(rows)
    if not stocks.empty:
        stocks = stocks.sort_values(['candidate', 'stock_rules_pass', 'leader_rank'], ascending=False)
        candidates = stocks[stocks.candidate]
    else:
        candidates = stocks.copy()
    stocks.to_csv(out / 'stock_diagnostics_all.csv', index=False)
    candidates.to_csv(out / 'stock_candidates.csv', index=False)
    summary = dict(requested_date=cfg.get('requested_date', cfg['asof']), asof=cfg['asof'], watched_sectors=len(watch), stocks_checked=len(stocks),
        recent_confirmations=len(recent), exploratory_stock_rules_pass=int(stocks.stock_rules_pass.sum()) if len(stocks) else 0,
        combined_candidates=len(candidates), stock_rules_note='复用旧个股量价/趋势规则，仅探索性检查，尚无本次模型下的选股回测验证')
    save_json(out / 'summary.json', summary)
    font_manager.fontManager.addfont(str(PROJECT / cfg['font']))
    plt.rcParams.update({'font.family': font_manager.FontProperties(fname=PROJECT / cfg['font']).get_name(), 'axes.unicode_minus': False, 'pdf.fonttype': 42})
    panel = pd.read_parquet(ds.root / 'daily.parquet')
    if len(watch):
        with PdfPages(out / 'sector_watch_charts.pdf') as book:
            for r in watch.head(cfg['maximum_review_cases']).itertuples():
                f = panel[panel.sector_code.eq(r.sector_code) & panel.segment_id.eq(r.segment_id) & panel.index_valid & panel.date.ge(pd.Timestamp(r.platform_start))]
                fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True, layout='constrained')
                axes[0].plot(f.date, f.index_close)
                axes[0].axhspan(r.platform_low, r.reference_high, alpha=.2, color='#dba969', label='同轮平台至上涨初期价格参照')
                axes[0].axhline(r.support_at_peak, ls=':', color='purple', label='峰前支撑')
                axes[0].set(title=f'{r.name} {r.sector_code} · 截至 {cfg["asof"]} · 今日确认：{"是" if r.confirmed else "否"}', ylabel='板块点位')
                axes[0].legend()
                axes[1].bar(f.date, f.turnover_pct, color='#367aa3')
                axes[1].set(ylabel='板块换手率（%）')
                # 日期轴与横向参照带混合时，显式固定时间范围，避免自动缩放拉回 1970 年。
                for ax in axes:
                    ax.set_xlim(f.date.min() - pd.Timedelta(days=1), f.date.max() + pd.Timedelta(days=1))
                fig.autofmt_xdate(rotation=25)
                book.savefig(fig); plt.close(fig)
    lines = ['# 最新板块与个股扫描', '', f'请求日期：{cfg.get("requested_date", cfg["asof"])}；当日日线未完成时使用最近完整交易日。', '', f'截至 {cfg["asof"]} 收盘。只更新数据，事件门槛及两个网络模型/评分阈值均未调整。', '',
        '## 结果', '', json.dumps(summary, ensure_ascii=False, indent=2), '',
        '## 近期确认事件', '', recent.to_csv(index=False), '',
        '## 今日板块观察池', '', watch[['name','sector_code','watch','trigger','confirmed','rally_return','drawdown','reference_distance']].to_csv(index=False), '',
        '## 个股通过情况', '', candidates.to_csv(index=False), '',
        '个股尚未经过本次板块模型下的选股回测。本次只复用此前五日放量下跌后缩量减速、峰值前上行趋势和先涨后回落规则做探索性检查；历史龙头表现只用于排序。',
        '候选还要求板块今天保持确认，近五个交易日出现新确认且至少一个既有模型通过；这项实时联合使用方式尚未单独回测，不能套用历史事件平均收益。',
        '未通过个股不作为买入候选；stock_diagnostics_all.csv 保留每项原因。所有信息是收盘后可知，下一交易日开盘前仍需核对报价与成交条件。',
        '历史行业分类沿用冻结补充资料；对观察板块另外在线核对当前成员，差异记录在 membership_crosscheck.csv，不将当前名单回填历史。',
        '本次没有下单，也没有据此建立10万元组合或验证账户最大回撤30%。','']
    (out / 'README.md').write_text('\n'.join(lines))
    save_json(out / 'manifest.json', dict(status='complete', sector_scan=str(args.scan), config=cfg, summary=summary,
        artifacts={p.name: digest(p) for p in out.iterdir() if p.is_file() and p.name != 'manifest.json'}))
    (HERE / 'LATEST_CANDIDATES').write_text(str(out.relative_to(PROJECT)) + '\n')
    print('COMPLETE', out, json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()

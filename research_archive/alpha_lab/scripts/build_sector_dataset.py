# 历史实验脚本归档：运行依赖原研究环境及未发布输入。
#!/usr/bin/env python3
"""从冻结全市场行情与独立行业补充表构建板块数据版本。"""
from datetime import datetime, timezone
from pathlib import Path
import argparse
import json
import os
import shutil
import sys
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
sys.path.insert(0, str(ROOT / 'src'))
from datahub import DataHub
from sector_dataset import FIELDS, digest, build_membership, make_wide, aggregate


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/sector_dataset_v1.yaml')
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    if os.environ.get('DATAHUB_CONTENT_ID') != cfg['content_id']:
        raise RuntimeError('必须通过 frozen_research.py run 使用指定冻结版本')
    hub = DataHub()
    if hub.cfg.root.resolve() != (PROJECT / cfg['release']).resolve():
        raise RuntimeError('行情目录不符')
    supplement = PROJECT / cfg['supplement']
    source_manifest = json.loads((supplement / 'manifest.json').read_text())
    for name, expected in source_manifest['hashes'].items():
        if digest(supplement / name) != expected:
            raise RuntimeError(f'行业源发生变化: {name}')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    out = PROJECT / cfg['output_root'] / ('sw_l2_v1_' + stamp)
    out.mkdir(parents=True, exist_ok=False)
    print(f'OUTPUT {out}', flush=True)
    save_json(out / 'manifest.json', dict(status='building', config=cfg))
    source_dir = out / 'sources'
    source_dir.mkdir()
    for p in [args.config, Path(__file__), ROOT / 'src/sector_dataset.py', ROOT / 'tests/test_sector_dataset.py']:
        shutil.copy2(p, source_dir / p.name)
    catalog = pd.read_parquet(supplement / 'catalog.parquet')
    members = pd.read_parquet(supplement / 'members.parquet')
    basic = pd.read_parquet(hub.cfg.meta_dir / 'stock_basic.parquet')
    catalog.to_parquet(out / 'catalog_sw_l2.parquet', index=False)
    members.to_parquet(out / 'membership_intervals.parquet', index=False)
    dc = pd.read_parquet(PROJECT / cfg['dc_catalog'])
    dc[['ts_code', 'trade_date', 'name', 'idx_type', 'level']].to_parquet(out / 'catalog_dc.parquet', index=False)
    shutil.copy2(supplement / 'requests.json', out / 'membership_requests.json')
    print('LOAD_FULL_MARKET', flush=True)
    raw = hub.cn.daily(start=cfg['start'], end=cfg['end'], fields=FIELDS)
    dates = hub.calendar.cn_days(cfg['start'], cfg['end'])
    codes = sorted(set(raw.ts_code) | set(members.con_code))
    mapping, sectors, active, life = build_membership(members, catalog, dates, codes, basic, cfg)
    wide = make_wide(raw, dates, codes)
    raw_rows, quote_stocks = len(raw), raw.ts_code.nunique()
    del raw
    observed = np.isfinite(wide['hfq_close']) & (wide['hfq_close'] > 0)
    excluded = observed & ~life
    for values in wide.values():
        values[~life] = np.nan
    usable = observed & life
    coverage = pd.DataFrame(dict(date=dates, quoted_stocks=observed.sum(axis=1),
                                outside_listing_life=excluded.sum(axis=1),
                                usable_quoted_stocks=usable.sum(axis=1),
                                assigned=((mapping >= 0) & usable).sum(axis=1),
                                missing=((mapping == -1) & usable).sum(axis=1),
                                conflicting=((mapping == -2) & usable).sum(axis=1)))
    coverage['assignment_fraction'] = coverage.assigned / coverage.usable_quoted_stocks
    coverage.to_parquet(out / 'coverage_daily.parquet', index=False)
    coverage.groupby(coverage.date.dt.year).agg(
        trading_days=('date', 'size'), mean_stocks=('usable_quoted_stocks', 'mean'),
        assignment_mean=('assignment_fraction', 'mean'), assignment_min=('assignment_fraction', 'min'),
        conflict_stock_days=('conflicting', 'sum'), excluded_stock_days=('outside_listing_life', 'sum')
    ).to_csv(out / 'coverage_yearly.csv')
    pd.DataFrame(mapping, index=dates, columns=codes).to_parquet(out / 'membership_daily.parquet')
    save_json(out / 'sector_codes.json', sectors)
    daily = aggregate(wide, dates, mapping, sectors, active, cfg, progress=True)
    daily['member_request_failed'] = daily.sector_code.isin(source_manifest['failed_sectors'])
    daily.to_parquet(out / 'daily.parquet', index=False)
    daily.groupby(['classification_version', 'sector_code']).agg(
        days=('date', 'size'), valid_days=('index_valid', 'sum'),
        mean_members=('known_members', 'mean'), mean_index_coverage=('index_member_coverage', 'mean'),
        segments=('segment_id', 'max'), request_failed=('member_request_failed', 'max')
    ).to_csv(out / 'coverage_sectors.csv')
    print('VERIFY', flush=True)
    assert not daily.duplicated(['date', 'sector_code']).any()
    good = daily[daily.index_valid]
    assert (good.max_weight <= cfg['maximum_member_weight'] + 1e-10).all()
    assert (good.index_high_bound + 1e-8 >= good[['index_open', 'index_close']].max(axis=1)).all()
    assert (good.index_low_bound - 1e-8 <= good[['index_open', 'index_close']].min(axis=1)).all()
    delta = (daily.matched_cap_change_yuan.fillna(0) + daily.member_add_cap_yuan -
             daily.member_remove_cap_yuan + daily.missing_cap_effect_yuan)
    valid_delta = daily.observed_cap_change_yuan.notna()
    np.testing.assert_allclose(delta[valid_delta], daily.observed_cap_change_yuan[valid_delta], rtol=1e-8, atol=.05)
    assert (daily.breadth_members + daily.breadth_unknown == daily.known_members).all()
    assert (coverage.assigned + coverage.missing + coverage.conflicting == coverage.usable_quoted_stocks).all()
    # 对物理截断的行情前缀重新读取、建宽表及归属，验证后续数据不会改变早期结果。
    cut = pd.Timestamp(cfg['prefix_check_end'])
    short_dates = dates[dates <= cut]
    short_raw = hub.cn.daily(start=cfg['start'], end=str(cut.date()), fields=FIELDS)
    short_map, short_sectors, short_active, short_life = build_membership(members, catalog, short_dates, codes, basic, cfg)
    short_wide = make_wide(short_raw, short_dates, codes)
    for x in short_wide.values():
        x[~short_life] = np.nan
    prefix = aggregate(short_wide, short_dates, short_map, short_sectors, short_active, cfg)
    pd.testing.assert_frame_equal(prefix.reset_index(drop=True),
        daily.loc[daily.date <= cut, prefix.columns].reset_index(drop=True), check_exact=True)
    # 同一连续段的指数点位必须复现收益，重启日不许跨段计算。
    ordered = daily.sort_values(['sector_code', 'date'])
    chain = ordered.groupby(['sector_code', 'segment_id']).index_close.pct_change(fill_method=None)
    selected = ordered.chain_return.notna()
    np.testing.assert_allclose(chain[selected], ordered.loc[selected, 'chain_return'], rtol=1e-10, atol=1e-12)
    summary = dict(start=str(dates.min().date()), end=str(dates.max().date()), trading_days=len(dates),
                   raw_stock_days=raw_rows, quote_stocks=int(quote_stocks), sector_codes=len(sectors),
                   sectors_with_members=int(daily.loc[daily.known_members.gt(0), 'sector_code'].nunique()),
                   sectors_with_valid_index=int(good.sector_code.nunique()), daily_rows=len(daily),
                   valid_index_rows=len(good), latest_valid_sectors=int(good.date.eq(dates[-1]).sum()),
                   mean_assignment=float(coverage.assignment_fraction.mean()),
                   latest_assignment=float(coverage.assignment_fraction.iloc[-1]),
                   conflict_stock_days=int(coverage.conflicting.sum()),
                   excluded_before_listing_or_after_delisting=int(excluded.sum()),
                   stocks_without_basic=int(len(set(codes) - set(basic.ts_code))),
                   dc_catalog_rows=len(dc), dc_levels=dc.level.value_counts().to_dict(),
                   dc_membership_dataset_built=False, historical_membership_certified=False,
                   failed_member_requests=len(source_manifest['failed_sectors']))
    save_json(out / 'summary.json', summary)
    save_json(out / 'validation.json', dict(unique_keys=True, weight_caps=True, price_bounds=True,
              cap_decomposition=True, breadth_counts=True, coverage_counts=True, index_chain=True,
              physical_prefix_exact=True, prefix_end=str(cut.date())))
    source_paths = [PROJECT / cfg['release'] / 'snapshot_manifest.json', supplement / 'manifest.json',
                    supplement / 'catalog.parquet', supplement / 'members.parquet', PROJECT / cfg['dc_catalog']]
    artifacts = {str(p.relative_to(out)): digest(p) for p in sorted(out.rglob('*'))
                 if p.is_file() and p.name != 'manifest.json'}
    content_id = hashlib_content(artifacts)
    save_json(out / 'manifest.json', dict(status='complete', dataset_id=out.name, content_id=content_id,
              created_at=datetime.now(timezone.utc).isoformat(), market_content_id=cfg['content_id'],
              config=cfg, source_hashes={str(p.relative_to(PROJECT)): digest(p) for p in source_paths},
              verified_supplement_files=len(source_manifest['hashes']), artifacts=artifacts,
              summary=summary, limitations=['历史成员来源未获完整性或发布时间 PIT 认证',
              '缺失行情不向前填充，代理指数按有效成员重归一化',
              '高低价为非同步分量边界，不能用于盘中止盈止损',
              '市值变化包含价格、股本变化，不能直接解释为估值变化',
              '东方财富仅保存目录，未构建其完整历史成员板块面板']))
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    print(f'COMPLETE {out}', flush=True)


def hashlib_content(artifacts):
    import hashlib
    return hashlib.sha256(json.dumps(artifacts, sort_keys=True).encode()).hexdigest()


if __name__ == '__main__':
    main()

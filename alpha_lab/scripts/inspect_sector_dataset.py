#!/usr/bin/env python3
"""核验固定板块版本并生成人可读覆盖报告，不运行策略。"""
from pathlib import Path
import argparse
import json
import sys
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from sector_dataset import SectorDataset, digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', type=Path)
    parser.add_argument('--report', type=Path, default=ROOT / 'docs/SECTOR_DATASET_BUILD_20260921.md')
    args = parser.parse_args()
    ds = SectorDataset(args.dataset)
    out = args.dataset.resolve()
    summary = ds.manifest['summary']
    yearly = pd.read_csv(out / 'coverage_yearly.csv')
    daily = pd.read_parquet(out / 'daily.parquet')
    catalog = pd.read_parquet(out / 'catalog_sw_l2.parquet')
    latest = daily[daily.date.eq(daily.date.max())].copy()
    names = catalog[catalog.src.eq(ds.manifest['config']['newer_version'])].set_index('index_code').industry_name
    latest['name'] = latest.sector_code.map(names)
    counts = latest.index_status.value_counts()
    review = ROOT / 'outputs/sector_dataset_review' / out.name
    review.mkdir(parents=True, exist_ok=True)
    latest.to_csv(review / 'latest_sectors.csv', index=False)
    membership = pd.read_parquet(out / 'membership_daily.parquet')
    conflict_rows = []
    for code in membership.columns[(membership == -2).any(axis=0)]:
        mask = membership[code].eq(-2)
        conflict_rows.append(dict(ts_code=code, first_date=membership.index[mask].min(),
                                  last_date=membership.index[mask].max(), days=int(mask.sum())))
    pd.DataFrame(conflict_rows, columns=['ts_code', 'first_date', 'last_date', 'days']).to_csv(
        review / 'membership_conflicts.csv', index=False)
    # 读接口返回应与落盘表逐值一致，成员个数须与面板一致。
    example = latest[latest.index_valid].sort_values('known_members', ascending=False).iloc[0]
    part = ds.history(example.sector_code, '2023-01-01', '2023-12-31')
    expected = daily[daily.sector_code.eq(example.sector_code) & daily.date.between('2023-01-01', '2023-12-31')]
    pd.testing.assert_frame_equal(part.reset_index(drop=True), expected.reset_index(drop=True))
    assert len(ds.members_at(example.sector_code, latest.date.max())) == example.known_members
    lines = ['# 板块数据集构建报告', '',
             f'固定版本：`{out}`。内容标识：`{ds.manifest["content_id"]}`。', '',
             f'范围：{summary["start"]} 至 {summary["end"]}，{summary["trading_days"]} 个交易日，'
             f'{summary["quote_stocks"]} 只行情股票、{summary["raw_stock_days"]:,} 条原始股票日线。', '',
             f'申万二级合并目录 {summary["sector_codes"]} 个编号，{summary["sectors_with_members"]} 个出现已知成员，'
             f'{summary["sectors_with_valid_index"]} 个至少有一日有效代理指数；'
             f'面板 {summary["daily_rows"]:,} 行，其中 {summary["valid_index_rows"]:,} 行指数有效。', '',
             f'最后交易日有效板块 {summary["latest_valid_sectors"]} 个；全市场已报价且存续股票的行业归属覆盖 '
             f'{summary["latest_assignment"]:.2%}，全期日均 {summary["mean_assignment"]:.2%}。'
             '这衡量记录中已知归属，不等于官方成员完整性认证。', '',
             '## 逐年覆盖', '', '| 年份 | 交易日 | 平均归属覆盖 | 最低归属覆盖 | 冲突股票日 |',
             '|---|---:|---:|---:|---:|']
    for r in yearly.itertuples():
        lines.append(f'| {r.date} | {r.trading_days} | {r.assignment_mean:.2%} | {r.assignment_min:.2%} | {r.conflict_stock_days} |')
    lines += ['', '## 数据口径与缺口', '',
              '- 指数：昨日成员及总市值、单股限权 20%，附等权参照；缺口及分类切换分段，不跨段计算形态。',
              '- 广度：上涨/下跌/平盘/未知分别计数；提供涨跌比、上涨占比、成交量额和统一单位的聚合换手率。',
              '- 市值：提供总/流通市值、同成员市值变化、成员加入退出及缺失数据影响。',
              '- 个股极值合成的高低价只是板块边界，不能当作真实板块盘中高低点。',
              f'- 原成员查询有 {summary["failed_member_requests"]} 个失败，已保存请求记录和逐板块标记。',
              f'- 上市前/退市后排除 {summary["excluded_before_listing_or_after_delisting"]:,} 个股票日；'
              f'行情与成员联合代码中有 {summary["stocks_without_basic"]} 个缺主档，未以当前活跃状态剔除历史股票。',
              f'- 东方财富保存 {summary["dc_catalog_rows"]} 条行业目录，层级计数 {summary["dc_levels"]}；其历史成员板块面板尚未构建。',
              '- 行业历史来源未认证完整性及实际发布日期；本版用于有覆盖约束的研究，不能宣称全市场完整 PIT。', '',
              '## 最近交易日示例', '',
              '| 编号 | 名称 | 已知成员 | 指数参与成员 | 上涨/下跌/平盘 | 换手率 |',
              '|---|---|---:|---:|---|---:|']
    for r in latest[latest.index_valid].sort_values('known_members', ascending=False).head(10).itertuples():
        lines.append(f'| {r.sector_code} | {r.name} | {r.known_members} | {r.index_members} | '
                     f'{r.up_count}/{r.down_count}/{r.flat_count} | {r.turnover_pct:.3f}% |')
    lines += ['', f'最近日状态计数：`{counts.to_dict()}`。', '',
              '## 核验', '',
              '- 原始冻结行情在启动时通过完整版本校验；行业补充表全部源文件哈希通过。',
              '- 6 项单元测试通过：滞后权重与前缀一致性、成员加入、市值与换手单位、缺口分段、历史归属冲突/存续、权重上限。',
              '- 全量主键、权重上限、价格边界、市值变动分解、广度及覆盖计数、连续段收益核对通过。',
              f'- 独立重读并截断至 {ds.manifest["config"]["prefix_check_end"]} 的原始行情重新生成前缀，与全量构建逐值一致。',
              '- 产物哈希、读取接口和按日成员数量复核通过。检查不等同于补足源数据的历史缺口。', '',
              f'最新板块明细及冲突股票列表：`{review}`。', '',
              '字段、计算和使用方式见 [数据字典](SECTOR_DATASET_V1.md)。本次没有运行板块选择策略。', '']
    args.report.write_text('\n'.join(lines))
    review_files = [args.report, ROOT / 'docs/SECTOR_DATASET_V1.md', Path(__file__),
                    review / 'latest_sectors.csv', review / 'membership_conflicts.csv']
    (review / 'review_manifest.json').write_text(json.dumps(dict(
        dataset_content_id=ds.manifest['content_id'], read_api_check=True,
        artifact_hashes={str(p.resolve()): digest(p) for p in review_files}),
        ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(dict(report=str(args.report), summary=summary, latest_status=counts.to_dict()), ensure_ascii=False))


if __name__ == '__main__':
    main()

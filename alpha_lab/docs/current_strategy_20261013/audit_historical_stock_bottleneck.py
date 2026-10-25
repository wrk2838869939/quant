"""复核板块确认后五日内的个股筛选瓶颈；不计算收益或实际成交。"""
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / 'outputs/sector_entry_window_v1/20261013T064932_361598Z'
SCORES = ROOT / 'outputs/sector_fuzzy_random_halves/20260923T111853_561924Z/all_role_scores_before_evaluation.parquet'
CATALOG = ROOT / 'data/sector_live_datasets/sw_l2_v1_20261013T053143_382413Z/catalog_sw_l2.parquet'
OUT = Path(__file__).parent / 'historical_stock_bottleneck'


def main():
    """以独立板块事件为单位，统计五日内是否存在完整个股候选。"""
    events = pd.read_parquet(RUN / 'event_scores.parquet')
    versions = pd.read_parquet(RUN / 'events.parquet')
    versions = versions[['merged_event_id', 'classification_version']].rename(columns={'merged_event_id': 'event_id'})
    events = events.merge(versions, on='event_id', validate='one_to_one')
    catalog = pd.read_parquet(CATALOG).rename(columns={'index_code': 'sector_code', 'src': 'classification_version'})
    events = events.merge(catalog[['sector_code', 'classification_version', 'industry_name']],
                          on=['sector_code', 'classification_version'], validate='many_to_one')
    scores = pd.read_parquet(SCORES)
    scores = scores[scores.model.eq('fnn') & scores.retention.eq(0.5)]
    assert not scores.duplicated(['event_id', 'round']).any()
    model = scores.groupby('event_id').agg(model_pass=('selected', 'sum'), model_rows=('selected', 'size'))
    assert model.model_rows.eq(2).all()
    events = events.merge(model, on='event_id', how='left', validate='one_to_one')
    funnel = pd.read_parquet(RUN / 'stock_funnel.parquet')
    assert not funnel.duplicated(['event_id', 'date']).any()
    maxima = funnel.groupby('event_id')[['qualified', 'trend', 'shape', 'stock_pass']].max()
    assert funnel.groupby('event_id').size().eq(5).all()
    events = events.merge(maxima, on='event_id', validate='one_to_one')
    events['year'] = events.date.dt.year
    events['blocked'] = events.stock_pass.eq(0)
    events['reason'] = '存在完整个股候选'
    events.loc[events.blocked & events['shape'].eq(0), 'reason'] = '五日内形态通过数始终为零'
    events.loc[events.blocked & events['shape'].gt(0) & events.trend.eq(0), 'reason'] = '有形态通过但趋势通过数始终为零'
    events.loc[events.blocked & events['shape'].gt(0) & events.trend.gt(0), 'reason'] = '趋势与形态分别有通过，但无同日同股同时满足全部条件'
    OUT.mkdir(exist_ok=True)
    events.to_csv(OUT / 'events.csv', index=False)
    summaries = []
    for scope, mask in [('frozen_both', events.model_pass.eq(2)),
                        ('chronological', events.chronological_selected)]:
        subset = events[mask]
        yearly = subset.groupby('year').agg(events=('event_id', 'size'), blocked=('blocked', 'sum')).reset_index()
        yearly.insert(0, 'scope', scope)
        summaries.append(yearly)
        prior = subset[subset.year.lt(2026)]
        print(scope, 'all', len(subset), 'blocked', int(subset.blocked.sum()),
              'before2026', len(prior), 'blocked', int(prior.blocked.sum()))
        print(prior[prior.blocked].reason.value_counts().to_string())
    pd.concat(summaries, ignore_index=True).to_csv(OUT / 'yearly.csv', index=False)
    examples = events[events.model_pass.eq(2) & events.blocked & events.year.lt(2026)].sort_values('date', ascending=False)
    print(examples[['date', 'industry_name', 'trend', 'shape', 'reason']].head(15).to_string(index=False))
    print('注意：趋势、形态列为五日内单日最大通过数，并非去重个股数；原始五日检查未要求板块持续有效。')
    print('冻结模型含事后训练信息；时间顺序模型仍是继承参数的研究回放，均非当时实盘信号。')


if __name__ == '__main__':
    main()

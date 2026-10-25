"""固定邻域、成本与回撤过程诊断；不按结果改参数。"""
from pathlib import Path
import argparse
import json
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from build_selection_stability import save
from analyze_selection_stability import pnl_by_asset


def at_date(item, date):
    return {k: v.loc[v.index <= date] if k == 'daily' else v.loc[v.date <= date]
            for k, v in item.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    run = parser.parse_args().run
    protocol = json.loads((run / 'protocol.json').read_text())
    df = pd.read_csv(run / 'summary.csv')
    assert len(df) == len(protocol['cases']) == 112
    out = run / 'analysis'; out.mkdir(exist_ok=True)
    base = df.loc[df.policy.eq('baseline')].set_index(['cohort', 'offset', 'extra_cost_bps'])
    pairs = []
    for row in df.loc[df.policy.ne('baseline')].itertuples():
        b = base.loc[(row.cohort, row.offset, row.extra_cost_bps)]
        pairs.append(dict(account=row.name, cohort=row.cohort, offset=row.offset, policy=row.policy,
            extra_cost_bps=row.extra_cost_bps, ann_delta=row.ann-b.ann, total_delta=row.total-b.total,
            mdd_delta=row.mdd-b.mdd, sharpe_delta=row.sharpe-b.sharpe, trade_fraction=row.n_trades/b.n_trades-1,
            fee_delta=row.total_fees-b.total_fees))
    pairs = pd.DataFrame(pairs)
    pairs.to_csv(out / 'paired_results.csv', index=False)
    medians = pairs.groupby(['policy','extra_cost_bps']).agg(
        ann_delta=('ann_delta','median'), mdd_delta=('mdd_delta','median'),
        sharpe_delta=('sharpe_delta','median'), trade_fraction=('trade_fraction','median'),
        annual_improved=('ann_delta', lambda x: int((x>0).sum())),
        mdd_worse=('mdd_delta', lambda x: int((x<0).sum())))
    medians.to_csv(out / 'paired_medians.csv')
    yearly = []; bounds = []; episodes = []; weights = []; contributions = []; matching = []
    for row in df.itertuples():
        folder = run / 'accounts' / row.name
        daily = pd.read_parquet(folder / 'daily.parquet')
        prior = 80000.
        for year, group in daily.groupby(daily.index.year):
            yearly.append(dict(account=row.name, policy=row.policy, cohort=row.cohort, offset=row.offset,
                extra_cost_bps=row.extra_cost_bps, year=year, total=group.equity.iloc[-1]/prior-1,
                sessions=len(group), end=str(group.index[-1].date())))
            prior = group.equity.iloc[-1]
        if row.policy not in ['baseline', 'rank10_confirm2'] or row.extra_cost_bps != 0: continue
        item = dict(daily=daily, **{k:pd.read_parquet(folder/f'{k}.parquet') for k in ['trades','holdings','events']})
        trades, events = item['trades'], item['events']
        dividend = events.loc[events.kind.eq('dividend_receivable'), 'amount'].sum()
        bounds.append(dict(account=row.name, gross_dividend=dividend, fixed_path_max_cash_dividend_tax=.2*dividend,
            fraction_initial_capital=.2*dividend/80000,
            limitation='全按20%计；只界定既定现金分红的直接税额，不是税后回测或总收益误差上界'))
        for kind, group in trades.groupby('execution_kind'):
            matching.append(dict(account=row.name, execution_kind=kind, trades=len(group), gross=group.gross.sum()))
        dd = daily.equity / daily.equity.cummax().clip(lower=80000) - 1
        trough = dd.idxmin()
        peak = daily.loc[:trough].equity.idxmax()
        assert daily.at[peak,'equity'] >= 80000  # 本组回撤起点确为已发生的净值峰值。
        period = daily.loc[peak:trough]
        ev = events.loc[events.date.gt(peak) & events.date.le(trough)]
        episodes.append(dict(account=row.name, policy=row.policy, cohort=row.cohort, offset=row.offset,
            peak=str(peak.date()), trough=str(trough.date()), mdd=dd.min(), sessions=len(period),
            peak_equity=daily.at[peak,'equity'], trough_equity=daily.at[trough,'equity'],
            mean_invested=(period.stock_value/period.equity).mean(),
            min_invested=(period.stock_value/period.equity).min(),
            peak_invested=period.stock_value.iloc[0]/period.equity.iloc[0],
            trough_invested=period.stock_value.iloc[-1]/period.equity.iloc[-1],
            fully_cash_days=int(period.n_hold.eq(0).sum()),
            exit_requests=int(ev.kind.eq('hold_exit_requested').sum())))
        diff = pnl_by_asset(at_date(item,trough)).subtract(pnl_by_asset(at_date(item,peak)),fill_value=0)
        assert abs(diff.sum() - (daily.at[trough,'equity']-daily.at[peak,'equity'])) < 1e-6
        for code, amount in diff.items(): contributions.append(dict(account=row.name,ts_code=code,pnl_yuan=amount))
        for tag, date in [('peak',peak),('trough',trough)]:
            for h in item['holdings'].loc[item['holdings'].date.eq(date)].itertuples():
                weights.append(dict(account=row.name, point=tag,date=str(date.date()),ts_code=h.ts_code,
                    shares=h.shares, price=h.price,weight=h.value/daily.at[date,'equity']))
    for name, rows in [('yearly_returns',yearly),('cash_dividend_tax_bound',bounds),('drawdown_episodes',episodes),
                       ('drawdown_holdings',weights),('drawdown_asset_pnl',contributions),('fill_assumptions',matching)]:
        pd.DataFrame(rows).to_csv(out / f'{name}.csv', index=False)
    zero = medians.xs(0, level='extra_cost_bps')
    candidate = medians.loc[('rank10_confirm2',0)]
    screen = protocol['screens']
    checks = dict(neighborhood_positive_fraction=float(zero.ann_delta.gt(0).mean()),
        candidate_annual_improved=int(candidate.annual_improved), candidate_mdd_worse=int(candidate.mdd_worse),
        cost30_median_ann_delta=float(medians.loc[('rank10_confirm2',30),'ann_delta']))
    passes = dict(neighborhood=checks['neighborhood_positive_fraction'] >= screen['neighbor_positive_median_ann_fraction_min'],
        annual_consistency=checks['candidate_annual_improved'] >= screen['candidate_positive_ann_pairs_min'],
        defense=checks['candidate_mdd_worse'] <= screen['candidate_mdd_worse_pairs_max'],
        cost=checks['cost30_median_ann_delta'] > 0)
    save(out / 'assessment.json', dict(observed=checks, predeclared_screens=passes, all_pass=all(passes.values()),
        promotion='not_approved_by_this_research', independent_oos=False,
        note='配对账户窗口重叠，不能将通过比例解释为独立胜率或统计显著性'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for ax, field, scale, title, cmap in zip(axes, ['ann_delta','mdd_worse'], [100,1],
            ['Median paired annual return change (pp)', 'Accounts with worse drawdown (out of 8)'], ['RdYlGn','YlOrRd']):
        values = np.array([[zero.loc[f'rank{b}_confirm{c}',field]*scale for c in [1,2,3]] for b in [8,10,12]])
        im = ax.imshow(values,cmap=cmap)
        for i in range(3):
            for j in range(3): ax.text(j,i,f'{values[i,j]:+.2f}' if field=='ann_delta' else f'{values[i,j]:.0f}',ha='center',va='center')
        ax.set(xticks=[0,1,2],xticklabels=[1,2,3],yticks=[0,1,2],yticklabels=[8,10,12],
               xlabel='Consecutive monthly confirmations',ylabel='Rank threshold',title=title)
        fig.colorbar(im,ax=ax,shrink=.7)
    fig.suptitle('Fixed parameter neighborhood; overlapping historical accounts, not new OOS')
    fig.savefig(out/'parameter_neighborhood.png',dpi=180)
    fig.savefig(out/'parameter_neighborhood.pdf')
    plt.close(fig)
    print(medians.to_string())
    print(json.dumps(checks,ensure_ascii=False))


if __name__ == '__main__': main()

"""汇总全部退出快慢实验，并在固定基线持仓上检查提前退出的误判与延迟。"""
import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from portfolio import daily_metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve(); out = run / 'execution'
    manifest = json.loads((run / 'manifest.json').read_text())
    close = pd.read_parquet(run / 'daily_close.parquet')
    market = pd.read_parquet(out / 'market.parquet').set_index(['date', 'ts_code'])
    weak = pd.read_parquet(run / 'monthly_weak.parquet')
    known = pd.read_parquet(run / 'monthly_observable.parquet')
    count = pd.read_parquet(run / 'daily_weak_count.parquet')
    coverage = pd.read_parquet(run / 'daily_known_count.parquet')
    streak = pd.read_parquet(run / 'monthly_failure_streak.parquet')
    variants = {v:pd.read_parquet(run / 'hold_variants' / (v+'.parquet'))
                for v in manifest['parameters']['variants']}
    reviews = weak.index
    rows, years, old_rows, mechanism, paired, actual_exits, monthly, delayed, full_rows = [], [], [], [], [], [], [], [], []
    curves = {}

    def outcome(date, code, horizon):
        """描述性复权开盘到未来收盘变动；不是执行收益，尾部缺失不补齐。"""
        i = close.index.get_indexer([date])[0]
        if i < 0 or i + horizon >= len(close) or (date,code) not in market.index:
            return np.nan
        quote = market.loc[(date,code)]
        price = quote['open'] * quote['adj_factor']
        return float(close.iloc[i+horizon][code]/price-1) if price > 0 else np.nan

    for case in manifest['cases']:
        folder = out / case['name']; v = case['variant']
        d = pd.read_csv(folder / 'daily.csv', index_col='date', parse_dates=['date'])
        e = pd.read_csv(folder / 'events.csv', parse_dates=['date', 'signal_date'])
        t = pd.read_csv(folder / 'trades.csv', parse_dates=['date', 'signal_date'])
        episodes = pd.read_csv(folder / 'entry_phase_episodes.csv', parse_dates=['entry_date','close_date'])
        recent = d.loc['2024-01-01':]
        full = json.loads((folder / 'summary.json').read_text())
        full_rows.append(dict(name=case['name'],variant=v,start=case['start'],capital=case['capital'],
                              end_equity=d.equity.iloc[-1],**{k:full[k] for k in ['total','ann','mdd','sharpe']}))
        summary = json.loads((folder / 'recent_summary.json').read_text())
        durations = [len(d.loc[r.entry_date:r.close_date]) for r in episodes.itertuples() if pd.notna(r.close_date)]
        summary.update(variant=v, capital=case['capital'], cost_multiplier=case['cost_multiplier'],
                       start=case['start'], reverse_input=case['reverse_input'],
                       max_holdings=int(d.n_hold.max()), mean_holdings=float(recent.n_hold.mean()),
                       cash_days=int(recent.n_hold.eq(0).sum()),
                       closed_episodes=len(durations), median_closed_holding_days=float(np.median(durations)) if durations else np.nan,
                       recent_gross_turnover=float(t.loc[t.date.ge('2024-01-01'),'gross'].sum()/recent.equity.mean()))
        rows.append(summary)
        y = pd.read_csv(folder / 'yearly.csv');y['variant']=v;years.append(y)
        if case['name'].startswith('continued_'):
            metrics,_=daily_metrics(d.loc[:'2023-12-31'].equity,case['capital'],244)
            old_rows.append(dict(variant=v,name=case['name'],**metrics))
        if case['name'].startswith('cold_'):
            curves[v]=recent.equity/case['capital']
            last=case['capital']
            for month,block in recent.groupby(recent.index.to_period('M')):
                monthly.append(dict(variant=v,month=str(month),total=block.equity.iloc[-1]/last-1,
                                    cash_fraction=float((block.cash/block.equity).mean())))
                last=block.equity.iloc[-1]
        fresh=e[e.kind.eq('hold_exit_requested') & e.reason.eq('hold_false')]
        for label,lo,hi in [('old','2022-01-01','2023-12-31'),('recent','2024-01-01',manifest['end'])]:
            f=fresh[fresh.date.between(lo,hi)]
            accelerated=sum(bool(variants['baseline'].at[r.signal_date,r.ts_code]) for r in f.itertuples())
            mechanism.append(dict(name=case['name'],variant=v,window=label,first_exit_requests=len(f),
                                  earlier_than_baseline_state=accelerated))
        # 真实各政策退出后价格，仅解释路径；不同政策样本不同，不当作因果比较。
        if case['name'].startswith(('cold_', 'continued_')):
            for r in fresh.itertuples():
                sold=t[t.ts_code.eq(r.ts_code)&t.side.eq('sell')&t.date.ge(r.date)]
                rebuy=t[t.ts_code.eq(r.ts_code)&t.side.eq('buy')&t.date.gt(r.date)]
                actual_exits.append(dict(name=case['name'],variant=v,ts_code=r.ts_code,signal_date=r.signal_date,
                    request_date=r.date,first_sell_date=sold.date.min(),
                    was_accelerated=bool(variants['baseline'].at[r.signal_date,r.ts_code]),
                    failure_streak=streak.at[r.signal_date,r.ts_code],weak_days=count.at[r.signal_date,r.ts_code],
                    known_days=coverage.at[r.signal_date,r.ts_code],
                    forward20_adjusted_price_return=outcome(r.date,r.ts_code,20),
                    forward60_adjusted_price_return=outcome(r.date,r.ts_code,60),
                    next_rebuy_date=rebuy.date.min()))
        # 固定原基线已持仓、首次失败的样本，所有变体面对同一组价格路径。
        if case['name'] in ['cold_baseline','continued_baseline']:
            for r in fresh.itertuples():
                date,code=r.signal_date,r.ts_code
                if case['name']=='continued_baseline' and date >= pd.Timestamp('2024-01-01'):
                    continue
                if variants['slow'].at[date,code] != 1:
                    continue
                j=reviews.get_loc(date)
                next_review=reviews[j+1] if j+1<len(reviews) else pd.NaT
                value=weak.at[next_review,code] if pd.notna(next_review) and known.at[next_review,code] else np.nan
                delayed.append(dict(window='old' if date.year<2024 else 'recent',ts_code=code,signal_date=date,
                    request_date=r.date,next_review=next_review,
                    recovered_next_review=(not bool(value)) if pd.notna(value) else np.nan,
                    forward20_adjusted_price_return=outcome(r.date,code,20),
                    forward60_adjusted_price_return=outcome(r.date,code,60)))
            for r in e[e.kind.eq('hold_retained')].itertuples():
                date,code=r.signal_date,r.ts_code
                if streak.at[date,code] != 1 or not known.at[date,code]:
                    continue
                if case['name']=='continued_baseline' and date >= pd.Timestamp('2024-01-01'):
                    continue  # 近期固定用冷启动基线，避免重叠重复计数。
                j=reviews.get_loc(date); following=reviews[j+1:]
                next_review=following[0] if len(following) else pd.NaT
                next_value=weak.at[next_review,code] if pd.notna(next_review) and known.at[next_review,code] else np.nan
                later=variants['baseline'].loc[following,code]
                next_false=later[later.eq(0)].index.min()
                row=dict(window='old' if date.year<2024 else 'recent',ts_code=code,signal_date=date,request_date=r.date,
                    weak_days=count.at[date,code],known_days=coverage.at[date,code],next_review=next_review,
                    recovered_next_review=(not bool(next_value)) if pd.notna(next_value) else np.nan,
                    next_baseline_false=next_false,
                    forward20_adjusted_price_return=outcome(r.date,code,20),
                    forward60_adjusted_price_return=outcome(r.date,code,60))
                for name,hold in variants.items():
                    row[name+'_exit']=hold.at[date,code]==0
                paired.append(row)
    summary=pd.DataFrame(rows);old=pd.DataFrame(old_rows);fixed=pd.DataFrame(paired)
    aggregates=[]
    for window,block in fixed.groupby('window'):
        for v in variants:
            selected=block[block[v+'_exit']]
            valid=selected.forward20_adjusted_price_return.dropna()
            recovery=selected.recovered_next_review.dropna()
            aggregates.append(dict(window=window,variant=v,fixed_first_failures=len(block),early_exits=len(selected),
                next_review_known=len(recovery),next_review_recovered=int(recovery.sum()),
                forward20_known=len(valid),forward20_positive=int(valid.gt(0).sum()),
                forward20_mean=float(valid.mean()),forward20_median=float(valid.median())))
    tables=dict(recent_summary=summary,full_period_summary=pd.DataFrame(full_rows),annual=pd.concat(years),old_window=old,
                exit_mechanisms=pd.DataFrame(mechanism),fixed_first_failures=fixed,
                fixed_first_failure_summary=pd.DataFrame(aggregates),actual_exit_paths=pd.DataFrame(actual_exits),
                cold_monthly=pd.DataFrame(monthly),fixed_baseline_exits_delayed_by_slow=pd.DataFrame(delayed))
    contributions=[]
    baseline_pnl=pd.read_csv(out/'cold_baseline/recent_daily_asset_pnl.csv',index_col='date').sum()
    for v in variants:
        p=pd.read_csv(out/('cold_'+v)/'recent_daily_asset_pnl.csv',index_col='date').sum()
        joined=pd.concat([baseline_pnl.rename('baseline_pnl_yuan'),p.rename('variant_pnl_yuan')],axis=1).fillna(0)
        joined['delta_yuan']=joined.variant_pnl_yuan-joined.baseline_pnl_yuan
        joined['variant']=v
        contributions.append(joined.reset_index(names='ts_code'))
    tables['cold_asset_pnl_differences']=pd.concat(contributions)
    gates=[]
    for window,frame in [('old',old),('recent_cold',summary[summary.name.str.startswith('cold_')]),
                         ('recent_continued',summary[summary.name.str.startswith('continued_')])]:
        b=frame[frame.variant.eq('baseline')].iloc[0]
        for v in variants:
            a=frame[frame.variant.eq(v)].iloc[0]
            gates.append(dict(window=window,variant=v,total_delta=a.total-b.total,mdd_delta=a.mdd-b.mdd,
                              sharpe_delta=a.sharpe-b.sharpe,calmar=a.ann/abs(a.mdd) if a.mdd else np.nan))
    tables['window_comparison']=pd.DataFrame(gates)
    for name,frame in tables.items():frame.to_csv(run/(name+'.csv'),index=False)
    fig,axes=plt.subplots(2,1,figsize=(12,8))
    for v,curve in curves.items():
        axes[0].plot(curve.index,curve,label=v,linewidth=2 if v in ['baseline','persist10'] else 1,alpha=.85)
        axes[1].plot(curve.index,(curve/curve.cummax().clip(lower=1)-1)*100,label=v)
    axes[0].set_title('Monthly execution, daily evidence: previously observed development data')
    axes[0].set_ylabel('NAV (50k cash start = 1)');axes[0].legend(ncol=3)
    axes[1].set_ylabel('Drawdown (%)')
    for ax in axes:ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(run/'sensitivity_paths.png',dpi=150);plt.close(fig)
    print(summary[summary.name.str.startswith('cold_')][['variant','total','ann','mdd','sharpe','recent_trades']].to_string(index=False))
    print('Fixed first failures\n'+pd.DataFrame(aggregates).to_string(index=False))


if __name__=='__main__':main()

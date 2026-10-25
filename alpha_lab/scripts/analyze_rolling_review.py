"""完整报告所有日历的账户分布、初始组合影响及风险月份，不挑选最优起点。"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import yaml

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from research_profit_defense import digest,save
from research_rolling_review import verify_manifest


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    run=p.parse_args().run
    manifest=verify_manifest(run);cfg=yaml.safe_load((run/'config.yaml').read_text())
    out=run/'analysis';out.mkdir(exist_ok=False)
    summaries=[];years=[];months=[];initial=[];cash_checks=[];prices={};books={};orders=fills=0
    rolling_windows=[];rolling_statistics=[];asset_months=[];holding_reviews=[]
    close=pd.read_parquet(run/'inputs/daily_close.parquet')
    long_ma=close.rolling(120,min_periods=120).mean()
    r60=close/close.shift(60)-1;r120=close/close.shift(120)-1
    for case in manifest['cases']:
        folder=run/'execution'/case['name']
        receipt=json.loads((folder/'completion.json').read_text())
        for name,h in receipt['hashes'].items():assert digest(folder/name)==h,(case['name'],name)
        s=json.loads((folder/'summary.json').read_text())
        d=pd.read_csv(folder/'daily.csv',index_col='date',parse_dates=True)
        t=pd.read_csv(folder/'trades.csv',parse_dates=['date','decision_date'])
        h=pd.read_csv(folder/'holdings.csv',parse_dates=['date'])
        e=pd.read_csv(folder/'events.csv')
        # 从逐日逐股市值变动、交易现金流与分红权利重建月度归因。
        values=h.pivot(index='date',columns='ts_code',values='value').reindex(d.index).fillna(0.)
        mark=values.diff();mark.iloc[0]=values.iloc[0]
        flow=t.assign(flow=np.where(t.side.eq('sell'),t.gross,-t.gross)-t.fee).pivot_table(index='date',columns='ts_code',values='flow',aggfunc='sum')
        asset=mark.add(flow.reindex(d.index).fillna(0.),fill_value=0.)
        div=e[e.kind.eq('dividend_receivable')].copy();div['date']=pd.to_datetime(div.date)
        if len(div):asset=asset.add(div.pivot_table(index='date',columns='ts_code',values='amount',aggfunc='sum').reindex(d.index).fillna(0.),fill_value=0.)
        assert not e.kind.eq('delist_writeoff').any(),'新增退市现金事件需显式归因'
        expected=d.equity.diff();expected.iloc[0]=d.equity.iloc[0]-cfg['capital']
        assert (asset.sum(axis=1)-expected).abs().max()<1e-6
        am=asset.resample('ME').sum().stack().rename('pnl_yuan').reset_index()
        am['name']=case['name'];am['month']=am.date.dt.to_period('M').astype(str);asset_months.append(am)
        if case['policy']=='hybrid':
            held=pd.read_parquet(run/'schedules'/case['schedule']/'hold.parquet')
            for date in held.index.intersection(d.index):
                for item in h[h.date.eq(date)].itertuples():
                    holding_reviews.append(dict(name=case['name'],date=date,ts_code=item.ts_code,shares=item.shares,
                        ret60=r60.at[date,item.ts_code],ret120=r120.at[date,item.ts_code],
                        ma120_gap=close.at[date,item.ts_code]/long_ma.at[date,item.ts_code]-1,
                        hold_allowed=held.at[date,item.ts_code],weight=item.value/d.at[date,'equity']))
        prices[case['name']]=d.equity
        books[case['name']]=h
        ret=d.equity.pct_change();ret.iloc[0]=d.equity.iloc[0]/cfg['capital']-1
        actual=dict(total=d.equity.iloc[-1]/cfg['capital']-1,
            ann=(d.equity.iloc[-1]/cfg['capital'])**(244/len(d))-1,
            mdd=(d.equity/d.equity.cummax().clip(lower=cfg['capital'])-1).min(),
            sharpe=ret.mean()/ret.std(ddof=1)*np.sqrt(244))
        for key,value in actual.items():assert abs(s[key]-value)<1e-10,(case['name'],key)
        assert d.cash.ge(-1e-7).all() and d.n_hold.le(5).all()
        assert (t.decision_date<t.date).all()
        assert t.loc[t.side.eq('buy'),'shares'].mod(100).eq(0).all()
        order=e[e.kind.eq('order_planned')]
        for date,block in order[order.side.eq('buy')].groupby('decision_date'):
            assert block.reserved_cash.sum()<=d.loc[pd.Timestamp(date),'cash']+1e-7
        orders+=len(order);fills+=len(t)
        s.update(first_buy=str(t.loc[t.side.eq('buy'),'date'].min().date()),
            initial_cash_days=int(d.index.get_loc(t.loc[t.side.eq('buy'),'date'].min())),
            trade_days=int(t.date.nunique()),end_equity=d.equity.iloc[-1])
        summaries.append(s)
        y=pd.read_csv(folder/'yearly.csv');years.append(y)
        monthly=d.equity.resample('ME').last().pct_change()
        monthly.iloc[0]=d.equity.resample('ME').last().iloc[0]/cfg['capital']-1
        months.extend(dict(name=case['name'],month=str(date.to_period('M')),ret=value)
                      for date,value in monthly.items())
        # 与年度化一致的244交易日滚动评价；承接实际持仓，非每日新开账户。
        trail=d.equity/d.equity.shift(244)-1
        valid=trail.dropna()
        rolling_windows.extend(dict(name=case['name'],end=date,start=d.index[d.index.get_loc(date)-244],ret=value)
                               for date,value in valid.items())
        if len(valid):
            rolling_statistics.append(dict(name=case['name'],windows=len(valid),minimum=valid.min(),
                median=valid.median(),maximum=valid.max(),negative_fraction=valid.lt(0).mean(),
                worst_end=valid.idxmin(),interpretation='overlapping windows; not independent samples or fresh accounts'))
        first=t[t.date.eq(t.date.min())]
        initial.extend(dict(name=case['name'],**row) for row in first[['date','ts_code','side','shares','price','gross','fee']].to_dict('records'))
        cash_checks.append(dict(name=case['name'],days=len(d),orders=len(order),fills=len(t),accounting=s['cash_max_error']))
    summary=pd.DataFrame(summaries)
    pd.concat(asset_months).to_csv(out/'monthly_asset_pnl.csv',index=False)
    pd.DataFrame(holding_reviews).to_csv(out/'review_held_states.csv',index=False)
    summary.to_csv(out/'summary.csv',index=False);pd.concat(years).to_csv(out/'yearly.csv',index=False)
    monthly=pd.DataFrame(months);monthly.to_csv(out/'monthly.csv',index=False)
    pd.DataFrame(initial).to_csv(out/'initial_trades.csv',index=False)
    pd.DataFrame(rolling_windows).to_csv(out/'rolling_year_windows.csv',index=False)
    pd.DataFrame(rolling_statistics).to_csv(out/'rolling_year_summary.csv',index=False)
    # 先按是否滚动分组，不把跨起点平均曲线包装成一个可买的账户。
    summary['family']=np.where(summary.schedule.str.startswith('calendar'),'calendar','rolling')
    distribution=[];pairs=[];comparisons=[]
    metrics=['total','ann','mdd','sharpe','n_trades','mean_invested','trade_days']
    for (window,family,policy),group in summary.groupby(['window','family','policy']):
        for metric in metrics:
            for q in cfg['report_quantiles']:
                distribution.append(dict(window=window,family=family,policy=policy,metric=metric,quantile=q,value=group[metric].quantile(q)))
    for window in cfg['windows']:
        group=summary[summary.window.eq(window)&summary.policy.eq('hybrid')]
        base=group[group.schedule.eq('calendar_01')].iloc[0]
        base_h=books[base['name']].groupby('date').ts_code.agg(set)
        for row in group.itertuples():
            hold=summary[summary.window.eq(window)&summary.schedule.eq(row.schedule)&summary.policy.eq('initial_buyhold')].iloc[0]
            pairs.append(dict(window=window,schedule=row.schedule,hybrid_total=row.total,buyhold_total=hold.total,
                active_path_delta_yuan=row.end_equity-hold.end_equity,active_path_delta_return=row.total-hold.total))
            h=books[row.name].groupby('date').ts_code.agg(set)
            shared=base_h.index.intersection(h.index)
            overlap=[len(base_h.at[d]&h.at[d])/len(base_h.at[d]|h.at[d]) for d in shared]
            # 相同起始交易的对照用于识别后续复核路径，不强制把不同起点混为同一入场。
            first=lambda name:pd.read_csv(run/'execution'/name/'trades.csv').pipe(lambda t:t[t.date.eq(t.date.min())])
            a,b=first(row.name),first(base['name'])
            same_initial=set(a.ts_code)==set(b.ts_code)
            comparisons.append(dict(window=window,schedule=row.schedule,total_delta=row.total-base.total,
                mdd_delta=row.mdd-base.mdd,sharpe_delta=row.sharpe-base.sharpe,
                mean_daily_holding_jaccard=float(np.mean(overlap)),same_initial_stock_set=same_initial,
                comparison_days=len(shared)))
            # 初始持有对照的整轮实际买单必须相同（包括成交日/股数/费用）。
            bh=pd.read_csv(run/'execution'/hold['name']/'trades.csv')
            original=pd.read_csv(run/'execution'/row.name/'trades.csv')
            original=original[original.decision_date.eq(original.decision_date.min())].reset_index(drop=True)
            pd.testing.assert_frame_equal(original,bh.reset_index(drop=True),check_exact=True)
    pd.DataFrame(distribution).to_csv(out/'distributions.csv',index=False)
    pd.DataFrame(pairs).to_csv(out/'active_vs_initial_hold.csv',index=False)
    pd.DataFrame(comparisons).to_csv(out/'versus_month_start.csv',index=False)
    risk_months=['2023-10','2024-10','2025-09','2026-06']
    monthly[monthly.month.isin(risk_months)].to_csv(out/'risk_months.csv',index=False)
    # 仅展示独立可执行账户曲线，禁止混合净值作为实盘收益。
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import fontManager,FontProperties
    fontManager.addfont('/root/autodl-tmp/fonts/SimHei.ttf')
    plt.rcParams['font.family']=FontProperties(fname='/root/autodl-tmp/fonts/SimHei.ttf').get_name()
    plt.rcParams['axes.unicode_minus']=False
    fig,axes=plt.subplots(2,2,figsize=(13,9))
    for row,window in enumerate(cfg['windows']):
        for col,family in enumerate(['calendar','rolling']):
            group=summary[summary.window.eq(window)&summary.family.eq(family)&summary.policy.eq('hybrid')]
            ax=axes[row,col]
            for item in group.itertuples():
                curve=prices[item.name]
                label=(f'每月{int(item.schedule[-2:])}日起' if family=='calendar' else f'20日间隔，偏移{int(item.schedule[-2:])}日')
                ax.plot(curve.index,curve/cfg['capital'],label=label,linewidth=1.4)
            ticks=pd.to_datetime([f'{y}-01-01' for y in range(2022 if window=='continued' else 2024,2027)])
            ax.set_xticks(ticks);ax.set_xticklabels([str(t.year) for t in ticks]);ax.grid(alpha=.2)
            ax.set_title(('2022连续' if window=='continued' else '2024现金启动')+' / '+('月内错开' if family=='calendar' else '滚动复核'))
            ax.set_ylabel('账户净值');ax.legend(fontsize=9)
    fig.suptitle('仅改变复核日历：全部起点，不选冠军｜截止2026-09-08')
    fig.tight_layout();fig.savefig(out/'calendar_paths.png',dpi=160);fig.savefig(out/'calendar_paths.pdf');plt.close(fig)
    parent=ROOT/cfg['parent'];old=json.loads((parent/'completion.json').read_text())
    for name,h in old['artifact_sha256'].items():assert digest(parent/name)==h,name
    save(out/'verification.json',dict(status='passed',cases=len(summary),orders=orders,fills=fills,checks=cash_checks,
        metrics_independently_recomputed=True,initial_buyhold_orders_exact=True,parent_artifacts_unchanged=len(old['artifact_sha256'])))
    save(out/'completion.json',dict(status='complete',source_sha256=digest(__file__),
        artifacts={p.name:digest(p) for p in out.iterdir()}))
    print(summary[summary.policy.eq('hybrid')][['window','schedule','total','ann','mdd','sharpe','n_trades']].round(4).to_string(index=False),flush=True)


if __name__=='__main__':main()

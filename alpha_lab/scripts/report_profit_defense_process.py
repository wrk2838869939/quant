"""从已完成账本生成下跌过程证据与图表，不改变信号或回放。"""
import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import yaml

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from research_profit_defense import digest,save
from analyze_profit_defense import ledger
from defense_diagnostics import risk_evidence


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    run=parser.parse_args().run
    out=run/'process';out.mkdir(exist_ok=False)
    cfg=yaml.safe_load((run/'config.yaml').read_text())
    source=ROOT/cfg['signal_parent']
    prices=pd.read_parquet(source/'daily_close.parquet')
    states=risk_evidence(prices,cfg['defense'])
    basics=pd.read_parquet(ROOT/cfg['parent']/'inputs/basics.parquet').set_index('ts_code')
    index_path=ROOT/'data/raw/index_daily.parquet'
    ix=pd.read_parquet(index_path)
    ix['date']=pd.to_datetime(ix.trade_date.astype(str))
    ix=ix.pivot(index='date',columns='ts_code',values='close').sort_index()
    months,assets,exit_paths,pauses=[],[],[],[]
    for window in cfg['windows']:
        folder=run/'execution'/f'{window}_hybrid'
        d,h,t,e,v,price,div,fee,pnl=ledger(folder,cfg['capital'])
        monthly=d.equity.resample('ME').last().pct_change()
        monthly.iloc[0]=d.equity.resample('ME').last().iloc[0]/cfg['capital']-1
        # 全部月份均导出；最差三月仅用于详查，不用于选规则。
        monthly.rename('return').to_csv(out/f'{window}_monthly.csv')
        risk=pd.read_csv(run/'analysis'/f'{window}_hybrid_risk_timeline.csv',index_col='date',parse_dates=True)
        for period in monthly.nsmallest(3).index.to_period('M'):
            dates=d.index[d.index.to_period('M')==period]
            first=d.index.get_loc(dates[0]);prev=d.index[first-1] if first>0 else dates[0]
            initial=d.equity.loc[prev] if first>0 else cfg['capital']
            stop=dates[-1]
            rr=risk.loc[dates]
            alarm_days=rr.index[rr.alarm_fraction.gt(0)]
            first_alarm=alarm_days[0] if len(alarm_days) else pd.NaT
            before_alarm=(d.index[d.index.get_loc(first_alarm)-1] if pd.notna(first_alarm) else stop)
            lost_before_alarm=d.equity.loc[before_alarm]-initial
            months.append(dict(window=window,month=str(period),start=prev,end=stop,
                start_equity=initial,end_equity=d.equity.loc[stop],return_value=d.equity.loc[stop]/initial-1,
                first_any_alarm=first_alarm,pnl_before_alarm=lost_before_alarm,
                price_pnl=price.loc[dates].sum().sum(),dividends=div.loc[dates].sum().sum(),
                fees=fee.loc[dates].sum().sum(),max_alarm_fraction=rr.alarm_fraction.max(),
                max_weak_fraction=rr.weak_fraction.max(),
                hs300_price_return=ix.at[stop,'000300.SH']/ix.at[prev,'000300.SH']-1,
                sse_price_return=ix.at[stop,'000001.SH']/ix.at[prev,'000001.SH']-1))
            for code,amount in pnl.loc[dates].sum().items():
                if amount==0:
                    continue
                present=h[h.ts_code.eq(code)&h.date.between(prev,stop)].date
                alarm=states['alarm'].loc[dates,code]
                alarm_at=alarm.index[alarm][0] if alarm.any() else pd.NaT
                request=e[e.kind.eq('hold_exit_requested')&e.ts_code.eq(code)&e.date.ge(dates[0])]
                row=dict(window=window,month=str(period),ts_code=code,name=basics.at[code,'name'],
                    net_pnl=amount,price_pnl=price.loc[dates,code].sum(),dividends=div.loc[dates,code].sum(),
                    fees=fee.loc[dates,code].sum(),first_alarm_in_month=alarm_at,
                    first_exit_request_on_or_after_month=request.date.min(),
                    adjusted_period_return=prices.at[stop,code]/prices.at[prev,code]-1,
                    ret120_end=states['ret120'].at[stop,code],ma_gap_end=states['ma_gap'].at[stop,code],
                    weak_days20_end=states['weak_count'].at[stop,code])
                assets.append(row)
        original=pd.read_csv(folder/'asset_pnl.csv').set_index('ts_code').pnl_yuan
        fast=pd.read_csv(run/'execution'/f'{window}_weekly_exit/asset_pnl.csv').set_index('ts_code').pnl_yuan
        pd.DataFrame(dict(hybrid=original,weekly_exit=fast)).fillna(0).assign(
            difference=lambda z:z.weekly_exit-z.hybrid).sort_values('difference').to_csv(out/f'{window}_weekly_pnl_difference.csv')
        # 固定退出后价格路径：只作反事实描述，不等于卖后重新配置的收益差。
        for variant in ['hybrid','weekly_exit']:
            f=run/'execution'/f'{window}_{variant}'
            ev=pd.read_csv(f/'events.csv',parse_dates=['date'])
            tr=pd.read_csv(f/'trades.csv',parse_dates=['date','decision_date'])
            req=ev[ev.kind.eq('hold_exit_requested')]
            for record in req.itertuples():
                sales=tr[tr.side.eq('sell')&tr.ts_code.eq(record.ts_code)&tr.date.gt(record.date)]
                sell_date=sales.date.min()
                if pd.isna(sell_date):
                    continue
                i=prices.index.get_loc(sell_date)
                row=dict(window=window,variant=variant,ts_code=record.ts_code,
                    request=record.date,first_sell=sell_date,reason=record.reason)
                for horizon in [20,60]:
                    row[f'after_sell_{horizon}d']=(prices.iloc[i+horizon][record.ts_code]/prices.iloc[i][record.ts_code]-1
                        if i+horizon<len(prices) else np.nan)
                exit_paths.append(row)
        for variant in ['weekly_exit','pause_only','weekly_pause','weekly_liquidate']:
            ev=pd.read_csv(run/'execution'/f'{window}_{variant}/defense_events.csv')
            pauses.append(dict(window=window,variant=variant,pause_triggers=int(ev.kind.eq('pause_trigger').sum()),
                               recoveries=int(ev.kind.eq('recovery').sum())))
    pd.DataFrame(months).to_csv(out/'worst_months.csv',index=False)
    pd.DataFrame(assets).to_csv(out/'worst_month_assets.csv',index=False)
    pd.DataFrame(exit_paths).to_csv(out/'exit_followup_prices.csv',index=False)
    pd.DataFrame(pauses).to_csv(out/'pause_counts.csv',index=False)
    # 日期显式标签，避免本机时区令月/年定位器显示前一个周期。
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties, fontManager
    fontManager.addfont('/root/autodl-tmp/fonts/SimHei.ttf')
    font=FontProperties(fname='/root/autodl-tmp/fonts/SimHei.ttf')
    plt.rcParams['font.family']=font.get_name()
    plt.rcParams['axes.unicode_minus']=False
    labels=dict(hybrid='Hybrid',initial_buyhold='初始组合持有',bank_buyhold='大型银行持有',weekly_exit='周频提前退出')
    colors=dict(hybrid='#1864ab',initial_buyhold='#2b8a3e',bank_buyhold='#e67700',weekly_exit='#c92a2a')
    fig,axes=plt.subplots(2,2,figsize=(13,8))
    for j,window in enumerate(cfg['windows']):
        for variant,label in labels.items():
            d=pd.read_csv(run/'execution'/f'{window}_{variant}/daily.csv',index_col='date',parse_dates=True)
            axes[0,j].plot(d.index,d.equity/10000,label=label,color=colors[variant])
            axes[1,j].plot(d.index,(d.equity/d.equity.cummax().clip(lower=cfg['capital'])-1)*100,color=colors[variant])
        ticks=pd.to_datetime([f'{y}-01-01' for y in range(2022 if window=='continued' else 2024,2027)])
        for ax in axes[:,j]:
            ax.set_xticks(ticks);ax.set_xticklabels([str(t.year) for t in ticks]);ax.grid(alpha=.2)
        axes[0,j].set_title(('2022连续' if window=='continued' else '2024现金启动')+'：相同5万元本金')
        axes[0,j].set_ylabel('账户资产（万元）');axes[1,j].set_ylabel('回撤（%）')
        axes[0,j].legend(fontsize=9)
    fig.suptitle('盈利来源与防守对照｜已观察历史，截至2026-09-08')
    fig.tight_layout();fig.savefig(out/'account_comparisons.png',dpi=160);fig.savefig(out/'account_comparisons.pdf');plt.close(fig)
    risk=pd.read_csv(run/'analysis/cold_hybrid_risk_timeline.csv',index_col='date',parse_dates=True).loc['2026-05-29':'2026-07-31']
    fig,axes=plt.subplots(2,1,figsize=(12,7),sharex=True)
    axes[0].plot(risk.index,risk.equity/10000,color=colors['hybrid'],label='账户资产')
    axes[0].set_ylabel('万元');axes[0].set_title('2024启动账户：2026年6月下跌与风险触发过程')
    for col,label,color in [('weak_fraction','任一趋势条件转弱','#e67700'),
                             ('broken_fraction','长期两条件均弱','#7048e8'),
                             ('alarm_fraction','长期破坏且持续转弱','#c92a2a')]:
        axes[1].plot(risk.index,risk[col]*100,label=label,color=color)
    axes[1].axhline(50,color='gray',linestyle=':',label='本轮暂停门槛')
    axes[1].set_ylabel('占当日股票市值（%）');axes[1].set_ylim(-3,103)
    ticks=pd.to_datetime(['2026-06-01','2026-06-15','2026-06-30','2026-07-15','2026-07-31'])
    axes[1].set_xticks(ticks);axes[1].set_xticklabels([t.strftime('%m-%d') for t in ticks])
    for ax in axes:
        ax.axvline(pd.Timestamp('2026-06-23'),color='gray',linestyle='--',alpha=.5)
        ax.grid(alpha=.2);ax.legend(fontsize=9)
    fig.tight_layout();fig.savefig(out/'june_risk_timeline.png',dpi=160);fig.savefig(out/'june_risk_timeline.pdf');plt.close(fig)
    save(out/'completion.json',dict(status='complete',source_sha256=digest(__file__),
        index_source_sha256=digest(index_path),artifacts={p.name:digest(p) for p in out.iterdir()},
        caveat='worst months and post-exit prices are descriptive only; no strategy uses these future labels'))
    print(pd.DataFrame(months).to_string(index=False),flush=True)


if __name__=='__main__':
    main()

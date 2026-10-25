"""按冻结篮子风险阶段诊断退出与恢复；收益路径仅用于事后归因。"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / 'outputs/targeted_restore/20260914T081149_007200Z'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--run', type=Path, default=PARENT)
    args = parser.parse_args()
    out, parent = args.out, args.run
    out.mkdir(parents=True, exist_ok=False)
    protocol = json.loads((parent / 'protocol.json').read_text())
    cash_parent = Path(protocol['cash_parent'])
    selection = Path(json.loads((cash_parent / 'protocol.json').read_text())['parent'])
    close = pd.read_parquet(selection / 'signals/close.parquet')
    ma20 = close.rolling(20).mean()
    ret20 = close / close.shift(20) - 1
    episodes, phases = [], []
    for case in protocol['cases']:
        if not case['restore'] or case['cost_bps'] != 0:
            continue
        account = parent / 'accounts' / case['name']
        daily = pd.read_parquet(account / 'daily.parquet')
        events = pd.read_parquet(account / 'events.parquet')
        trades = pd.read_parquet(account / 'trades.parquet')
        rank = cash_parent / 'accounts' / f"{case['cohort']}_{case['offset']:02d}_rank_cost0"
        rank_daily = pd.read_parquet(rank / 'daily.parquet')
        rank_trades = pd.read_parquet(rank / 'trades.parquet')
        assert daily.index.equals(rank_daily.index)
        states = pd.read_csv(account / 'risk_states.csv', parse_dates=['date'])
        alarms = states.loc[states.cap_before.eq(1) & states.cap_after.lt(1)]
        for alarm in alarms.itertuples():
            later = states.loc[states.date.gt(alarm.date)]
            recovered = later.loc[later.cap_after.eq(1), 'date'].min()
            end = recovered if pd.notna(recovered) else daily.index[-1]
            codes = alarm.basket.split('|')
            weights = pd.Series([float(x) for x in alarm.weights.split('|')], index=codes)
            segment = close.loc[alarm.date:end, codes]
            complete = not segment.isna().any().any()
            path = segment.div(segment.iloc[0]).mul(weights).sum(axis=1, min_count=len(codes))
            if not complete:
                path[:] = np.nan  # 缺报价的整段价格归因标未知，保留事件与真实账本。
            weak = close.loc[alarm.date:end, codes].lt(ma20.loc[alarm.date:end, codes]).mul(weights).sum(axis=1).where(segment.notna().all(axis=1))
            momentum = ret20.loc[alarm.date:end, codes].mul(weights).sum(axis=1)
            # 当前持仓权重回推构成固定单位篮子；不混合个股不同日期峰值。
            history = close.loc[:alarm.date, codes].tail(60).div(close.loc[alarm.date, codes]).mul(weights).sum(axis=1)
            basket_dd = 1 - history.iloc[-1] / history.max()
            weeks = states.loc[states.date.between(alarm.date, end), 'date']
            reclaim = weak.loc[weak.index.isin(weeks) & weak.le(.4)].index.min()
            tradable = events.loc[events.kind.eq('order_filled') & events.reason.eq('portfolio_cash_cap') & events.date.gt(alarm.date) & events.date.le(end)]
            info = dict(account=case['name'], label=case['label'], alarm=alarm.date, recovery=recovered,
                        sessions=len(path)-1, censored=pd.isna(recovered), cash_sell_fills=len(tradable), complete_price_path=complete,
                        avg_individual_dd=alarm.drawdown, actual_basket_dd=basket_dd,
                        peak_mismatch=alarm.drawdown-basket_dd, basket_return=path.iloc[-1]-1,
                        basket_worst=path.min()-1, rebound_from_trough=path.iloc[-1]/path.min()-1,
                        first_weak_reclaim=reclaim,
                        extra_wait_sessions=int(((path.index>reclaim)&(path.index<=end)).sum()) if pd.notna(reclaim) else np.nan,
                        return_after_first_reclaim=path.iloc[-1]/path.loc[reclaim]-1 if pd.notna(reclaim) else np.nan,
                        weeks_weak_recovered_but_ret_negative=int((weak.reindex(weeks).le(.4)&momentum.reindex(weeks).le(0)).sum()))
            start = close.index.get_loc(alarm.date)
            for horizon in [5,20,60]:
                if start+horizon < len(close):
                    f = close.iloc[start:start+horizon+1][codes].div(close.loc[alarm.date,codes]).mul(weights).sum(axis=1, min_count=len(codes))
                    if f.isna().any():
                        f[:] = np.nan
                    info[f'forward_{horizon}'] = f.iloc[-1]-1
                    info[f'worst_{horizon}'] = f.min()-1
            episodes.append(info)
        # 阶段归因严格加总到真实净值差；这是两套路径的差，不假称纯择时alpha。
        risk = events.loc[events.kind.eq('portfolio_cash_state')].set_index('date').paused.astype(bool)
        phase = risk.astype('boolean').reindex(daily.index).ffill().fillna(False).shift(1, fill_value=False)
        pnl_a = daily.equity.diff().fillna(daily.equity.iloc[0]-80000)
        pnl_b = rank_daily.equity.diff().fillna(rank_daily.equity.iloc[0]-80000)
        fees_a = trades.groupby('date').fee.sum().reindex(daily.index, fill_value=0)
        fees_b = rank_trades.groupby('date').fee.sum().reindex(daily.index, fill_value=0)
        delta = pnl_a-pnl_b
        assert abs(delta.sum()-(daily.equity.iloc[-1]-rank_daily.equity.iloc[-1])) < 1e-6
        for flag in [True,False]:
            ix = phase.eq(flag)
            phases.append(dict(account=case['name'],label=case['label'],phase='risk_active' if flag else 'risk_clear',
                sessions=int(ix.sum()),net_pnl_delta_yuan=delta[ix].sum(),fee_drag_yuan=(fees_a-fees_b)[ix].sum(),
                gross_path_delta_yuan=(delta+fees_a-fees_b)[ix].sum()))
    ep = pd.DataFrame(episodes)
    ep.to_csv(out/'episodes.csv',index=False)
    pd.DataFrame(phases).to_csv(out/'phase_attribution.csv',index=False)
    summary = ep.groupby('label').agg(episodes=('account','size'),duration_median=('sessions','median'), complete=('complete_price_path','sum'),
        return_median=('basket_return','median'),positive=('basket_return',lambda x:int((x>0).sum())),
        worst_median=('basket_worst','median'),rebound_median=('rebound_from_trough','median'),
        wait_after_reclaim_median=('extra_wait_sessions','median'),
        gain_during_extra_wait_median=('return_after_first_reclaim','median'),
        peak_mismatch_median=('peak_mismatch','median'),forward20_median=('forward_20','median'))
    summary.to_csv(out/'episode_summary.csv')
    print(summary.to_string())
    print(pd.DataFrame(phases).groupby(['label','phase'])[['net_pnl_delta_yuan','fee_drag_yuan','gross_path_delta_yuan']].median().to_string())


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""审计冻结开发账本的决策漏斗、现金、交易成本和逐股损益；不选新参数。"""
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from investable import investable_signals
from portfolio import board_of
from path_diagnostics import asset_daily_states, holding_episodes


def main():
    spec = yaml.safe_load((ROOT / 'configs/process_diagnostics.yaml').read_text())
    prepared = ROOT / spec['prepared']
    cfg = yaml.safe_load((prepared / 'config.yaml').read_text())
    lo, hi = map(pd.Timestamp, cfg['valid'])
    if hi >= pd.Timestamp('2024-01-01'):
        raise ValueError('不能读取冻结TEST')
    out = ROOT / spec['output_root'] / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    out.mkdir(parents=True, exist_ok=False)
    prefix = spec['strategy'] + '_cost1_'
    daily = pd.read_csv(prepared / (prefix+'daily.csv'), index_col='date', parse_dates=True)
    trades = pd.read_csv(prepared / (prefix+'trades.csv'), parse_dates=['date', 'signal_date'])
    holdings = pd.read_csv(prepared / (prefix+'holdings.csv'), parse_dates=['date'])
    events = pd.read_csv(prepared / (prefix+'events.csv'), parse_dates=['date'])
    raw = pd.read_parquet(ROOT / spec['raw_signals'])
    frozen = pd.read_parquet(prepared / 'candidate_signals.parquet')
    if raw.index.min() < lo or raw.index.max() > hi:
        raise ValueError('信号日期超出开发范围')
    pcfg = dict(cfg['portfolio'])
    pcfg['capital'] -= spec['reserve_cash']
    market = pd.read_parquet(ROOT / 'data/panel/daily_panel.parquet',
        columns=['date', 'ts_code', 'close', 'vol_share'],
        filters=[('date', '>=', lo), ('date', '<=', hi), ('date', 'in', list(raw.index))],
        use_threads=False)
    close = market.pivot(index='date', columns='ts_code', values='close').reindex_like(raw)
    volume = market.pivot(index='date', columns='ts_code', values='vol_share').reindex_like(raw)
    basics = pd.read_parquet(ROOT / 'data/raw/stock_basic_all.parquet')
    for col in ['list_date', 'delist_date']:
        basics[col] = pd.to_datetime(basics[col], errors='coerce')
    names = pd.read_parquet(ROOT / 'data/raw/namechange_all.parquet')
    names['start'] = pd.to_datetime(names.start_date, errors='coerce')
    names['end'] = pd.to_datetime(names.end_date, errors='coerce')
    st = names[names.name.str.contains('ST', case=False, na=False)][['ts_code', 'start', 'end']]
    multiplier = 1.05  # 冻结父实验参数，仅复现原有过滤，不是新策略假设。
    legacy, allowed = investable_signals(raw, close, volume, basics, st, pcfg, multiplier)
    pd.testing.assert_frame_equal(legacy, frozen, check_dtype=False)
    # 仅解除价格预算，保留原有合法性/成交量约束，找出价格过滤的单独影响。
    no_price_cfg = dict(pcfg, capital=np.finfo(float).max / 2)
    legal, _ = investable_signals(raw, close, volume, basics, st, no_price_cfg, multiplier)
    minimum = pd.Series({code: pcfg['boards'][board_of(code)]['min_buy'] for code in raw.columns})
    gross = close * minimum * multiplier
    lot_cost = gross + np.maximum(gross * pcfg['buy_cost'], pcfg['min_fee'])
    funnel, blocked = [], []
    for date in raw.index:
        ranks = legal.loc[date].dropna().sort_values(ascending=False, kind='stable')
        selected = ranks.head(pcfg['top_n'])
        lost = selected.index[~allowed.loc[date, selected.index]]
        affordable_account = lot_cost.loc[date] <= pcfg['capital']
        funnel.append(dict(date=date, raw_candidates=int(raw.loc[date].count()),
            legal_candidates=len(ranks), legacy_candidates=int(legacy.loc[date].count()),
            equal_budget_excluded=int((legal.loc[date].notna() & legacy.loc[date].isna()).sum()),
            top_ranked_lost=len(lost), top_ranked_lost_but_account_affordable=int(affordable_account[lost].sum())))
        for code in lost:
            blocked.append(dict(date=date, ts_code=code, legal_rank=int(ranks.index.get_loc(code)+1),
                score=ranks[code], signal_close=close.loc[date, code], min_buy=minimum[code],
                buffered_lot_cost=lot_cost.loc[date, code], seat_budget=pcfg['capital']/pcfg['top_n'],
                account_affordable=bool(affordable_account[code])))
    pd.DataFrame(funnel).to_csv(out/'selection_funnel.csv', index=False)
    pd.DataFrame(blocked).to_csv(out/'price_blocked_candidates.csv', index=False)
    states = asset_daily_states(daily, trades, holdings, events, cfg['portfolio']['capital'])
    states, episodes = holding_episodes(states, trades, daily.index)
    states.to_parquet(out/'asset_daily_states.parquet', index=False)
    episodes.to_csv(out/'holding_episodes.csv', index=False)
    age = states.groupby('age_bucket', observed=True).agg(pnl=('pnl','sum'), fees=('fees','sum'),
        capital_days=('previous_value','sum'), observations=('date','size'))
    age['pnl_per_capital_day'] = age.pnl / age.capital_days.replace(0,np.nan)
    age.to_csv(out/'age_pnl.csv')
    cycles = []
    for signal_date, group in trades.groupby('signal_date'):
        buys = group[group.side == 'buy'].sort_values('date', kind='stable')
        sells = group[group.side == 'sell']
        execution_date = daily.index[daily.index > signal_date][0]
        row = daily.loc[execution_date]
        active_cash = row.cash - spec['reserve_cash']
        first_cash = float(buys.iloc[0].cash_after + buys.iloc[0].gross + buys.iloc[0].fee) if len(buys) else np.nan
        cycles.append(dict(signal_date=signal_date, execution_date=execution_date,
            buy_count=len(buys), sell_count=len(sells), buy_gross=buys.gross.sum(),
            sell_gross=sells.gross.sum(), fees=group.fee.sum(), cash_before_first_buy=first_cash,
            execution_close_active_cash=active_cash, execution_close_holdings=row.n_hold,
            min_buy_notional=buys.gross.min(), max_buy_notional=buys.gross.max()))
    pd.DataFrame(cycles).to_csv(out/'rebalance_states.csv', index=False)
    trades['effective_fee'] = trades.fee / trades.gross
    trades['at_minimum_fee'] = np.isclose(trades.fee, pcfg['min_fee'])
    trades.to_csv(out/'order_costs.csv', index=False)
    costs = trades.groupby('side').agg(orders=('fee','size'), fees=('fee','sum'),
        median_gross=('gross','median'), minimum_fee_fraction=('at_minimum_fee','mean'),
        median_effective_fee=('effective_fee','median'))
    costs.to_csv(out/'cost_summary.csv')
    pnl_day = states.groupby('date').agg(asset_pnl=('pnl','sum'), losing_names=('pnl',lambda x:(x<0).sum()),
        profit_names=('pnl',lambda x:(x>0).sum()))
    detail = daily.join(pnl_day)
    detail['active_cash'] = detail.cash-spec['reserve_cash']
    detail['drawdown'] = detail.equity/detail.equity.cummax()-1
    detail.to_csv(out/'daily_process.csv')
    detail.nsmallest(spec['worst_days'], 'ret').to_csv(out/'worst_days.csv')
    trough = detail.drawdown.idxmin()
    peak = detail.loc[:trough].equity.idxmax()
    dd_states = states[(states.date>peak)&(states.date<=trough)]
    dd_assets = dd_states.groupby('ts_code').pnl.sum().sort_values()
    dd_assets.to_csv(out/'drawdown_asset_pnl.csv')
    closed = episodes[episodes.closed]
    f = pd.DataFrame(funnel)
    summary = dict(output=str(out.relative_to(ROOT)), sample='2022-2023 repeatedly used development sample',
        test_market_data_used=False, frozen_signals_exactly_reproduced=True,
        initial_total_capital=cfg['portfolio']['capital'], initial_active_capital=pcfg['capital'],
        fixed_reserve_cash=spec['reserve_cash'], mean_raw_candidates=f.raw_candidates.mean(),
        mean_legal_candidates=f.legal_candidates.mean(), mean_equal_budget_excluded=f.equal_budget_excluded.mean(),
        top_ranked_lost=int(f.top_ranked_lost.sum()),
        top_ranked_lost_but_account_affordable=int(f.top_ranked_lost_but_account_affordable.sum()),
        mean_active_cash=detail.active_cash.mean(), mean_holdings=daily.n_hold.mean(),
        daily_pnl_max_reconciliation_error=float((detail.asset_pnl.fillna(0)-daily.equity.diff().fillna(daily.equity.iloc[0]-cfg['portfolio']['capital'])).abs().max()),
        first_rebalance=cycles[0], costs=costs.reset_index().to_dict('records'),
        drawdown_peak=str(peak.date()), drawdown_trough=str(trough.date()),
        drawdown_pnl=float(dd_assets.sum()), drawdown_losing_names=int((dd_assets<0).sum()),
        drawdown_total_names=len(dd_assets), drawdown_worst5_pnl=float(dd_assets.head(5).sum()),
        closed_episodes=len(closed), median_closed_holding_days=float(closed.trading_days.median()),
        closed_winners=int((closed.pnl>0).sum()),
        profitable_at_some_close_but_closed_loss=int(((closed.max_marked_gain>0)&(closed.pnl<0)).sum()),
        limitations=['未计算卖出旧股与买入新股的可执行反事实收益；尚不能判定换仓创造净收益',
            '持仓年龄分组存在留存选择，不可据此选择最佳退出日',
            '使用旧研究费率，未代表用户实际券商费用或滑点',
            '股价预算过滤改变了候选池；该账本不能作为个人账户方案'])
    (out/'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    sources=[Path(__file__),ROOT/'src/path_diagnostics.py',ROOT/'src/investable.py',ROOT/'configs/process_diagnostics.yaml']
    inputs=[ROOT/spec['raw_signals'], prepared/'candidate_signals.parquet',prepared/'config.yaml']
    inputs += [prepared/(prefix+suffix+'.csv') for suffix in ['daily','trades','holdings','events']]
    (out/'manifest.json').write_text(json.dumps(dict(status='complete', sources={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}, inputs={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}), indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str), flush=True)


if __name__ == '__main__':
    main()

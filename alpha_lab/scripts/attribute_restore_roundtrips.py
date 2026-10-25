"""用真实回测卖出/补回配对核算现金价差、漏领股息和费用；不替代全账户归因。"""
import argparse
from collections import defaultdict, deque
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    protocol = json.loads((args.run/'protocol.json').read_text())
    cash = Path(protocol['cash_parent'])
    selection = Path(json.loads((cash/'protocol.json').read_text())['parent'])
    actions = pd.read_parquet(selection/'inputs/normalized_actions.parquet')
    close = pd.read_parquet(selection/'signals/close.parquet')
    pairs = []
    for case in protocol['cases']:
        if not case.get('restore',True) or case['cost_bps'] != 0:
            continue
        folder = args.run/'accounts'/case['name']
        events = pd.read_parquet(folder/'events.parquet')
        trades = pd.read_parquet(folder/'trades.parquet').set_index('order_id')
        lots = defaultdict(deque)
        positions = defaultdict(int)
        for e in events.itertuples():
            if e.kind == 'restore_episode_started':
                lots.clear()
            if e.kind in ['restore_quota_invalidated','delist_writeoff','bonus_shares']:
                for key in list(lots):
                    if key[0] == e.ts_code:
                        del lots[key]
                if e.kind == 'bonus_shares':
                    positions[e.ts_code] += int(e.shares)
                elif e.kind == 'delist_writeoff':
                    positions[e.ts_code] = 0
            if e.kind != 'order_filled':
                continue
            positions[e.ts_code] += int(e.shares)*(1 if e.side == 'buy' else -1)
            if positions[e.ts_code] == 0 or (e.side == 'buy' and e.reason != 'targeted_restore'):
                for key in list(lots):
                    if key[0] == e.ts_code:
                        del lots[key]
            if positions[e.ts_code] == 0:
                continue
            if e.reason not in ['portfolio_cash_cap','targeted_restore']:
                continue
            trade = trades.loc[e.order_id]
            key = (e.ts_code,int(e.restore_epoch))
            if e.side == 'sell':
                lots[key].append(dict(date=e.date,price=trade.price,remaining=int(e.shares),fee_per_share=trade.fee/e.shares))
                continue
            quantity = int(e.shares)
            while quantity:
                lot = lots[key][0]
                n = min(quantity,lot['remaining'])
                missing = actions.loc[actions.ts_code.eq(e.ts_code)&actions.record_date.ge(lot['date'])&actions.record_date.lt(e.date)]
                share_action = missing.stk_div.ne(0).any()
                # 本批未发生股数事件配对；若有，不把未调整原价当作现金收益。
                dividend = np.nan if share_action else missing.cash_div_tax.sum()*n
                price_benefit = n*(lot['price']-trade.price)
                fees = n*(lot['fee_per_share']+trade.fee/e.shares)
                pos = close.index.get_loc(e.date)
                after20 = close.iloc[pos+20][e.ts_code]/close.at[e.date,e.ts_code]-1 if pos+20<len(close) else np.nan
                pairs.append(dict(account=case['name'],label=case['label'],code=e.ts_code,epoch=key[1],
                    sell_date=lot['date'],restore_date=e.date,shares=n,sell_price=lot['price'],restore_price=trade.price,
                    sold_notional=n*lot['price'],price_benefit_yuan=price_benefit,missed_dividend_yuan=dividend,
                    fees_yuan=fees,net_benefit_yuan=price_benefit-dividend-fees,post_restore20_proxy=after20,
                    share_action_unknown=bool(share_action)))
                quantity -= n
                lot['remaining'] -= n
                if not lot['remaining']:
                    lots[key].popleft()
    pairs = pd.DataFrame(pairs)
    pairs.to_csv(args.out/'matched_lots.csv', index=False)
    by = pairs.groupby(['account','label']).agg(matched_parts=('shares','size'),sold_notional=('sold_notional','sum'),
        price_benefit_yuan=('price_benefit_yuan','sum'),missed_dividend_yuan=('missed_dividend_yuan','sum'),
        fees_yuan=('fees_yuan','sum'),net_benefit_yuan=('net_benefit_yuan','sum'))
    by.to_csv(args.out/'accounts.csv')
    summary = pairs.groupby('label').agg(parts=('shares','size'),price_benefit=('price_benefit_yuan','sum'),
        dividend=('missed_dividend_yuan','sum'),fees=('fees_yuan','sum'),net=('net_benefit_yuan','sum'),
        benefit_parts=('net_benefit_yuan',lambda x:int(x.gt(0).sum())),unknown=('share_action_unknown','sum'),
        post_restore20_median=('post_restore20_proxy','median'))
    summary.to_csv(args.out/'summary.csv')
    (args.out/'limitations.json').write_text(json.dumps(dict(
        contract='同轮次同股实际风控卖出和实际恢复FIFO配对；价差减漏领税前股息减两端按股费用，与始终持有这些配对股数比较。',
        unpaired_sales='未补回部分不在此表，不能把配对总额当作全策略总损益。',
        subsequent_returns='补回后20日复权价格仅诊断代理，不保证实持20日。',
        independent_samples=False),ensure_ascii=False,indent=2))
    print(summary.to_string())


if __name__ == '__main__':
    main()

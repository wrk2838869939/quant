"""只用真实风控减仓成交建立恢复额度；额度可持久化，不等于买入授权。"""
from copy import deepcopy


class RestoreQuota:
    def __init__(self, payload=None):
        self.data=deepcopy(payload) if payload is not None else dict(epoch=0,shares={},receipts={})
        if (set(self.data)!={'epoch','shares','receipts'} or type(self.data['epoch']) is not int
                or self.data['epoch']<0 or not isinstance(self.data['shares'],dict)
                or not isinstance(self.data['receipts'],dict)
                or any(not isinstance(c,str) or type(q) is not int or q<0 for c,q in self.data['shares'].items())):
            raise ValueError('恢复额度持久化状态非法')
        balances={}
        for receipt_id,r in self.data['receipts'].items():
            if (not isinstance(receipt_id,str) or not receipt_id or not isinstance(r,dict)
                    or set(r)!={'code','side','shares','reason','epoch'}):raise ValueError('恢复成交证据结构非法')
            self._validate(receipt_id,**r)
            if r['epoch']==self.data['epoch']:
                amount=r['shares'] if r['side']=='sell' and r['reason']=='portfolio_cash_cap' else (
                    -r['shares'] if r['side']=='buy' and r['reason']=='targeted_restore' else 0)
                balances[r['code']]=balances.get(r['code'],0)+amount
        if any(self.data['shares'].get(c,0)>balances.get(c,0) for c in set(balances)|set(self.data['shares'])):
            raise ValueError('持久化恢复额度缺少足额实际成交证据')

    def start_episode(self):
        self.data['epoch']+=1
        self.data['shares']={}

    def clear(self,code):
        self.data['shares'].pop(code,None)

    def available(self,code):return self.data['shares'].get(code,0)

    @staticmethod
    def _validate(receipt_id,code,side,shares,reason,epoch):
        if (not isinstance(receipt_id,str) or not receipt_id or type(shares) is not int or shares<=0 or type(epoch) is not int or epoch<0
                or not isinstance(code,str) or not code or not isinstance(reason,str) or side not in ['buy','sell']
                or (reason=='portfolio_cash_cap' and side!='sell') or (reason=='targeted_restore' and side!='buy')):
            raise ValueError('恢复额度需要唯一成交号、正整数股数和有效轮次')

    def fill(self,receipt_id,code,side,shares,reason,epoch):
        self._validate(receipt_id,code,side,shares,reason,epoch)
        receipt=dict(code=code,side=side,shares=shares,reason=reason,epoch=epoch)
        old=self.data['receipts'].get(receipt_id)
        if old is not None:
            if old!=receipt:raise ValueError('恢复额度成交回报冲突')
            return
        if side not in ['buy','sell']:raise ValueError('未知成交方向')
        if epoch==self.data['epoch']:
            if side=='sell' and reason=='portfolio_cash_cap':self.data['shares'][code]=self.available(code)+shares
            elif side=='buy' and reason=='targeted_restore':
                if shares>self.available(code):raise ValueError('恢复成交超过本次真实减仓额度')
                self.data['shares'][code]=self.available(code)-shares
        elif reason=='targeted_restore':raise ValueError('恢复成交属于过期风控轮次')
        self.data['receipts'][receipt_id]=receipt

    def snapshot(self):return deepcopy(self.data)

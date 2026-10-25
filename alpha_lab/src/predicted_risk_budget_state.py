"""冻结风险概率只控制预算；高风险期间允许按较低预算新建仓。"""
import numpy as np


class PredictedRiskBudget:
    variant='staged'

    def __init__(self, features, probabilities, enter, release, risk_cap):
        if not 0<=release<enter<=1 or risk_cap not in [.5,.8]:
            raise ValueError('只允许预定阈值与预算')
        self.features,self.probabilities=features,probabilities
        self.enter,self.release,self.risk_cap=enter,release,risk_cap
        self.weekly=set(probabilities.index.to_series().groupby(probabilities.index.to_period('W')).first())
        self.weights=None;self.cap=1.;self.records=[];self.last_date=None;self.recovered_today=False

    def on_close(self,date,values,equity):
        if self.last_date is not None and date<=self.last_date:
            raise ValueError('状态时间必须递增')
        self.last_date=date;self.recovered_today=False
        if date in self.weekly:
            p=self.probabilities.get(date,np.nan)
            if not np.isfinite(p):
                raise ValueError('缺冻结市场风险概率，不能猜测安全')
            before=self.cap
            if p>=self.enter:
                self.weights={'market_risk':1.}
                self.cap=self.risk_cap
            elif self.weights is not None and p<=self.release:
                self.weights=None;self.cap=1.;self.recovered_today=True
            self.records.append(dict(date=date,probability=p,cap_before=before,cap_after=self.cap,
                frozen=self.weights is not None,equity=equity))
        # 不将风险预算与停止建仓混淆；恢复额度只在风险轮次解除后可使用。
        return self.cap,False

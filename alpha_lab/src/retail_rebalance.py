"""现金、旧仓和整手申报共同约束的研究分配器；收益估计不是确定收益。

只优化显式提供的预期利润与交易成本，逐股风险上限由调用方指定。
不按平均席位预算排除股价；不强制填满持仓数。相关性风险尚未建模。
"""
import numpy as np
import pandas as pd
from scipy.optimize import Bounds,LinearConstraint,milp


def plan_rebalance(candidates,*,cash,buy_rate,sell_rate,min_fee,forecast_kind,max_positions=None,slippage_rate=0.,solver_relative_gap=0.001,required_new_codes=(),required_new_shares=None):
    """联合分配；保留旧仓下界，可显式强制已由上游选定的新仓最小申报量。"""
    if not np.isfinite(solver_relative_gap) or not 0 <= solver_relative_gap <= .01:
        raise ValueError('求解相对误差必须介于0和1%')
    cols=['ts_code','price','expected_return','shares','available','min_buy','buy_step','sell_step','max_notional','can_buy','can_sell']
    if forecast_kind!='expected_simple_return' or not set(cols).issubset(candidates):
        raise ValueError('缺少显式收益、旧仓或交易规则，不能把rank直接作为收益')
    x=candidates[cols].copy().reset_index(drop=True);n=len(x)
    # 旧仓保留是显式硬约束；省略时完全沿用原有零下界。
    x['min_target_shares']=(candidates['min_target_shares'].to_numpy()
        if 'min_target_shares' in candidates else 0)
    required_new_shares=dict(required_new_shares or {})
    required_new_codes=set(required_new_codes)|set(required_new_shares)
    if not required_new_codes.issubset(set(x.ts_code)):
        raise ValueError('强制新仓代码不在候选中')
    if not n:return dict(plan=x,cash_after=cash,expected_net_profit=0.)
    numeric=x[cols[1:]+['min_target_shares']].to_numpy(float)
    if x.ts_code.duplicated().any() or not np.isfinite(numeric).all() or not np.isfinite([cash,buy_rate,sell_rate,min_fee,slippage_rate]).all() or min(cash,buy_rate,sell_rate,min_fee,slippage_rate)<0:raise ValueError('无效分配输入')
    for field in ['shares','available','min_buy','buy_step','sell_step','min_target_shares']:
        if (x[field]!=np.floor(x[field])).any() or (x[field]<0).any():raise ValueError('股数/步长必须为非负整数')
    if (x.price<=0).any() or (x[['min_buy','buy_step','sell_step']]<=0).any().any() or (x.available>x.shares).any() or (x.max_notional<0).any() or (x.expected_return<-1).any():raise ValueError('价格、风险或可卖量非法')
    if not x.can_buy.isin([0,1]).all() or not x.can_sell.isin([0,1]).all():raise ValueError('交易许可必须为布尔值')
    if (x.min_target_shares>x.shares).any():raise ValueError('保留股数下界不得超过旧仓')
    required_new=x.ts_code.isin(required_new_codes).to_numpy()
    if ((x.shares.to_numpy()!=0)&required_new).any() or ((~x.can_buy.to_numpy(bool))&required_new).any():
        raise ValueError('强制新仓必须没有旧仓且允许买入')
    required_quantities=x.min_buy.to_numpy(float).copy()
    for i, row in x.iterrows():
        if row.ts_code in required_new_shares:
            quantity=required_new_shares[row.ts_code]
            if not np.isfinite(quantity) or quantity!=int(quantity) or quantity<row.min_buy or (quantity-row.min_buy)%row.buy_step!=0:
                raise ValueError('强制新仓股数必须满足最低申报量和整手步长')
            required_quantities[i]=quantity
    px=x.price.to_numpy(float);current=x.shares.to_numpy(float)
    available=np.where(x.can_sell.to_numpy(bool),x.available.to_numpy(float),0.)
    locked=current-available
    cap=np.maximum(x.max_notional.to_numpy(float),locked*px)
    mb=x.min_buy.to_numpy(float);bs=x.buy_step.to_numpy(float);ss=x.sell_step.to_numpy(float)
    odd=available%ss;max_sell=np.floor(available/ss)
    max_buy=np.maximum(0,np.floor((cap/px-mb)/bs))
    size=9*n
    # 分块：买许可、买步数、卖许可、卖步数、卖零股、目标持仓、买费、卖费、期末退出费。
    buy=np.zeros((n,size));sell=buy.copy()
    for i in range(n):
        buy[i,i]=mb[i];buy[i,n+i]=bs[i]
        sell[i,3*n+i]=ss[i];sell[i,4*n+i]=odd[i]
    delta=buy-sell;target_value=delta*px[:,None]
    objective=-(target_value*x.expected_return.to_numpy(float)[:,None]).sum(axis=0)
    objective[6*n:]=1.
    slippage=(buy+sell)*px[:,None]*slippage_rate
    objective+=slippage.sum(axis=0)+target_value.sum(axis=0)*slippage_rate
    rows=[];lower=[];upper=[]
    def add(row,lo=-np.inf,hi=np.inf):rows.append(row);lower.append(lo);upper.append(hi)
    for i in range(n):
        if required_new[i]:
            row=np.zeros(size);row[i]=1;add(row,lo=1)
            add(buy[i],lo=required_quantities[i])
        row=np.zeros(size);row[n+i]=1;row[i]=-max_buy[i];add(row,hi=0)
        row=np.zeros(size);row[3*n+i]=1;row[2*n+i]=-max_sell[i];add(row,hi=0)
        row=np.zeros(size);row[4*n+i]=1;row[2*n+i]=-1;add(row,hi=0)
        row=np.zeros(size);row[i]=1;row[2*n+i]=1;add(row,hi=1)
        add(sell[i],hi=available[i])
        add(delta[i],lo=x.min_target_shares.iloc[i]-current[i])
        row=target_value[i].copy();row[5*n+i]=-cap[i];add(row,hi=-px[i]*current[i])
        row=delta[i].copy();row[5*n+i]-=1;add(row,lo=-current[i])
        for index,activity,amount,constant,rate in [(6*n,i,buy[i]*px[i],0.,buy_rate),(7*n,2*n+i,sell[i]*px[i],0.,sell_rate),(8*n,5*n+i,target_value[i],current[i]*px[i],sell_rate)]:
            row=amount*rate;row[index+i]-=1;add(row,hi=-constant*rate)
            row=np.zeros(size);row[activity]=min_fee;row[index+i]=-1;add(row,hi=0)
    budget=(buy*px[:,None]-sell*px[:,None]+slippage).sum(axis=0);budget[6*n:8*n]+=1;add(budget,hi=cash)
    if max_positions is not None:
        if int(max_positions)!=max_positions or max_positions<0:raise ValueError('持仓上限非法')
        row=np.zeros(size);row[5*n:6*n]=1;add(row,hi=max_positions)
    ub=np.r_[x.can_buy.to_numpy(float),max_buy,(available>0).astype(float),max_sell,(odd>0).astype(float),np.ones(n),np.full(3*n,np.inf)]
    # 不可交易的旧仓由 available=0/can_sell=false 保留；不存在虚构卖出释放现金。
    result=milp(objective,integrality=np.r_[np.ones(6*n),np.zeros(3*n)],
        bounds=Bounds(np.zeros(size),ub),constraints=LinearConstraint(np.array(rows),lower,upper),
        options={'mip_rel_gap':solver_relative_gap,'time_limit':30.})
    if not result.success:raise RuntimeError(f'联合整手分配未达到误差界限，禁止下单: status={result.status}, gap={getattr(result, "mip_gap", None)}')
    b=np.rint(buy@result.x).astype(int);s=np.rint(sell@result.x).astype(int);q=current+b-s
    bf=np.where(b>0,np.maximum(b*px*buy_rate,min_fee),0.)
    sf=np.where(s>0,np.maximum(s*px*sell_rate,min_fee),0.)
    ef=np.where(q>0,np.maximum(q*px*sell_rate,min_fee)+q*px*slippage_rate,0.)
    slippage_cost=(b+s)*px*slippage_rate
    after=float(cash+(s*px-sf-b*px-bf-slippage_cost).sum())
    if after < -1e-5 or (s>available).any() or (q<locked).any() or (q<x.min_target_shares).any() or (q*px>cap+1e-5).any() or ((b>0)&(s>0)).any():raise RuntimeError('分配器独立现金/锁定/风险/保留股数对账失败')
    if ((b>0)&((b<mb)|((b-mb)%bs!=0))).any() or (b[~x.can_buy.astype(bool)]>0).any() or (s[~x.can_sell.astype(bool)]>0).any():raise RuntimeError('分配器交易单位或许可越界')
    if (b[required_new]<required_quantities[required_new]).any():raise RuntimeError('已选新仓最低申报量未执行')
    x['buy_shares']=b;x['sell_shares']=s;x['target_shares']=q.astype(int)
    x['buy_fee']=bf;x['sell_fee']=sf;x['expected_exit_fee']=ef
    x['slippage_cost']=slippage_cost
    x['unavoidable_locked_excess']=np.maximum(0,locked*px-x.max_notional)
    return dict(plan=x,cash_after=max(0.,after),solver_gap=float(result.mip_gap),solver_absolute_gap=float(abs(result.fun-result.mip_dual_bound)),expected_net_profit=float((q*px*x.expected_return-bf-sf-ef-slippage_cost).sum()))

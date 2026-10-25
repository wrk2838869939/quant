"""小账户新建仓分配原型：整手、不等权、现金可选，不从排名伪造收益预测。

只处理无既有持仓的新建仓批次。调用方须给出同一持有期的预期简单收益、
逐股风险金额上限和成本假设；尚未接入回测或实盘。目标为预期净利润，
不是完整风险效用模型，相关性风险必须由上游约束另行处理。
"""
import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp


def allocate_new_positions(candidates, *, cash, buy_rate, exit_rate,
                           minimum_fee, max_positions, forecast_kind):
    """联合决定各股买入股数；现金不产生虚构收益，负净优势可全部不买。

    candidates 的必需列：ts_code, price, expected_return, min_buy, buy_step,
    max_notional。成本率含哪些费用由调用方明确，出场按入场名义金额估计
    同口径费用（不是精确税务/未来成交预测，也不从当前现金预扣）。max_notional 为明确提供的风险
    上限，不允许用 cash / max_positions 自动生成。适用于新建仓，不能直接
    用来替换带旧仓/T+1约束的调仓器。
    """
    required = ['ts_code', 'price', 'expected_return', 'min_buy', 'buy_step', 'max_notional']
    if forecast_kind != 'expected_simple_return':
        raise ValueError('需要已校准的同持有期简单收益预测，不能把排名当收益')
    if not set(required).issubset(candidates.columns):
        raise ValueError('缺少显式收益、交易单位或风险上限输入')
    params = [cash, buy_rate, exit_rate, minimum_fee]
    if not np.isfinite(params).all() or min(params) < 0:
        raise ValueError('现金和成本必须有限且非负')
    if isinstance(max_positions, bool) or int(max_positions) != max_positions or max_positions < 0:
        raise ValueError('最大持仓数必须是非负整数上限')
    x = candidates[required].copy().reset_index(drop=True)
    if x.ts_code.isna().any() or x.ts_code.duplicated().any():
        raise ValueError('股票代码缺失或重复')
    vals = x[required[1:]].to_numpy(dtype=float)
    if not np.isfinite(vals).all() or (x.price<=0).any() or (x.max_notional<0).any() or (x.expected_return < -1).any():
        raise ValueError('价格、收益或风险上限非法')
    for col in ['min_buy','buy_step']:
        if (x[col]<=0).any() or (x[col] != np.floor(x[col])).any():
            raise ValueError('交易单位必须为正整数')
    if not len(x):
        return dict(orders=x.assign(shares=pd.Series(dtype=int)), cash_after=float(cash), expected_net_profit=0.)
    n = len(x)
    price, minimum, step = (x[c].to_numpy(float) for c in ['price','min_buy','buy_step'])
    cap = np.minimum(x.max_notional.to_numpy(float), cash)
    feasible = cap + 1e-9 >= minimum*price
    max_k = np.maximum(0, np.floor((cap/price-minimum)/step + 1e-10))
    # 每股4变量：是否持仓y、超过最小申报量的整数步数k、买入费b、预计退出费s。
    objective = np.zeros(4*n)
    q_matrix = np.zeros((n,4*n))
    for i in range(n):
        q_matrix[i,i] = minimum[i]
        q_matrix[i,n+i] = step[i]
    notional = q_matrix * price[:,None]
    objective -= (notional*x.expected_return.to_numpy(float)[:,None]).sum(axis=0)
    objective[2*n:] = 1
    rows, upper = [], []
    for i in range(n):
        row = np.zeros(4*n); row[n+i]=1; row[i]=-max_k[i]
        rows.append(row); upper.append(0.)
        rows.append(notional[i]); upper.append(cap[i])
        for offset, rate in [(2*n,buy_rate),(3*n,exit_rate)]:
            row = notional[i]*rate; row[offset+i]-=1
            rows.append(row); upper.append(0.)
            row=np.zeros(4*n); row[i]=minimum_fee; row[offset+i]=-1
            rows.append(row); upper.append(0.)
    row=notional.sum(axis=0); row[2*n:3*n]+=1
    rows.append(row); upper.append(cash)
    row=np.zeros(4*n); row[:n]=1
    rows.append(row); upper.append(max_positions)
    bounds_upper=np.r_[feasible.astype(float),max_k,
        np.maximum(cap*buy_rate,minimum_fee),np.maximum(cap*exit_rate,minimum_fee)]
    result=milp(objective,integrality=np.r_[np.ones(2*n),np.zeros(2*n)],
        bounds=Bounds(np.zeros(4*n),bounds_upper),
        constraints=LinearConstraint(np.array(rows),-np.inf,np.array(upper)),
        options={'mip_rel_gap':0.})
    if not result.success:
        raise RuntimeError('整手联合分配未得到最优解，禁止继续下单')
    shares=np.rint(q_matrix@result.x).astype(int)
    gross=shares*price
    buy_fee=np.where(shares>0,np.maximum(gross*buy_rate,minimum_fee),0.)
    exit_fee=np.where(shares>0,np.maximum(gross*exit_rate,minimum_fee),0.)
    pnl=gross*x.expected_return.to_numpy(float)-buy_fee-exit_fee
    cash_after=cash-float((gross+buy_fee).sum())
    if cash_after < -1e-6 or np.any(gross>x.max_notional+1e-6):
        raise RuntimeError('独立核算发现现金或风险上限越界')
    if np.any((shares>0)&((shares<minimum)|((shares-minimum)%step!=0))) or (shares>0).sum()>max_positions:
        raise RuntimeError('独立核算发现交易单位或持仓数越界')
    x['shares']=shares; x['gross']=gross; x['buy_fee']=buy_fee
    x['expected_exit_fee']=exit_fee; x['expected_net_profit']=pnl
    reasons=[]
    for i,row in x.iterrows():
        lot=row.price*row.min_buy
        if row.shares:
            reason='allocated'
        elif lot>row.max_notional+1e-9:
            reason='explicit_risk_cap_below_one_lot'
        elif lot+max(lot*buy_rate,minimum_fee)>cash+1e-9:
            reason='whole_account_cash_below_one_lot'
        else:
            reason='net_benefit_or_joint_constraints'
        reasons.append(reason)
    x['decision']=reasons
    return dict(orders=x,cash_after=max(0.,cash_after),expected_net_profit=float(pnl.sum()))

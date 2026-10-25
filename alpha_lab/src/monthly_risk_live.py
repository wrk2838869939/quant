"""月度市场风险的收盘适配；概率只使用冻结模型和截至当时的指数。"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from restore_quota import RestoreQuota
from rolling_review import decision_dates
from shadow_planner import ShadowError, fingerprint

ROOT = Path(__file__).resolve().parents[1]


def read_index_close(path):
    data = pd.read_parquet(path)
    if 'ts_code' not in data or set(data.ts_code) != {'000300.SH'}:
        raise ShadowError('风险模型只接受明确标记的沪深300单指数')
    dates = pd.to_datetime(data['date'] if 'date' in data else data['trade_date'])
    return pd.Series(data.close.to_numpy(), index=pd.DatetimeIndex(dates)).sort_index()


def check_index_receipt(receipt, date):
    item = receipt.get('market_index', {})
    if (item.get('code') != '000300.SH' or item.get('asof') != date or item.get('complete') is not True
            or not item.get('source') or not item.get('sha256')):
        raise ShadowError('缺沪深300当日完整性与来源回执')
    stamp = pd.Timestamp(item.get('downloaded_at'))
    if pd.isna(stamp) or stamp.tzinfo is None or stamp.tz_convert('Asia/Shanghai') < pd.Timestamp(date, tz='Asia/Shanghai')+pd.Timedelta(hours=15):
        raise ShadowError('指数获取时刻早于收盘或不可验证')


def load_model(cfg):
    spec = cfg['market_risk']
    path = ROOT / spec['model_path']
    if hashlib.sha256(path.read_bytes()).hexdigest() != spec['model_sha256']:
        raise ShadowError('冻结市场风险模型文件改变')
    if (spec['enter'] != .30245275630569923 or spec['release'] != .16239408013966727
            or spec['risk_cap'] != .5 or cfg['rank_buffer'] != 10 or cfg['rank_confirmations'] != 2):
        raise ShadowError('人工适配仅验证过冻结月度50%候选')
    return json.loads(path.read_text())


def reviews(calendar, cfg, date):
    spec = cfg['schedule']
    dates = pd.DatetimeIndex([d for d in calendar['sessions'] if spec['anchor'] <= d <= date])
    return decision_dates(dates, spec['kind'], spec['parameter'], spec['interval'], spec['anchor'])


def probability(features, model):
    x = np.array([features.get(k, np.nan) for k in model['features']], dtype=float)
    if not np.isfinite(x).all():
        raise ShadowError('市场风险特征未知，不能默认安全')
    z = (x - model['center']) / np.array(model['scale']) @ np.array(model['coefficients']) + model['intercept']
    return float(np.exp(-np.logaddexp(0., -z)))


def risk_history(index, date, calendar, cfg):
    """截断后计算，缺交易日不插值；不读取训练标签或未来价格。"""
    model = load_model(cfg)
    x = index.loc[index.index <= pd.Timestamp(date)].sort_index()
    if x.index.has_duplicates or not len(x) or x.index[-1] != pd.Timestamp(date):
        raise ShadowError('指数重复或未更新到决策日')
    expected = pd.DatetimeIndex([d for d in calendar['sessions'] if x.index.min() <= pd.Timestamp(d) <= pd.Timestamp(date)])
    if not x.index.equals(expected) or not np.isfinite(x).all() or not x.gt(0).all():
        raise ShadowError('指数交易日缺失或价格非法')
    ret = x / x.shift(1) - 1
    vol = ret.rolling(20, min_periods=20).std()
    frame = pd.DataFrame(dict(ret20=x/x.shift(20)-1,
        drawdown60=1-x/x.rolling(60, min_periods=60).max(), vol20=vol,
        vol_shock=ret.rolling(5, min_periods=5).std()/vol))
    result = []
    for day in reviews(calendar, cfg, date):
        if day not in frame.index or day <= pd.Timestamp(model['train_label_end']):
            raise ShadowError('指数历史不足或风险决策早于模型成熟')
        features = frame.loc[day].to_dict()
        result.append(dict(date=str(day.date()), features=features, probability=probability(features, model)))
    return result


def advance(state, bundle, calendar, cfg, positions):
    """连续复核与风险轮次持久化；首次空仓启动只回放风险，不虚构成交额度。"""
    model = load_model(cfg)
    if bundle.get('strategy') != cfg['strategy'] or bundle.get('strategy_config') != fingerprint(cfg):
        raise ShadowError('输入包不是本候选配置构建')
    dates = [str(d.date()) for d in reviews(calendar, cfg, bundle['asof'])]
    history = bundle.get('market_risk_history', [])
    if not dates or [r['date'] for r in history] != dates:
        raise ShadowError('缺完整月度风险复核历史')
    if bundle['signal_date'] != dates[-1] or bundle['is_review'] != (dates[-1] == bundle['asof']):
        raise ShadowError('输入复核日期偏离冻结日历')
    previous = state.get('monthly_risk')
    if previous is None and positions:
        raise ShadowError('首次接入候选必须为空仓，已有持仓需要明确迁移记录')
    risk = deepcopy(previous or dict(active=False, last_review=None, probability=None,
                                     quota=RestoreQuota().snapshot(), config=fingerprint(cfg)))
    if risk['config'] != fingerprint(cfg):
        raise ShadowError('账户策略配置改变，禁止静默迁移风险状态')
    quota = RestoreQuota(risk['quota'])
    if previous and risk['last_review'] not in dates:
        raise ShadowError('持久化风险日期不在当前日历')
    if previous and fingerprint([r for r in history if r['date'] <= risk['last_review']]) != risk['history_hash']:
        raise ShadowError('已冻结的风险复核历史改变，须核对数据修订')
    pending = [r for r in history if risk['last_review'] is None or r['date'] > risk['last_review']]
    if previous and any(r['date'] != bundle['asof'] for r in pending):
        raise ShadowError('漏过月度复核，须核对缺失期间账户，不能静默追补交易')
    for row in history:
        p = probability(row['features'], model)
        if not np.isfinite(row['probability']) or abs(p-row['probability']) > 1e-12:
            raise ShadowError('风险概率与冻结系数不一致')
    for row in pending:
        was = risk['active']
        p = row['probability']
        if p >= cfg['market_risk']['enter']:
            risk['active'] = True
        elif was and p <= cfg['market_risk']['release']:
            risk['active'] = False
        if not was and risk['active']:
            quota.start_episode()
        if was and not risk['active']:
            state['pending_sells'] = {c:v for c,v in state['pending_sells'].items() if v['reason'] != 'portfolio_cash_cap'}
        risk.update(last_review=row['date'], probability=p)
    for code in list(quota.data['shares']):
        if code not in positions:
            quota.clear(code)
    risk['quota'] = quota.snapshot()
    risk['history_hash'] = fingerprint(history)
    state['monthly_risk'] = risk
    return cfg['market_risk']['risk_cap'] if risk['active'] else 1., quota


def apply_fill(state, order, fill, balance):
    """调用方的同一SQLite事务内更新额度；取消/草稿/核对均不调用。"""
    risk = state['memory'].get('monthly_risk')
    if risk is None:
        return
    quota = RestoreQuota(risk['quota'])
    reason, code = order['reason'], order['ts_code']
    if reason == 'targeted_restore':
        if risk['active'] or code not in state['book_positions']:
            raise ShadowError('恢复买入的状态或原持仓已失效')
    quota.fill(fill['execution_id'], code, order['side'], int(fill['shares']), reason,
               order.get('restore_epoch', quota.data['epoch']))
    if not balance or (order['side'] == 'buy' and reason != 'targeted_restore'):
        quota.clear(code)
    risk['quota'] = quota.snapshot()

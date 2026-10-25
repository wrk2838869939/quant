"""当日全A截面重建冻结策略输入；模型参数固定，次日行情不参与。"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from conditional_hold import build_conditional_holds
from patient_trend import trend_state
from rolling_review import decision_dates
from trend_entry import entry_features
from shadow_planner import ShadowError, fingerprint, validate_calendar


def build_bundle(raw, market, basics, st, actions, date, calendar, config, model,
                 held_codes, receipt, index_close=None):
    """只消费date及以前的特征和已公告状态，股票全集来自输入全A截面。"""
    stamp = pd.Timestamp(date)
    next_date = validate_calendar(calendar, date)
    if (receipt.get('asof') != date or receipt.get('complete') is not True
            or not receipt.get('source') or not receipt.get('downloaded_at')):
        raise ShadowError('全市场日线缺有来源的完整性回执')
    downloaded = pd.Timestamp(receipt['downloaded_at'])
    closing = config['execution_session'] == 'closing_auction'
    if closing:
        from execution_clock import check_proxy
        check_proxy(receipt, date)
        next_date = date
    if not closing and (downloaded.tzinfo is None or downloaded.tz_convert('Asia/Shanghai') < pd.Timestamp(date, tz='Asia/Shanghai') + pd.Timedelta(hours=15)):
        raise ShadowError('数据回执早于收盘或无时区')
    raw = raw[raw.date <= stamp].copy()
    market = market[market.date <= stamp].copy()
    if raw.duplicated(['date','ts_code']).any() or market.duplicated(['date','ts_code']).any():
        raise ShadowError('行情主键重复')
    if raw.date.max() != stamp or market.date.max() != stamp:
        raise ShadowError('行情尚未更新到指定决策日')
    if len(raw[raw.date.eq(stamp)]) != receipt['expected_rows']:
        raise ShadowError('当日全市场行数与下载回执不符')
    wide = {key: raw.pivot(index='date', columns='ts_code', values=key).sort_index()
            for key in ['hfq_close','total_mv_yuan','turnover_rate']}
    close = wide['hfq_close']
    known_calendar = pd.DatetimeIndex([d for d in calendar['sessions'] if d <= date])
    if not close.index[(close.index >= known_calendar.min())].equals(known_calendar):
        raise ShadowError('价格观察日期与所提供交易日历不一致')
    trend_spec = config['trend']
    medium, long = trend_spec['medium_window'], trend_spec['long_window']
    features = entry_features(close, wide['total_mv_yuan'], wide['turnover_rate'], medium=medium, long=long)
    _, trend = trend_state(close, trend_spec)
    observable = (close.gt(0) & close.shift(medium).gt(0) & close.shift(long).gt(0)
                  & close.where(close>0).rolling(long,min_periods=long).mean().notna())
    schedule = config['schedule']
    reviews = decision_dates(close.index[close.index >= pd.Timestamp(schedule['anchor'])],
        schedule['kind'], schedule['parameter'], schedule['interval'], schedule['anchor'])
    if not len(reviews):
        raise ShadowError('无已完成的复核日')
    signal_date = reviews[-1]
    if model['year'] != signal_date.year or model['train_label_end'] >= f'{model["year"]}-01-01':
        raise ShadowError('缺对应年度的成熟标签模型，不自动沿用过期模型')
    hold_spec = config['hold']
    variants, _ = build_conditional_holds(close, trend, observable, reviews, hold_spec)
    frame = pd.DataFrame({key:value.loc[signal_date] for key,value in features.items()})
    if basics.ts_code.duplicated().any():
        raise ShadowError('证券主表代码重复')
    if config['strategy'] == 'monthly_predicted_risk':
        unknown = set(raw.loc[raw.date.ge(stamp-pd.Timedelta(days=200)), 'ts_code']) - set(basics.ts_code)
        if unknown:
            raise ShadowError('证券主表缺近期报价股票，不能静默删掉候选：'+','.join(sorted(unknown)))
    b = basics.set_index('ts_code')
    listing = pd.to_datetime(b.list_date).reindex(close.columns)
    delisting = pd.to_datetime(b.delist_date).reindex(close.columns)
    if len(st) and (not {'start','end'}.issubset(st) or not {'ann_date','known_date'}.intersection(st.columns)):
        raise ShadowError('ST状态缺公告时点')
    def st_at(day, asof):
        if st.empty:
            return set()
        known = pd.to_datetime(st['ann_date'] if 'ann_date' in st else st['known_date'])
        if known.isna().any():
            raise ShadowError('ST公告日期未知')
        return set(st.loc[known.le(asof) & st.start.le(day) & (st.end.isna() | st.end.ge(day)), 'ts_code'])
    active_st = st_at(signal_date, signal_date)
    eligible = (signal_date-listing).dt.days.ge(config['min_listing_days']) & (delisting.isna() | delisting.gt(signal_date))
    eligible &= np.isfinite(frame).all(axis=1)
    eligible.loc[eligible.index.isin(active_st)] = False
    ranks = frame.loc[eligible].rank(pct=True)-.5
    main = ranks.index.str.match(r'^(?:(?:000|001|002|003)\d{3}\.SZ|(?:600|601|603|605)\d{3}\.SH)$')
    chosen = ranks.index[main]
    scores = (-ranks.loc[chosen, 'turnover']).where(trend.loc[signal_date, chosen]).dropna()
    scores = scores.sort_values(ascending=False, kind='stable').head(config['candidate_pool'])
    states = variants['hybrid'].loc[signal_date]
    extra = {}
    if config['strategy'] == 'monthly_predicted_risk':
        from selection_stability import buffered_holds
        from monthly_risk_live import risk_history, check_index_receipt
        from portfolio_cash_state import basket_features
        if index_close is None:
            raise ShadowError('新候选需要截至决策日的沪深300指数收盘历史')
        if not closing:
            check_index_receipt(receipt, date)
        # 排名退出须用全部合格股票，不能用top20截断后的名次或只看当前持仓。
        score_history = pd.DataFrame(float('nan'), index=reviews, columns=close.columns)
        for day in reviews:
            f = pd.DataFrame({key:value.loc[day] for key,value in features.items()})
            ok = (day-listing).dt.days.ge(config['min_listing_days']) & (delisting.isna() | delisting.gt(day))
            ok &= np.isfinite(f).all(axis=1)
            ok.loc[ok.index.isin(st_at(day, day))] = False
            rank = f.loc[ok, 'turnover'].rank(pct=True)-.5
            main_codes = rank.index[rank.index.str.match(r'^(?:(?:000|001|002|003)\d{3}\.SZ|(?:600|601|603|605)\d{3}\.SH)$')]
            score_history.loc[day, main_codes] = (-rank.loc[main_codes]).where(trend.loc[day, main_codes])
        adjusted, _, exit_mask = buffered_holds(score_history, variants['hybrid'].loc[reviews],
                                        config['rank_buffer'], config['rank_confirmations'], config['top_n'])
        states = adjusted.loc[signal_date]
        scores = score_history.loc[signal_date].dropna().sort_values(ascending=False, kind='stable').head(config['candidate_pool'])
        scores = scores.mask(exit_mask.loc[signal_date].reindex(scores.index)).dropna()
        recovery = basket_features(close)
        stock_recovery = {}
        for code in held_codes:
            ret = recovery['ret20'].at[stamp, code]
            stock_recovery[code] = dict(known=bool(recovery['known'].at[stamp, code]),
                weak=bool(recovery['weak'].at[stamp, code]), ret20=float(ret) if np.isfinite(ret) else None)
        extra = dict(strategy=config['strategy'], strategy_config=fingerprint(config),
            market_risk_history=risk_history(index_close, date, calendar, config), stock_recovery=stock_recovery)
    codes = sorted(set(scores.index) | set(held_codes))
    today = market[market.date.eq(stamp)].set_index('ts_code')
    if not set(held_codes).issubset(today.index):
        raise ShadowError('持仓缺当日报价，需要人工处理停牌/缺数状态')
    clean = close.where(close>0)
    rv, rd = config['risk_windows']['volatility'], config['risk_windows']['drawdown']
    vol = clean.pct_change(fill_method=None).rolling(rv,min_periods=rv).std().loc[signal_date]
    dd = (clean/clean.rolling(rd,min_periods=rd).max()-1).loc[signal_date]
    risk = {}
    unknown = []
    for code in codes:
        x = np.array([vol.get(code,np.nan), dd.get(code,np.nan)])
        if not np.isfinite(x).all():
            risk[code] = model['uniform_discount']
            unknown.append(code)
        else:
            prediction = model['intercept'] + ((x-np.array(model['center']))/np.array(model['scale'])) @ np.array(model['coefficients'])
            risk[code] = float(min(1., model['reference_risk']/max(-prediction,model['reference_risk'])))
    # 保留父策略未知风险回退的数值，但生产计划明确阻断未知持仓/候选。
    if unknown:
        raise ShadowError('实际候选或持仓风险特征未知：' + ','.join(unknown))
    blocked = st_at(pd.Timestamp(next_date), stamp)
    if len(actions):
        if not {'known_date','ex_date','ts_code'}.issubset(actions):
            raise ShadowError('公司行动缺已知时点')
        blocked |= set(actions.loc[actions.known_date.le(stamp) & actions.ex_date.eq(pd.Timestamp(next_date)), 'ts_code'])
    if 'delist_known_date' in b:
        known = pd.to_datetime(b.delist_known_date)
        blocked |= set(b.index[known.le(stamp) & pd.to_datetime(b.delist_date).le(pd.Timestamp(next_date))])
    today_st = st_at(stamp, stamp)
    return dict(schema_version=1, asof=date, signal_date=str(signal_date.date()),
        is_review=stamp==signal_date, eod_complete=not closing, input_kind='intraday_proxy' if closing else 'eod', model_year=model['year'],
        model_cutoff=model['train_label_end'], expected_return=model['expected_return'],
        signals=scores.to_dict(), holds={c:bool(states[c]) if pd.notna(states.get(c)) else None for c in held_codes},
        risk_discounts=risk, st_codes=sorted(today_st), blocked_next_day=sorted(blocked),
        quotes={c:dict(date=date,close=float(today.at[c,'close']),volume=float(today.at[c,'vol_share'])) for c in codes if c in today.index},
        eligible={c:c not in active_st | today_st | blocked for c in codes},
        provenance=dict(receipt=receipt, model=fingerprint(model), full_universe=len(close.columns),
                        source_vintage_externally_certified=False), **extra)

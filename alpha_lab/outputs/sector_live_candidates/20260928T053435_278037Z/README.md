# 最新板块与个股扫描

请求日期：2026-09-28；当日日线未完成时使用最近完整交易日。

截至 2026-09-24 收盘。只更新数据，事件门槛及两个网络模型/评分阈值均未调整。

## 结果

{
  "requested_date": "2026-09-28",
  "asof": "2026-09-24",
  "watched_sectors": 4,
  "stocks_checked": 52,
  "recent_confirmations": 0,
  "exploratory_stock_rules_pass": 0,
  "combined_candidates": 0,
  "stock_rules_note": "复用旧个股量价/趋势规则，仅探索性检查，尚无本次模型下的选股回测验证"
}

## 近期确认事件

date,sector_code,classification_version,known_members,prior_known_members,total_mv_yuan,total_mv_yuan_members,circ_mv_yuan,circ_mv_yuan_members,vol_share,vol_share_members,amount_yuan,amount_yuan_members,turnover_members,turnover_pct,up_count,down_count,flat_count,breadth_members,breadth_unknown,up_share,up_down_ratio,advancing_share,matched_cap_members,matched_cap_change_yuan,matched_cap_return,added_members,removed_members,member_add_cap_yuan,member_remove_cap_yuan,missing_cap_effect_yuan,observed_cap_change_yuan,index_members,index_member_coverage,index_valid,index_status,index_close,index_open,index_high_bound,index_low_bound,equal_index_close,index_return,equal_return,chain_return,max_weight,weight_hhi,segment_id,member_request_failed,event_id,platform_start,platform_end,platform_start_i,platform_low,platform_high,platform_median,platform_total_cap,platform_cap_members,breakout_date,breakout_i,reference_high,reference_ready,peak_date,peak_i,peak_price,trend_at_peak,trend_return_at_peak,trend_slope_at_peak,support_at_peak,rally_confirmed,rally_date,first_pullback_date,first_candidate_date,fast_fall_seen,added_members_since_platform,removed_members_since_platform,episode_age,rally_return,drawdown,pullback_days,return_10d,post_peak_fast_return,previous_loss_mean,recent_loss_mean,slowed,support_broken,near_reference,reference_distance,observed_cap_vs_platform,turnover_vs_previous20,pullback_gate,candidate,first_pullback,first_candidate,stage,watch,trigger,trigger_recent,support_recent,confirmed,first_watch,first_trigger,first_confirmed,window,priority,accepted,merged_event_id,source_event_id,name


## 今日板块观察池

name,sector_code,watch,trigger,confirmed,rally_return,drawdown,reference_distance
农产品加工,801012.SI,True,True,False,0.22430088014237,-0.140404920960482,0.0363183241502551
渔业,801015.SI,True,False,False,0.3704620453152627,-0.1352091429210762,0.1267307371294324
贵金属,801053.SI,True,True,False,0.3041620840703452,-0.1925263881520006,-0.0473162606037098
酒店餐饮,801219.SI,True,True,False,0.1698981044008447,-0.1218984212022903,-0.0403971480782598


## 个股通过情况

sector_code,sector_name,ts_code,name,membership_source,close,total_mv_yuan,turnover_today,return_5d,leader_rally_return,leader_rank,turnover_expansion,turnover_contraction,loss_slowdown,shape_pass,trend_pass,stock_rules_pass,sector_confirmed,model_pass_count,recent_signal_date,recent_low_reference,candidate,blockers


个股尚未经过本次板块模型下的选股回测。本次只复用此前五日放量下跌后缩量减速、峰值前上行趋势和先涨后回落规则做探索性检查；历史龙头表现只用于排序。
候选还要求板块今天保持确认，近五个交易日出现新确认且至少一个既有模型通过；这项实时联合使用方式尚未单独回测，不能套用历史事件平均收益。
未通过个股不作为买入候选；stock_diagnostics_all.csv 保留每项原因。所有信息是收盘后可知，下一交易日开盘前仍需核对报价与成交条件。
历史行业分类沿用冻结补充资料；对观察板块另外在线核对当前成员，差异记录在 membership_crosscheck.csv，不将当前名单回填历史。
本次没有下单，也没有据此建立10万元组合或验证账户最大回撤30%。

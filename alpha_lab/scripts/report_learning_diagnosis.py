#!/usr/bin/env python3
"""汇总共同归因；只读既有锁定结果，不重新选策略或参数。"""
from pathlib import Path
import argparse,json,hashlib,sys,shutil
import numpy as np
import pandas as pd
import yaml
ROOT=Path(__file__).resolve().parents[1];PROJECT=ROOT.parent
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts'),str(PROJECT/'datahub/releases/selection_v1_20260920/src')]
from diagnose_learning_strategies import load_data,diagnostic_model_predictions


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);args=ap.parse_args();run=args.run.resolve()
    m=json.loads((run/'manifest.json').read_text());assert m['status']=='complete'
    digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    assert all(digest(PROJECT/p)==h for p,h in m['source_hashes'].items())
    summary=pd.read_csv(run/'diagnosis_summary.csv');q=pd.read_csv(run/'same_model_train_future_ic.csv');s=pd.read_csv(run/'selection_stability.csv')
    a=pd.read_csv(run/'account_comparisons.csv')
    future=q[q.future_ic_days>0].copy();future['ic_change']=future.future_daily_ic-future.training_daily_ic
    future.groupby('family')[['training_daily_ic','future_daily_ic']].mean().to_csv(run/'ic_family_means.csv')
    tail=[];reload_rows=[]
    for source in m['input_runs']:
        old=PROJECT/source;extension='learning_extensions' in source
        plan=yaml.safe_load((old/'sources'/('learning_extensions_v1.yaml' if extension else 'learning_strategy_search_v1.yaml')).read_text())
        data,_=load_data(old,plan)
        choices=json.loads((old/'all_locked_choices.json').read_text())
        for family in plan['families']:
            folder=old/'outer'/family;signals=pd.read_parquet(folder/'signals.parquet');schedule=pd.read_parquet(folder/'top_n_schedule.parquet').iloc[:,0]
            for c in [x for x in choices if x['family']==family and x['selected'] is not None]:
                p=c['selected']['params'];year=c['outer_year'];label=data['labels'][p['horizon']]
                # 对诊断所用首个外层模型再次核对真实当日预测，确认缺失值填补与原运行一致。
                if family not in ['trade_filter','market_gate']:
                    import joblib
                    fit_path=next((old/'predictions'/f'outer_{year}'/family).glob('*/fits.json'))
                    fit=next(x for x in json.loads(fit_path.read_text()) if x['status']=='ok')
                    date=pd.Timestamp(fit['fit_date']);at=data['dates'].get_loc(date)
                    saved=joblib.load(fit_path.parent/'models'/fit['model_file'])
                    actual=diagnostic_model_predictions(saved,data,p,[at],extension)[0]
                    expected=pd.read_parquet(fit_path.parent/'return_predictions.parquet')
                    if family=='dual':
                        from learning_search import score_predictions
                        expected=score_predictions(expected,pd.read_parquet(fit_path.parent/'risk_predictions.parquet'),p)
                    np.testing.assert_allclose(actual,expected.loc[date],rtol=1e-12,atol=1e-12,equal_nan=True)
                    reload_rows.append(dict(family=family,year=year,passed=True))
                for date,score in signals[signals.index.year==year].iterrows():
                    at=data['dates'].get_loc(date)
                    if pd.isna(label['end'][at]) or label['end'][at]>np.datetime64(f'{year}-12-31'):continue
                    ranked=score.dropna().sort_values(ascending=False,kind='stable').head(int(schedule.loc[date]))
                    if ranked.empty:continue
                    ix=data['codes'].get_indexer(ranked.index);returns=label['returns'][at]
                    chosen=returns[ix];universe=returns[data['allowed'][at]]
                    if not np.isfinite(chosen).any() or not np.isfinite(universe).any():continue
                    tail.append(dict(family=family,year=year,date=str(date.date()),horizon=p['horizon'],top_n=int(schedule.loc[date]),
                                     selected_target_return=float(np.nanmean(chosen)),eligible_target_return=float(np.nanmean(universe)),
                                     target_excess=float(np.nanmean(chosen)-np.nanmean(universe)),selected_stocks=len(chosen),valid_selected_targets=int(np.isfinite(chosen).sum())))
    tails=pd.DataFrame(tail);tails.to_csv(run/'selection_tail_diagnostics.csv',index=False)
    tails.groupby('family').agg(signal_days=('date','size'),mean_price_target=('selected_target_return','mean'),
                               mean_eligible_target=('eligible_target_return','mean'),mean_target_excess=('target_excess','mean'),
                               selected_stocks=('selected_stocks','sum'),valid_targets=('valid_selected_targets','sum')).to_csv(run/'selection_tail_summary.csv')
    pd.DataFrame(reload_rows).to_csv(run/'diagnostic_model_replays.csv',index=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(12,6),constrained_layout=True);x=np.arange(len(summary));width=.25
    for offset,col,label in [(-1,'net_return','Learned, costs'),(0,'zero_cost_return','Learned, zero costs'),(1,'low_turn20_net_return','Matched low-turn, costs')]:
        ax.bar(x+offset*width,summary[col]*100,width,label=label)
    ax.axhline(0,color='black',lw=.6);ax.set_xticks(x,summary.family,rotation=35,ha='right');ax.set_ylabel('2021-2023 cumulative return (%)');ax.legend()
    fig.savefig(run/'cost_and_matched_control.png',dpi=150);plt.close(fig)
    names={'factor_weight':'因子权重','linear':'线性预测','lgbm':'LightGBM','dual':'收益风险双预测','neural':'MLP',
           'state_experts':'状态专家','adaptive_ic':'动态权重','trade_filter':'交易过滤','multi_horizon':'多周期','correlation_experts':'相关性专家','market_gate':'市场门控'}
    gross_loss=int((summary.zero_cost_return<0).sum());drag=int(((summary.zero_cost_return>0)&(summary.net_return<0)).sum())
    lowturn_wins=int((summary.low_turn20_net_return>summary.net_return).sum())
    lines=['# 学习策略共同归因（2026-09-20）','',f"运行：`{run.relative_to(PROJECT)}`。复用前两批11类的已锁定参数、预测和交易；未重新选参。",'',
           f"结论：{gross_loss}/11类在零成本重放下仍亏损；{drag}/11类零成本为正、原费用下转负。预设低换手规则在{lowturn_wins}/11套相同调仓日与持股数配置下净收益更高。费用不是全部原因，预测稳定性及信号到小组合的映射也需要解释。",'',
           '| 策略 | 原净收益 | 零成本重放 | 同配置低换手净收益 |', '|---|---:|---:|---:|']
    for r in summary.itertuples():lines.append(f'| {names[r.family]} | {r.net_return:.2%} | {r.zero_cost_return:.2%} | {r.low_turn20_net_return:.2%} |')
    lines+=['','零成本账户重放仍保留现金、整数手、涨跌停、公司行动和资格约束，只取消配置交易费用、最低佣金和滑点。现金路径及最终股数可能改变，因此两种收益之差不能直接等同于手续费合计，也不能把零成本结果视为可实现收益。',
            '规则对照继承各策略预先锁定的无可行年度，但不继承模型主动过滤造成的空仓；对照均从10万元连续运行。低换手不是从本次多个规则中再择优拼接的策略；它在本轮分析前已明确列入对照。历史观察与多次试验意味着这里没有独立显著性结论。', '',
            '## 训练到后续预测','',
            f"对每个有参数年度的首个外层模型做训练期/下一重训段比较，共{len(q)}个模型；{len(future)}个有成熟未来标签，另外{len(q)-len(future)}个因重训段短于标签跨度而无法评价。其中{int((future.ic_change<0).sum())}/{len(future)}个未来IC低于训练IC，{int((future.future_daily_ic<0).sum())}/{len(future)}个未来IC为负。",
            '下表训练/未来均值只用两侧均可评价的同一组模型。这是描述性首段诊断，不能外推成全年IC或模型寿命，短段重叠标签也不能当作独立样本。交易过滤与市场门控的固定宿主原始IC不算其学习技能，未混入此表。', '',
            '| 类别 | 训练期IC均值 | 后续段IC均值 |', '|---|---:|---:|']
    for family,g in future.groupby('family',sort=False):lines.append(f'| {names[family]} | {g.training_daily_ic.mean():.4f} | {g.future_daily_ic.mean():.4f} |')
    lines+=['','`selection_tail_diagnostics.csv`另列每次实际调仓信号前N名的未来价格标签均值、同日可选池均值及差值。该诊断不扣费、不模拟实际成交、各日持有期可能重叠，且剔除未来价格路径缺失者；仅用于检查全截面IC是否传导至少量候选，不是新增策略收益。', '',
            '## 选参稳定性','',
            f"33个家族/年度选择中{int(s.selected.sum())}个有可行参数；其中{int(((s.neighbor_count.fillna(0)==0)&s.selected).sum())}个选中点在旧搜索的严格邻域定义下没有已评估邻点。该事实说明邻域稳定性证据不足，不能据此断言参数不稳定，也不能把单点最佳当成平台。",
            '`selection_stability.csv`保存可行率、选中参数的内层目标、全候选目标中位数及邻域覆盖。新的CNN/HMM预先增加完整二维切片；本诊断未据外层表现修改旧参数。', '',
            '## 工程核验与范围','',
            f"11个原账户逐表重放一致。88套账户通过独立现金/持股/T+1/涨跌停核验，涉及{m['trades_checked']}笔交易，资金最大误差{m['cash_max_error']:.3g}元。用于训练/未来IC的{len(reload_rows)}个保存模型亦重新核对拟合当日预测。",
            '首次诊断因归档市场上下文保留暖机NaN、加载器未按原训练约定填零而中止；修复仅发生在诊断加载器，随后完整重跑。本报告不采用中止运行的分析结果，旧模型、行情和回测结果均未改动。',
            '数据仍为含全期质量隔离偏差的93股工程样本，2021–2023已观察。毛收益、对照收益或首段IC均不能证明新的可实盘优势。', '',
            '复现：通过冻结启动器运行 `alpha_lab/scripts/diagnose_learning_strategies.py`；再以输出目录作为本脚本的 `--run`。','']
    (run/'REPORT.md').write_text('\n'.join(lines));(ROOT/'docs/LEARNING_DIAGNOSIS_RESULTS_20260920.md').write_text('\n'.join(lines)+f'\n[全部过程数据](../outputs/learning_diagnosis/{run.name}/REPORT.md)\n')
    shutil.copy2(Path(__file__),run/'sources'/Path(__file__).name)
    validation=dict(status='complete',archived_replays=11,accounts=88,diagnostic_model_replays=len(reload_rows),gross_loss_families=gross_loss,cost_sign_reversals=drag,
                    matched_low_turn_outperforms=lowturn_wins,first_model_pairs=len(q),valid_future_pairs=len(future),future_ic_declines=int((future.ic_change<0).sum()),
                    script_sha256=digest(Path(__file__)),parameter_selection=False)
    (run/'reporting_manifest.json').write_text(json.dumps(validation,indent=2));print(json.dumps(validation,indent=2))


if __name__=='__main__':main()

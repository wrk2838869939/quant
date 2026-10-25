"""条件退出独立核验：冻结母集、五政策矩阵/前缀、30账户和两套原对照。"""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts')]
from verify_recent_low_turnover import (digest, require, close, verify_case, verify_copy_provenance,
    prepare_execution_evidence, verify_market_fills, performance_report)
from verify_entry_guard_research import csv, identical, verify_first_exit_requests
from verify_hold_sensitivity import verify_reverse
VARIANTS = ['baseline','persist10','accelerated','patient','hybrid']


def check_inputs(run, m):
    for name, expected in m['source_hashes'].items():
        require(digest(ROOT/name) == expected and digest(run/'sources'/name) == expected, '源码漂移: '+name)
    for name, expected in m['input_hashes'].items():
        require(digest(run/name) == expected, '输入漂移: '+name)
    require(digest(run/'PROTOCOL.md') == m['protocol_sha256'], '协议漂移')
    parent = Path(m['parent']); base = Path(m['baseline_parent'])
    for name in ['daily_close.parquet','daily_trend.parquet','daily_observable.parquet', 'execution_predictions.parquet',
                 'execution/low_turnover_signals.parquet','execution/low_turnover_point_forecasts.parquet','execution/monthly_hold_allowed.parquet']:
        require(digest(run/name) == digest(parent/name), '母集输入改变: '+name)
    for name in ['execution_predictions.parquet','execution/low_turnover_signals.parquet',
                 'execution/low_turnover_point_forecasts.parquet','execution/monthly_hold_allowed.parquet']:
        require(digest(run/name) == digest(base/name), '修复recent的入口/原hold改变: '+name)
    proof = json.loads((parent/'independent_hold_matrix_verification.json').read_text())
    require(proof['status'] == 'complete', '母集矩阵未通过独立核验')
    for name in ['daily_close.parquet','daily_trend.parquet','daily_observable.parquet']:
        require(digest(run/name) == proof['input_hashes'][name], '母集核验后的日线发生变化')
    require(digest(Path(m['projection_source'])) == m['projection_sha256'] == proof['projection_sha256'], '原价格投影漂移')
    spec = yaml.safe_load((run/'sources/configs/conditional_hold_research.yaml').read_text())
    wanted = dict(primary_variant='hybrid',medium_window=60,long_window=120,baseline_confirmations=2,
        patient_confirmations=3,persistence_window=20,persistent_weak_days=10,
        variants=dict(baseline=dict(accelerate='none',patient=False),persist10=dict(accelerate='all',patient=False),
          accelerated=dict(accelerate='broken',patient=False),patient=dict(accelerate='none',patient=True),
          hybrid=dict(accelerate='broken',patient=True)))
    require(spec == wanted == m['parameters'] and m['refit'] is False and m['fresh_holdout'] is False, '规则或研究口径变化')
    return spec, digest(parent/'independent_hold_matrix_verification.json')


def verify_matrices(run, m, spec, parent_proof):
    prices, trend, obs = [pd.read_parquet(run/(n+'.parquet')) for n in ['daily_close','daily_trend','daily_observable']]
    signals = pd.read_parquet(run/'execution/low_turnover_signals.parquet'); reviews = signals.index
    original = pd.read_parquet(run/'execution/monthly_hold_allowed.parquet')
    codes = sorted(set().union(*(set(row.dropna().sort_values(ascending=False,kind='stable').head(20).index) for _,row in signals.iterrows())))
    require(codes == m['execution_universe'] == list(prices.columns), '172代码不是原Top20并集')
    require(reviews.equals(original.index) and reviews.isin(prices.index).all(), '复核日历改变')
    def frame(a):
        return pd.DataFrame(a,index=reviews,columns=prices.columns)
    a = prices.where(prices.gt(0)).to_numpy(float); known = obs.to_numpy(bool); weak = known & ~trend.to_numpy(bool)
    indices = prices.index.get_indexer(reviews); n, width = len(reviews), len(codes)
    count = np.full((n,width),np.nan); coverage=count.copy(); streak=np.zeros((n,width),int)
    lr = count.copy(); numpy_ma=count.copy()
    for j,i in enumerate(indices):
        if i>=19:
            count[j]=weak[i-19:i+1].sum(axis=0);coverage[j]=known[i-19:i+1].sum(axis=0)
        streak[j]=np.where(weak[i],(streak[j-1] if j else np.zeros(width,int))+1,0)
        with np.errstate(invalid='ignore',divide='ignore'):
            lr[j]=a[i]/a[i-120]-1
        numpy_ma[j]=np.mean(a[i-119:i+1],axis=0)
    # 数值均值以numpy独立复算；分类严格复现冻结pandas比较算子，不加策略容差。
    native_ma=prices.where(prices.gt(0)).rolling(120,min_periods=120).mean().loc[reviews].to_numpy()
    np.testing.assert_allclose(numpy_ma,native_ma,rtol=5e-15,atol=1e-12,equal_nan=True)
    at=a[indices]; k=known[indices]; w=weak[indices]
    intact=k&(lr>0)&(at>native_ma);broken=k&(lr<=0)&(at<=native_ma)
    alt_intact=k&(lr>0)&(at>numpy_ma);alt_broken=k&(lr<=0)&(at<=numpy_ma)
    boundary=[]
    for i,j in np.argwhere((intact!=alt_intact)|(broken!=alt_broken)):
        require(abs(at[i,j]-native_ma[i,j]) <= np.finfo(float).eps*max(1.,abs(at[i,j]))*8, '非机器精度分类差异')
        boundary.append(dict(date=str(reviews[i].date()),ts_code=codes[j],price=float(at[i,j]),
                            numpy_ma=float(numpy_ma[i,j]),pandas_ma=float(native_ma[i,j]),
                            frozen_intact=bool(intact[i,j]),numpy_intact=bool(alt_intact[i,j]),
                            frozen_broken=bool(broken[i,j]),numpy_broken=bool(alt_broken[i,j])))
    states=np.full((n,width),'unknown',object);states[k&~w]='strong';states[w]='mixed'
    states[w&intact]='long_intact_weak';states[w&broken]='long_broken_weak'
    evidence=dict(monthly_observable=k,monthly_weak=w,monthly_failure_streak=streak,daily_weak_count=count,
        daily_known_count=coverage,long_intact=intact,long_broken=broken,weak_regime=states,
        long_return=lr,long_ma=native_ma,long_ma_gap=at/native_ma-1)
    for name,array in evidence.items():
        pd.testing.assert_frame_equal(frame(array),pd.read_parquet(run/(name+'.parquet')),check_exact=True)
    persistent=w&(coverage==20)&(count>=10)
    expected={};numerical_changes={}
    for name,rule in spec['variants'].items():
        threshold=np.where(intact&w,3,2) if rule['patient'] else np.full((n,width),2)
        exits=streak>=threshold
        if rule['accelerate']=='all':exits|=persistent
        if rule['accelerate']=='broken':exits|=persistent&broken
        expected[name]=frame(np.where(k,(~exits).astype(float),np.nan))
        pd.testing.assert_frame_equal(expected[name],pd.read_parquet(run/'hold_variants'/(name+'.parquet')),check_exact=True)
        alt_threshold=np.where(alt_intact&w,3,2) if rule['patient'] else np.full((n,width),2)
        alt_exits=streak>=alt_threshold
        if rule['accelerate']=='all':alt_exits|=persistent
        if rule['accelerate']=='broken':alt_exits|=persistent&alt_broken
        numerical_changes[name]=int(((alt_exits!=exits)&k).sum())
    for name in ['baseline','persist10']:
        pd.testing.assert_frame_equal(expected[name],pd.read_parquet(Path(m['parent'])/'hold_variants'/(name+'.parquet')),check_exact=True)
    orders=[]
    for left,right in [('persist10','accelerated'),('accelerated','baseline'),('baseline','patient'),('accelerated','hybrid'),('hybrid','patient')]:
        violations=int(((expected[left]>expected[right])&frame(k)).sum().sum());require(violations==0,'允许度偏序违反')
        orders.append(dict(left=left,right=right,known_cells=int(k.sum()),violations=violations))
    from conditional_hold import build_conditional_holds
    from run_hold_sensitivity import effective_hold
    prefix_checks=0
    for index,cutoff in enumerate(reviews):
        partial,_=build_conditional_holds(prices.loc[:cutoff],trend.loc[:cutoff],obs.loc[:cutoff],reviews[:index+1],spec)
        for name in VARIANTS:
            pd.testing.assert_frame_equal(partial[name],expected[name].loc[:cutoff],check_exact=True);prefix_checks+=1
    for name in VARIANTS:
        actual=effective_hold(original,expected[name]);wanted=original.copy();wanted.loc[:,codes]=expected[name]
        pd.testing.assert_frame_equal(actual,wanted,check_exact=True)
        if name=='baseline':pd.testing.assert_frame_equal(actual,original,check_exact=True)
    numerical=dict(boundaries=boundary,changed_hold_cells=numerical_changes,
        parent_daily_boundary_sensitivity=str(Path(m['parent'])/'matrix_numerical_sensitivity.json'),
        note='此处只审计月度长期状态的numpy/pandas边界；每日原趋势沿用已核验冻结母集')
    (run/'independent_conditional_numerical.json').write_text(json.dumps(numerical,ensure_ascii=False,indent=2)+'\n')
    require(not any(numerical_changes.values()),'分类机器精度差异改变hold许可，需单独处理')
    report=dict(status='complete',reviews=n,codes=width,variant_cells=n*width*5,prefix_checks=prefix_checks,
        reference_variants_exact=['baseline','persist10'],order_checks=orders,full_schema_preserved=True,
        numerical_boundary_cells=len(boundary),numerical_report_sha256=digest(run/'independent_conditional_numerical.json'),
        parent_matrix_proof_sha256=parent_proof,input_hashes=m['input_hashes'],verifier_sha256=digest(Path(__file__)))
    (run/'independent_conditional_matrix.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='input_hashes'},ensure_ascii=False),flush=True)
    return report


def comparisons(run,m):
    out=run/'execution';reverse=[];references=[];evidence=[];deferred=[]
    for variant in VARIANTS:
        left,right=out/('cold_'+variant),out/('reverse_'+variant);evidence.append(verify_reverse(right))
        names={p.name for p in left.glob('*.csv')};require(names=={p.name for p in right.glob('*.csv')},'倒序CSV集合不同')
        for name in sorted(names):reverse.append(dict(variant=variant,**identical(left/name,right/name)))
    mapping=dict(cold='cold_50000',continued='continued_50000',small='cold_20000',large='cold_100000',cost2='cost2_50000',reverse='reverse_50000')
    for variant in ['baseline','persist10']:
        for kind,base_name in mapping.items():
            folder=out/(kind+'_'+variant)
            parents=[Path(m['parent'])/'execution'/(kind+'_'+variant)]
            if variant=='baseline':parents.append(Path(m['baseline_parent'])/'execution'/base_name)
            for parent in parents:
                for source in sorted(parent.glob('*.csv')):
                    if not (folder/source.name).exists() and source.name.startswith('recent_'):
                        deferred.append(dict(case=folder.name,file=source.name));continue
                    require((folder/source.name).exists(),'缺少原对照CSV: '+source.name)
                    references.append(dict(case=folder.name,parent=str(parent),**identical(source,folder/source.name)))
    return reverse,references,evidence,deferred


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--matrix-only',action='store_true')
    args=p.parse_args();run=args.run.resolve();out=run/'execution';m=json.loads((run/'manifest.json').read_text())
    spec,parent_proof=check_inputs(run,m);matrix_path=run/'independent_conditional_matrix.json'
    if args.matrix_only or not matrix_path.exists():matrix=verify_matrices(run,m,spec,parent_proof)
    else:
        matrix=json.loads(matrix_path.read_text());require(matrix['status']=='complete' and matrix['input_hashes']==m['input_hashes'] and matrix['verifier_sha256']==digest(Path(__file__)),'矩阵核验来源变化')
    if args.matrix_only:return
    cases=m['cases'];require(len(cases)==len({c['name'] for c in cases})==30,'预登记非30账户')
    require({p.parent.name for p in out.glob('*/summary.json')}=={c['name'] for c in cases},'30账户未齐或额外账户')
    provenance=verify_copy_provenance();dm=json.loads((out/'data_manifest.json').read_text())
    for name,h in dm['hashes'].items():require(digest(out/name)==h,'执行证据变化: '+name)
    pred=pd.read_parquet(run/'execution_predictions.parquet');pred['date']=pd.to_datetime(pred.date)
    require(digest(run/'execution_predictions.parquet')==dm['prediction_hash'],'预测漂移')
    indexed=pred.set_index(['date','ts_code']);require(not indexed.index.has_duplicates,'预测重复')
    signals=pd.read_parquet(out/'low_turnover_signals.parquet');require(set(signals.index).issubset(set(pred.date)),'月度预测覆盖缺失')
    top20={d:set(row.dropna().sort_values(ascending=False,kind='stable').head(20).index) for d,row in signals.iterrows()}
    market,limits=prepare_execution_evidence(pd.read_parquet(out/'market.parquet'),pd.read_parquet(out/'limits.parquet'))
    config=yaml.safe_load((out/'execution_config.yaml').read_text());reports=[];fills=[]
    for case in cases:
        folder=out/case['name'];saved=json.loads((folder/'portfolio_config.json').read_text());cfg=saved.get('portfolio',saved)
        expected=dict(config['portfolio']);expected.update(capital=case['capital'],top_n=5)
        for f in ['buy_cost','sell_cost','min_fee']:expected[f]*=case['cost_multiplier']
        require(cfg==expected,'账户配置偏离登记');summary=json.loads((folder/'summary.json').read_text())
        close(summary['capital'],case['capital'],'本金偏离');require(summary['policy']==case['policy']=='low_turnover','策略偏离')
        window=(pd.Timestamp(case['start']),pd.Timestamp(m['end']))
        account=verify_case(folder,pred,{'verification_window':window},{'portfolio':cfg})
        marks=verify_market_fills(folder,cfg,market,limits,window)
        fills.extend(dict(case=case['name'],**r) for r in marks.pop('fills'))
        exits=verify_first_exit_requests(folder,pd.read_parquet(run/'hold_variants'/(case['variant']+'.parquet')))
        trades=csv(folder/'trades.csv');buys=trades[trades.side.eq('buy')]
        for trade in buys.itertuples():
            date=pd.Timestamp(trade.signal_date);require(date in top20 and trade.ts_code in top20[date],'买入改变原monthly Top20')
            require(bool(indexed.loc[(date,trade.ts_code),'trend']) and np.isfinite(signals.loc[date,trade.ts_code]),'未恢复原入场门就买入')
        reports.append(dict(**account,**marks,first_exit_lifecycle_checks=exits,original_entry_checks=len(buys),
                            performance=performance_report(folder,case['capital'],pd.Timestamp(m['start']))))
        print('verified '+case['name'],flush=True)
    reverse,references,evidence,deferred=comparisons(run,m)
    report=dict(status='complete' if not deferred else 'derivative_outputs_pending',cases=reports,reverse_comparisons=reverse,
        reference_comparisons=references,reverse_input_evidence=evidence,deferred=deferred,matrix_report_sha256=digest(matrix_path),
        account_provenance=provenance,manifest_sha256=digest(run/'manifest.json'),input_hashes=m['input_hashes'],verifier_sha256=digest(Path(__file__)),
        reused_verifier_hashes={name:digest(ROOT/'scripts'/name) for name in ['verify_recent_low_turnover.py','verify_entry_guard_research.py','verify_hold_sensitivity.py']})
    (out/'independent_conditional_verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    pd.DataFrame(fills).to_csv(out/'independent_conditional_fill_checks.csv',index=False)
    (run/'INDEPENDENT_VERIFICATION.md').write_text('# 条件退出独立核验\n\n状态：'+report['status']+'。\n\n'
       f'{len(reports)}账户、{len(fills)}笔真实成交，{len(reverse)}项倒序CSV、{len(references)}项原对照CSV逐值通过。\n\n'
       '独立重建172列、57review的五种许可和长期状态；285项前缀验证；baseline/persist10原对照保持精确。\n\n'
       '允许度：persist10≤accelerated≤baseline≤patient，accelerated≤hybrid≤patient；hybrid与baseline无统一顺序。未知掩码和全A接口保持原样，确认计数不因分类切换重置。\n\n'
       '矩阵及数值边界见independent_conditional_matrix.json、independent_conditional_numerical.json；账户明细见execution/independent_conditional_verification.json。\n\n'
       '账户核验通过不代表政策有效；本轮是观察既有历史后的适应性研究，主假设hybrid不能事后改为最高收益分支。\n')
    print(json.dumps(dict(status=report['status'],cases=len(reports),fills=len(fills),reverse_csv=len(reverse),reference_csv=len(references)),ensure_ascii=False))


if __name__=='__main__':main()

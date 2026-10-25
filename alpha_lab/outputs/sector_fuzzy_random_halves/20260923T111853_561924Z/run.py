#!/usr/bin/env python3
"""固定事件池的逐年模糊筛选实验，外层年份绝不参与训练和选择。"""
from pathlib import Path
from datetime import datetime, timezone
import json
import os
import shutil
import sys
import unittest
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.linear_model import Ridge
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.backends.backend_pdf import PdfPages
HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(PROJECT / 'alpha_lab/src'))
from datahub import DataHub
from sector_dataset import SectorDataset, digest
from sector_candidate_quality import assign_clusters
from model import features_at_events, purged_history, inner_split, sample_weights, TrainScaler, fuzzy_score, fit_network, predict_network


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + '\n')


def stats(frame):
    values = frame.end_return.dropna()
    return dict(n=len(values), dates=frame.loc[frame.end_return.notna(), 'date'].nunique(),
        mean=values.mean(), median=values.median(), q10=values.quantile(.1), q90=values.quantile(.9),
        positive=values.gt(0).mean() if len(values) else np.nan,
        date_mean=frame.groupby('date').end_return.mean().mean(), min_return_mean=frame.min_return.mean())


def bootstrap_difference(frame, cfg):
    """按起点行情簇整簇抽样；选中和全池使用完全相同的重采样簇。"""
    groups = [g.index.to_numpy() for _, g in frame.groupby('cluster_id')]
    if len(groups) < 5:
        return dict(ci_mean_low=np.nan, ci_mean_high=np.nan, ci_median_low=np.nan, ci_median_high=np.nan, ci_q10_low=np.nan, ci_q10_high=np.nan)
    rng = np.random.default_rng(cfg['bootstrap_seed'])
    values, selection = frame.end_return.to_dict(), frame.selected.to_dict()
    samples = []
    for _ in range(cfg['bootstrap_repetitions']):
        indices = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        base = np.array([values[i] for i in indices])
        selected = base[[selection[i] for i in indices]]
        if len(selected):
            samples.append([selected.mean()-base.mean(), np.median(selected)-np.median(base), np.quantile(selected,.1)-np.quantile(base,.1)])
    intervals = np.quantile(samples, [.025, .975], axis=0)
    return {f'ci_{metric}_{side}': intervals[i,j] for j,metric in enumerate(['mean','median','q10']) for i,side in enumerate(['low','high'])}


def main():
    cfg = yaml.safe_load((HERE / 'config.yaml').read_text())
    torch.set_num_threads(1)
    test = unittest.TextTestRunner(verbosity=1).run(unittest.defaultTestLoader.discover(str(HERE), pattern='test_model.py'))
    assert test.wasSuccessful()
    source, returns, dataset = [PROJECT / cfg[k] for k in ['events','returns','dataset']]
    ds = SectorDataset(dataset)
    for directory in [source, returns]:
        manifest = json.loads((directory / 'manifest.json').read_text())
        assert manifest['status'] == 'complete' and manifest['dataset_content_id'] == ds.manifest['content_id']
        for name, value in manifest['artifacts'].items():
            assert digest(directory / name) == value, name
    assert os.environ.get('DATAHUB_CONTENT_ID') == ds.manifest['market_content_id']
    hub = DataHub()
    assert hub.cfg.root.resolve() == (PROJECT / cfg['release']).resolve()
    calendar = hub.calendar.cn_days(ds.manifest['config']['start'], ds.manifest['config']['end'])
    out = PROJECT / cfg['output_root'] / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    out.mkdir(parents=True)
    (out / 'models').mkdir()
    for name in ['config.yaml','PREREG.md','model.py','run.py','test_model.py']:
        shutil.copy2(HERE / name, out / name)
    save_json(out / 'manifest.json', dict(status='building', prereg_sha256=digest(HERE / 'PREREG.md')))
    print('OUTPUT', out, flush=True)
    signals = pd.read_parquet(source / 'confirmed_all_signals_with_dedup.parquet')
    signals = signals[signals.accepted].copy()
    signals['source_event_id'] = signals.event_id
    signals['event_id'] = signals.merged_event_id
    panel = pd.read_parquet(dataset / 'daily.parquet')
    features = features_at_events(signals, panel, cfg).sort_values(['date','sector_code']).reset_index(drop=True)
    cut = pd.Timestamp('2023-06-30')
    partial = features_at_events(signals[signals.date <= cut], panel[panel.date <= cut], cfg).sort_values(['date','sector_code']).reset_index(drop=True)
    pd.testing.assert_frame_equal(partial, features[features.date <= cut].reset_index(drop=True))
    assert len(features) == 330 and features.event_id.is_unique
    features.to_parquet(out / 'causal_features.parquet', index=False)
    outcomes = pd.read_parquet(returns / 'sector_event_outcomes.parquet')
    target = outcomes[outcomes.horizon.eq(cfg['target_horizon'])][['event_id','end_date','end_return','complete']].rename(columns={'end_return':'target'})
    data = features.merge(target,on='event_id',validate='one_to_one')
    predictions, folds, tuning, logs = [], [], [], []
    for year in cfg['test_years']:
        start = pd.Timestamp(year=year,month=1,day=1)
        test_frame = data[data.date.dt.year.eq(year)].copy()
        if test_frame.empty:
            continue
        train, cutoff = purged_history(data, start, calendar, cfg)
        try:
            fit, valid, inner_cutoff = inner_split(train, calendar, cfg)
        except ValueError as error:
            folds.append(dict(year=year,status='skipped',reason=str(error),train=len(train),test=len(test_frame)))
            continue
        assert train.end_date.lt(cutoff).all() and fit.end_date.lt(inner_cutoff).all()
        scaler = TrainScaler(cfg).fit(fit[cfg['features']])
        x, vx = scaler.transform(fit[cfg['features']]), scaler.transform(valid[cfg['features']])
        y, vy = fit.target.to_numpy()/cfg['target_scale'], valid.target.to_numpy()/cfg['target_scale']
        weights = sample_weights(fit.date, valid.date.min(), cfg['half_life_days'])
        vw = sample_weights(valid.date, start, None)
        ridge_candidates = []
        for alpha in cfg['ridge_alphas']:
            estimator = Ridge(alpha=alpha).fit(x,y,sample_weight=weights)
            value = float(np.average((estimator.predict(vx)-vy)**2, weights=vw))
            ridge_candidates.append((value,alpha))
            tuning.append(dict(year=year,model='ridge',alpha=alpha,validation_loss=value))
        best_alpha = min(ridge_candidates)[1]
        candidates = []
        for rules in cfg['fnn']['rules']:
            for lr in cfg['fnn']['learning_rates']:
                scores, epochs = [], []
                for seed in cfg['fnn']['tuning_seeds']:
                    _, best_epoch, loss, trajectory = fit_network(x,y,weights,rules,lr,seed,cfg['fnn'],validation=(vx,vy,vw))
                    scores.append(loss); epochs.append(best_epoch)
                    tuning.append(dict(year=year,model='fnn',rules=rules,learning_rate=lr,seed=seed,best_epoch=best_epoch,validation_loss=loss))
                    logs.extend(dict(year=year,phase='inner',rules=rules,learning_rate=lr,seed=seed,**r) for r in trajectory)
                candidates.append((float(np.mean(scores)),rules,lr,int(np.median(epochs))))
        loss, rules, lr, epochs = min(candidates)
        final_scaler = TrainScaler(cfg).fit(train[cfg['features']])
        tx = final_scaler.transform(train[cfg['features']])
        ox = final_scaler.transform(test_frame[cfg['features']])
        ty = train.target.to_numpy()/cfg['target_scale']
        tw = sample_weights(train.date,start,cfg['half_life_days'])
        estimator = Ridge(alpha=best_alpha).fit(tx,ty,sample_weight=tw)
        train_scores = {'ridge':estimator.predict(tx)*cfg['target_scale']}
        test_scores = {'ridge':estimator.predict(ox)*cfg['target_scale']}
        medians = train[cfg['features']].median().fillna(0)
        train_scores['fuzzy'] = fuzzy_score(train,medians,cfg)
        test_scores['fuzzy'] = fuzzy_score(test_frame,medians,cfg)
        train_network, test_network = [], []
        for seed in cfg['fnn']['ensemble_seeds']:
            network, _, _, trajectory = fit_network(tx,ty,tw,rules,lr,seed,cfg['fnn'],epochs=epochs)
            train_network.append(predict_network(network,tx)*cfg['target_scale'])
            test_network.append(predict_network(network,ox)*cfg['target_scale'])
            logs.extend(dict(year=year,phase='refit',rules=rules,learning_rate=lr,seed=seed,**r) for r in trajectory)
            torch.save(dict(state_dict=network.state_dict(),rules=rules,seed=seed,epochs=epochs,
                feature_names=cfg['features'],median=final_scaler.median,mean=final_scaler.mean,scale=final_scaler.scale),out/'models'/f'{year}_fnn_{seed}.pt')
        train_scores['fnn'] = np.mean(train_network,axis=0)
        test_scores['fnn'] = np.mean(test_network,axis=0)
        np.savez(out/'models'/f'{year}_ridge.npz',coef=estimator.coef_,intercept=estimator.intercept_,alpha=best_alpha,
            median=final_scaler.median,mean=final_scaler.mean,scale=final_scaler.scale)
        for model in ['fuzzy','ridge','fnn']:
            for retention in cfg['retention_fractions']:
                threshold = float(np.quantile(train_scores[model],1-retention))
                for r,score in zip(test_frame.itertuples(),test_scores[model]):
                    predictions.append(dict(event_id=r.event_id,date=r.date,sector_code=r.sector_code,year=year,
                        model=model,retention=retention,score=float(score),threshold=threshold,selected=bool(score>=threshold)))
        folds.append(dict(year=year,status='complete',train=len(train),train_dates=train.date.nunique(),fit=len(fit),valid=len(valid),
            valid_dates=valid.date.nunique(),test=len(test_frame),outer_cutoff=cutoff,train_last_label=train.end_date.max(),
            inner_cutoff=inner_cutoff,fit_last_label=fit.end_date.max(),rules=rules,learning_rate=lr,epochs=epochs,
            inner_loss=loss,ridge_alpha=best_alpha))
        save_json(out/f'fold_{year}_indices.json',dict(train=train.event_id.tolist(),fit=fit.event_id.tolist(),valid=valid.event_id.tolist(),test=test_frame.event_id.tolist(),
            fuzzy_medians=medians.to_dict(),outer_cutoff=cutoff,inner_cutoff=inner_cutoff))
        print('FOLD',year,'train',len(train),'valid',len(valid),'test',len(test_frame),'rules',rules,'lr',lr,'epochs',epochs,flush=True)
    pd.DataFrame(folds).to_csv(out/'folds.csv',index=False)
    pd.DataFrame(tuning).to_csv(out/'inner_search.csv',index=False)
    pd.DataFrame(logs).to_csv(out/'training_curves.csv',index=False)
    pred = pd.DataFrame(predictions)
    assert not pred.empty and not pred.duplicated(['event_id','model','retention']).any()
    pred.to_parquet(out/'predictions_before_labels.parquet',index=False)
    clusters = assign_clusters(pred.date.unique(),calendar,cfg['cluster_gap_sessions'])
    evaluated = pred.merge(outcomes,on=['event_id','date','sector_code'],validate='many_to_many').merge(clusters,on='date',validate='many_to_one')
    evaluated.to_parquet(out/'evaluated_predictions_WITH_FUTURE.parquet',index=False)
    event_clusters = pred.drop_duplicates('event_id')[['event_id','date']].merge(clusters,on='date')
    sizes = event_clusters.groupby('cluster_id').agg(events=('event_id','size'),dates=('date','nunique'),start=('date','min'),end=('date','max'))
    sizes.to_csv(out/'cluster_sizes.csv')
    largest = sizes.events.idxmax()
    summaries, annual, pairs, sensitivity = [], [], [], []
    for (model,retention,horizon), group in evaluated.groupby(['model','retention','horizon']):
        g = group[group.complete].copy()
        selected = g[g.selected]
        base, chosen = stats(g), stats(selected)
        row = dict(model=model,retention=retention,horizon=horizon,scored_events=len(group),selected_signals=int(group.selected.sum()),
            base_n=base['n'],**chosen,selected_fraction=chosen['n']/base['n'],clusters=g.cluster_id.nunique())
        for metric in ['mean','median','q10','q90','date_mean','positive','min_return_mean']:
            row['base_'+metric] = base[metric]
            row['delta_'+metric] = chosen[metric]-base[metric]
        if horizon == cfg['target_horizon']:
            row.update(bootstrap_difference(g,cfg))
            row['descriptive_right_shift'] = bool(chosen['n']>=cfg['minimum_selected_events'] and chosen['dates']>=cfg['minimum_selected_dates'] and
                row['selected_fraction']>=cfg['minimum_selected_fraction'] and all(row['delta_'+m]>0 for m in ['mean','median','q10']) and row['delta_date_mean']>0)
            row['intervals_support_shift'] = bool(row['descriptive_right_shift'] and all(row['ci_'+m+'_low']>0 for m in ['mean','median','q10']))
        summaries.append(row)
        for year,part in g.groupby('year'):
            a,b = stats(part),stats(part[part.selected])
            annual.append(dict(model=model,retention=retention,horizon=horizon,year=year,base_n=a['n'],**b,
                base_mean=a['mean'],delta_mean=b['mean']-a['mean'],delta_q10=b['q10']-a['q10']))
        same = g.groupby(['date','selected']).end_return.mean().unstack('selected').reindex(columns=[False,True]).dropna()
        pairs.append(dict(model=model,retention=retention,horizon=horizon,paired_dates=len(same),
            selected_minus_rejected=(same[True]-same[False]).mean()))
        left = g[g.cluster_id.ne(largest)]
        a,b=stats(left),stats(left[left.selected])
        sensitivity.append(dict(model=model,retention=retention,horizon=horizon,removed_cluster=int(largest),**b,base_mean=a['mean'],delta_mean=b['mean']-a['mean'],delta_q10=b['q10']-a['q10']))
    summary = pd.DataFrame(summaries)
    summary.to_csv(out/'selection_summary.csv',index=False)
    pd.DataFrame(annual).to_csv(out/'annual_selection.csv',index=False)
    pd.DataFrame(pairs).to_csv(out/'same_date_controls.csv',index=False)
    pd.DataFrame(sensitivity).to_csv(out/'largest_cluster_removed.csv',index=False)
    font_manager.fontManager.addfont(str(PROJECT/cfg['font']))
    plt.rcParams.update({'font.family':font_manager.FontProperties(fname=PROJECT/cfg['font']).get_name(),'axes.unicode_minus':False,'pdf.fonttype':42})
    labels = dict(fuzzy='固定模糊规则',ridge='岭回归',fnn='模糊神经网络')
    charts = []
    with PdfPages(out/'fuzzy_selection_report.pdf') as book:
        fig,axes=plt.subplots(1,3,figsize=(17,5),layout='constrained')
        for ax,horizon in zip(axes,cfg['report_horizons']):
            common=evaluated[(evaluated.horizon==horizon)&evaluated.complete]
            base=common.drop_duplicates('event_id').end_return.sort_values().to_numpy()*100
            ax.step(base,np.arange(1,len(base)+1)/len(base),label=f'不筛选 n={len(base)}',color='#555',lw=2)
            for model in labels:
                sub=common[(common.model==model)&(common.retention==cfg['primary_retention'])&common.selected]
                x=sub.end_return.sort_values().to_numpy()*100
                if len(x):ax.step(x,np.arange(1,len(x)+1)/len(x),label=f'{labels[model]} n={len(x)}')
            ax.set(title=f'{horizon}日收益经验分布',xlabel='期末收盘收益（%）',ylabel='累计比例')
            ax.legend(fontsize=8);ax.grid(alpha=.15)
        fig.suptitle('历史滚动测试：阈值在测试年前冻结；分布越靠右，收益整体越高',fontsize=14)
        fig.savefig(out/'return_ecdf.png',dpi=cfg['dpi']);book.savefig(fig);plt.close(fig);charts.append('return_ecdf.png')
        fig,axes=plt.subplots(1,3,figsize=(17,5),layout='constrained')
        primary=summary[(summary.horizon==cfg['target_horizon'])&(summary.retention==cfg['primary_retention'])].set_index('model')
        for ax,metric,title in zip(axes,['mean','median','q10'],['均值变化','中位数变化','10%分位变化']):
            for i,model in enumerate(labels):
                r=primary.loc[model]
                ax.scatter(i,r['delta_'+metric]*100)
                ax.vlines(i,r['ci_'+metric+'_low']*100,r['ci_'+metric+'_high']*100,lw=2)
            ax.axhline(0,color='#777',ls='--');ax.set_xticks(range(len(labels)),labels.values())
            ax.set(title=title,ylabel='相对不筛选的变化（百分点）');ax.grid(axis='y',alpha=.15)
        fig.suptitle('20日收益：整簇重采样95%区间；主保留参数50%',fontsize=14)
        fig.savefig(out/'distribution_shift.png',dpi=cfg['dpi']);book.savefig(fig);plt.close(fig);charts.append('distribution_shift.png')
        fig,axes=plt.subplots(1,2,figsize=(14,5),layout='constrained')
        for ax,year in zip(axes,[min(pred.year),max(pred.year)]):
            chosen=pd.DataFrame(folds).query('year == @year').iloc[0]
            curve=pd.DataFrame(logs)
            curve=curve[(curve.year==year)&(curve.phase=='inner')&(curve.rules==chosen.rules)&(curve.learning_rate==chosen.learning_rate)]
            for seed,g in curve.groupby('seed'):
                ax.plot(g.epoch,g.train_loss,label=f'训练 {seed}')
                ax.plot(g.epoch,g.validation_loss,ls='--',label=f'验证 {seed}')
            ax.set(title=f'{year}年内层选中组合',xlabel='迭代轮数',ylabel='加权Huber损失');ax.legend(fontsize=8)
        fig.suptitle('训练过程：验证早停轮数用于外层重训',fontsize=14)
        fig.savefig(out/'training_diagnostics.png',dpi=cfg['dpi']);book.savefig(fig);plt.close(fig);charts.append('training_diagnostics.png')
    fmt=lambda x:f'{x:.2%}' if pd.notna(x) else '缺失'
    lines=['# 模糊事件筛选实验：历史滚动报告','',
        f'实验分支 `experiment/sector-fuzzy-selection-20260923`。原330事件；参与测试{pred.event_id.nunique()}事件、{pred.date.nunique()}日期，{len(sizes)}个行情簇。', '',
        '## 主结果：20日，训练评分保留参数50%', '',
        '| 方法 | 完整选中/全池 | 均值 | 中位数 | 10%分位 | 日期等权均值 | 描述性右移 | 区间支持 |',
        '|---|---:|---:|---:|---:|---:|---|---|']
    first=primary.iloc[0]
    lines.append(f'| 不筛选 | {int(first.base_n)}/{int(first.base_n)} | {fmt(first.base_mean)} | {fmt(first.base_median)} | {fmt(first.base_q10)} | {fmt(first.base_date_mean)} | 基准 | 基准 |')
    for model,r in primary.iterrows():
        lines.append(f'| {labels[model]} | {int(r.n)}/{int(r.base_n)} | {fmt(r["mean"])} | {fmt(r["median"])} | {fmt(r.q10)} | {fmt(r.date_mean)} | {r.descriptive_right_shift} | {r.intervals_support_shift} |')
    lines+=['','## 相对不筛选的变化与区间','','| 方法 | 均值变化（95%区间） | 中位数变化（95%区间） | 10%分位变化（95%区间） |','|---|---|---|---|']
    for model,r in primary.iterrows():
        text=[f'{fmt(r["delta_"+m])} [{fmt(r["ci_"+m+"_low"])}, {fmt(r["ci_"+m+"_high"])}]' for m in ['mean','median','q10']]
        lines.append('| '+labels[model]+' | '+' | '.join(text)+' |')
    lines+=['','## 方法与适用范围','',
        '- 所有特征截至首次确认日，真实截断重算一致；标签、最终归组及事后涨跌不作为输入。',
        '- 每年训练截止、内层验证、标签成熟与5日隔离区见folds.csv；模型训练、验证、测试索引分别保存。训练期中位数填充、缩放、评分阈值均未使用测试年样本。',
        '- 网络是一阶Takagi–Sugeno模糊网络：高斯前件可训练、规则后件线性、归一化激活。规则数×学习率联合比较4组合、2种子，选定后3种子集成。不同组合性能不假定单调。',
        '- 最多1500轮，最少100轮，验证早停150轮；年度最终训练轮数来自内层最佳轮数中位数。训练窗口8年，权重半衰期3年，并按确认日期均衡训练权重。',
        '- 50%为主参数、70%为固定敏感性。实际测试保留率会变化；两种参数结果均在selection_summary.csv，不事后挑选保留率。',
        '- 描述性右移要求均值、中位数、10%分位及日期等权均值均改善，并满足预设样本覆盖。区间支持额外要求三个差值的95%簇区间下界均大于零；未做多重比较校正。',
        '- 相邻起点不超过20个交易日传递连接行情簇，不能保证簇独立。完整未来窗口仅在评价时使用，预测记录包含后续缺失事件；不是交易时过滤。',
        '- 事件规则与总体收益此前已经看过，当前仅为历史滚动检验，不能称作全新样本外证据。收益从确认日收盘计，未模拟可成交入场、费用或资金曲线。',
        '- 方法参考：[Jang 1993 ANFIS原论文](https://ieeexplore.ieee.org/document/256541/)，本实验使用Adam端到端训练，未复现其经典混合训练算法。','',
        '## 文件','',
        '- fuzzy_selection_report.pdf / overview.html：收益分布、差值区间和训练曲线。',
        '- selection_summary.csv：全部模型×50%/70%参数×3/10/20日收益、分位数、覆盖和差值。',
        '- annual_selection.csv、same_date_controls.csv、largest_cluster_removed.csv：逐年、同日选中/未选中对照及最大行情簇敏感性。',
        '- causal_features.parquet、predictions_before_labels.parquet为特征和事前评分；evaluated_predictions_WITH_FUTURE.parquet含未来标签，禁止直接作为输入。',
        '- inner_search.csv、training_curves.csv、models/、fold_*_indices.json记录全部调参与训练过程。','']
    (out/'README.md').write_text('\n'.join(lines))
    (out/'overview.html').write_text('<!doctype html><meta charset="utf-8"><title>模糊事件筛选实验</title><style>body{max-width:1400px;margin:auto;font-family:system-ui}img{width:100%}</style><h1>模糊事件筛选：历史滚动检验</h1>'+''.join(f'<img src="{name}">' for name in charts))
    save_json(out/'validation.json',dict(tests=test.testsRun,causal_feature_prefix_exact=True,purged_train_labels=True,
        prediction_keys_unique=True,source_hashes=True,source_events=330,test_events=pred.event_id.nunique()))
    save_json(out/'manifest.json',dict(status='complete',branch='experiment/sector-fuzzy-selection-20260923',dataset_content_id=ds.manifest['content_id'],
        config=cfg,source_manifest_sha256=digest(source/'manifest.json'),returns_manifest_sha256=digest(returns/'manifest.json'),
        artifacts={str(p.relative_to(out)):digest(p) for p in out.rglob('*') if p.is_file() and p.name!='manifest.json'}))
    print('COMPLETE',out,flush=True)
    print(primary[['n','base_n','mean','median','q10','delta_mean','delta_median','delta_q10','descriptive_right_shift','intervals_support_shift']].to_string(),flush=True)


if __name__=='__main__':
    main()

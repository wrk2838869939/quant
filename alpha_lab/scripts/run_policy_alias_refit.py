#!/usr/bin/env python3
"""仅修复官方限价历史代码连接，独立目录复用原build/fit/停止门。"""
import argparse
from contextlib import contextmanager
import hashlib
import inspect
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
import yaml

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
EFFECTIVE_DATE=pd.Timestamp('2019-12-16')
OLD_CODE='000043.SZ'
NEW_CODE='001914.SZ'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def rename_historical_limit(frame,date):
    """只改一个有证据的历史代码，不变更行、索引、字段或任何限价数值。"""
    date=pd.Timestamp(date)
    if date>=EFFECTIVE_DATE:return frame,False
    if not {'ts_code','trade_date','up_limit','down_limit'}.issubset(frame.columns):
        raise ValueError('官方限价缺少必要列')
    if not pd.to_datetime(frame.trade_date).eq(date).all():
        raise ValueError('限价内容日期和文件日期不一致')
    old=frame.ts_code.eq(OLD_CODE);new=frame.ts_code.eq(NEW_CODE)
    if old.sum()>1 or new.sum()>1 or (old.any() and new.any()):
        raise ValueError('历史别名存在重复或目标代码冲突，拒绝合并')
    if not old.any():return frame,False
    changed=frame.copy(deep=True)
    changed.loc[old,'ts_code']=NEW_CODE
    pd.testing.assert_frame_equal(changed.drop(columns='ts_code'),frame.drop(columns='ts_code'),check_exact=True)
    pd.testing.assert_series_equal(changed.loc[~old,'ts_code'],frame.loc[~old,'ts_code'])
    return changed,True


@contextmanager
def scoped_limit_alias_reader(build_function,cache_dir):
    """只截获原build函数直接读取指定日限价目录的调用，finally恢复pandas。"""
    original=pd.read_parquet;records=[];cache_dir=Path(cache_dir).resolve()
    def read_adapter(path,*args,**kwargs):
        frame=original(path,*args,**kwargs)
        caller=inspect.currentframe().f_back
        if caller.f_code is not build_function.__code__:
            return frame
        try:resolved=Path(path).resolve()
        except TypeError:return frame
        if resolved.parent!=cache_dir or resolved.suffix!='.parquet' or not resolved.stem.isdigit() or len(resolved.stem)!=8:
            return frame
        date=pd.to_datetime(resolved.stem,format='%Y%m%d')
        result,changed=rename_historical_limit(frame,date)
        if changed:
            old=frame.loc[frame.ts_code.eq(OLD_CODE)].iloc[0]
            records.append(dict(date=date,source_path=str(resolved),source_file_sha256=sha(resolved),
                old_code=OLD_CODE,new_code=NEW_CODE,up_limit=old.up_limit,down_limit=old.down_limit,
                original_rows=len(frame),returned_rows=len(result),numeric_fields_unchanged=True))
        return result
    pd.read_parquet=read_adapter
    try:yield records
    finally:
        pd.read_parquet=original


def compare_runs(parent,run,cfg):
    """全保留旧结果，显示影响股票、共同训练样本及全部八模型的变化。"""
    old=pd.read_parquet(parent/'policy_labels.parquet');new=pd.read_parquet(run/'policy_labels.parquet')
    pd.testing.assert_series_equal(old.date,new.date);pd.testing.assert_series_equal(old.ts_code,new.ts_code)
    pd.testing.assert_frame_equal(old.loc[old.ts_code.ne(NEW_CODE)],new.loc[new.ts_code.ne(NEW_CODE)],check_exact=True)
    pd.testing.assert_frame_equal(old[['date','ts_code','trend','monthly']+cfg['features']],
                                  new[['date','ts_code','trend','monthly']+cfg['features']],check_exact=True)
    identical=(old.eq(new)|(old.isna()&new.isna())).all(axis=1)
    changed=old.loc[~identical,['date','ts_code','trend']].copy()
    for key in ['entry_missing_120','execution_unknown_120']:
        changed['old_'+key]=old.loc[~identical,key];changed['new_'+key]=new.loc[~identical,key]
    for target in cfg['targets']:
        changed['old_'+target+'_return']=old.loc[~identical,target+'_return']
        changed['new_'+target+'_return']=new.loc[~identical,target+'_return']
    changed.to_csv(run/'alias_changed_opportunities.csv',index=False)
    old_meta=json.loads((parent/'policy_model_metadata.json').read_text())
    new_meta=json.loads((run/'policy_model_metadata.json').read_text())
    training=[];prediction_stats=[];diagnostic_rows=[]
    for stage in ['development','execution']:
        training.append(dict(stage=stage,cutoff=old_meta[stage]['cutoff'],
            old_common_training_rows=old_meta[stage]['common_training_rows'],
            new_common_training_rows=new_meta[stage]['common_training_rows'],
            added_training_rows=new_meta[stage]['common_training_rows']-old_meta[stage]['common_training_rows'],
            old_common_cohort_hash=old_meta[stage]['common_cohort_hash'],
            new_common_cohort_hash=new_meta[stage]['common_cohort_hash']))
        po=pd.read_parquet(parent/f'{stage}_predictions.parquet');pn=pd.read_parquet(run/f'{stage}_predictions.parquet')
        pd.testing.assert_frame_equal(po[['date','ts_code','trend']],pn[['date','ts_code','trend']],check_exact=True)
        for policy in ['low_turnover']+cfg['targets']:
            d=pn[policy+'__expected_return']-po[policy+'__expected_return']
            prediction_stats.append(dict(stage=stage,policy=policy,rows=len(po),
                max_absolute_prediction_change=float(d.abs().max()),mean_prediction_change=float(d.mean()),
                old_admitted=int(po[policy+'__score'].notna().sum()),new_admitted=int(pn[policy+'__score'].notna().sum())))
        for version,path in [('original',parent),('alias_corrected',run)]:
            t=pd.read_csv(path/f'{stage}_diagnostics/top_k_summary.csv')
            t=t[t.k.eq(5)&t.target_kind.eq('common_patient120')].copy()
            t.insert(0,'stage',stage);t.insert(0,'version',version);diagnostic_rows.append(t)
    pd.DataFrame(training).to_csv(run/'alias_training_cohort_comparison.csv',index=False)
    pd.DataFrame(prediction_stats).to_csv(run/'alias_eight_model_prediction_comparison.csv',index=False)
    pd.concat(diagnostic_rows,ignore_index=True).to_csv(run/'alias_eight_model_top5_comparison.csv',index=False)
    gates=[]
    for version,path in [('original',parent),('alias_corrected',run)]:
        t=pd.read_csv(path/'policy_target_gate.csv');t.insert(0,'version',version);gates.append(t)
    pd.concat(gates,ignore_index=True).to_csv(run/'alias_gate_comparison.csv',index=False)
    before=int(old.execution_unknown_120.sum());after=int(new.execution_unknown_120.sum())
    findings=dict(changed_label_rows=int((~identical).sum()),unchanged_other_stocks=True,
        old_unknown_opportunities=before,new_unknown_opportunities=after,
        old_unknown_trend=int((old.execution_unknown_120&old.trend).sum()),
        new_unknown_trend=int((new.execution_unknown_120&new.trend).sum()),
        training_changes=training)
    return findings


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--parent',type=Path,required=True);args=parser.parse_args()
    run=args.run.resolve();parent=args.parent.resolve()
    if run==parent:raise ValueError('必须使用独立追加目录')
    registration=json.loads((run/'alias_registration.json').read_text())
    if not registration['data_correction_only']:raise ValueError('未登记只修数据连接')
    if (run/'policy_labels.parquet').exists():raise ValueError('禁止覆盖已生成追加标签')
    import build_policy_labels
    from fit_policy_targets import run_fit
    fixed=['config.yaml','policy_features.parquet','policy_market.npz','limits_download.csv',
           'label_preparation.json','limits_manifest.json','p2_decision_protocol.json']
    for name in fixed:
        if not (run/name).exists():shutil.copy2(parent/name,run/name)
        if sha(run/name)!=sha(parent/name):raise ValueError(f'固定输入不同: {name}')
    cfg=yaml.safe_load((run/'config.yaml').read_text())
    original_build=json.loads((parent/'policy_labels_manifest.json').read_text())
    original_fit=json.loads((parent/'fit_policy_manifest.json').read_text())
    for rel in ['scripts/build_policy_labels.py','src/policy_path_labels.py','src/patient_trend.py']:
        if sha(ROOT/rel)!=original_build['source_hashes'][rel]:raise ValueError(f'原标签源码已变: {rel}')
    for rel in ['scripts/fit_policy_targets.py','src/value_forecast.py']:
        if sha(ROOT/rel)!=original_fit['source_hashes'][str(ROOT/rel)]:raise ValueError(f'原模型源码已变: {rel}')
    protected=fixed+['policy_labels.parquet','development_models.joblib','execution_models.joblib',
        'development_predictions.parquet','execution_predictions.parquet','policy_target_decision.json']
    protected_hashes={name:sha(parent/name) for name in protected}
    notice=run/'alias_sources/issuer_notice_2019_160.pdf'
    if not notice.exists():raise ValueError('缺少已核验发行人公告归档')
    source_files=[Path(__file__),ROOT/'tests/test_policy_alias_refit.py',ROOT/'scripts/build_policy_labels.py',
        ROOT/'scripts/fit_policy_targets.py',ROOT/'scripts/evaluate_policy_target_gate.py',
        ROOT/'src/policy_path_labels.py',ROOT/'src/patient_trend.py',ROOT/'src/value_forecast.py']
    source_hashes={str(p):sha(p) for p in source_files}
    for p in source_files:
        dest=run/'alias_sources'/p.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
    manifest=dict(status='building_labels',data_correction_only=True,parent=str(parent),
        same_original_config=True,same_original_stop_protocol=True,old_code=OLD_CODE,new_code=NEW_CODE,
        effective_date=str(EFFECTIVE_DATE.date()),announcement_sha256=sha(notice),
        original_frozen_hashes=protected_hashes,source_hashes=source_hashes,
        raw_cache_modified=False,original_run_modified=False,cash_cases_run=False)
    manifest_path=run/'alias_manifest.json'
    def save():manifest_path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2,default=str))
    save()
    original_reader=pd.read_parquet
    try:
        with scoped_limit_alias_reader(build_policy_labels.build,ROOT/'data/raw/policy_limits_by_date') as records:
            build_policy_labels.build(run,cfg)
        assert pd.read_parquet is original_reader
        pd.DataFrame(records).to_csv(run/'alias_applied_limit_rows.csv',index=False)
        manifest.update(status='fitting_models',mapped_limit_rows=len(records),pandas_reader_restored_before_fit=True);save()
        print('alias labels complete; original pandas reader restored',flush=True)
        run_fit(run)
        manifest['status']='evaluating_original_gate';save()
        subprocess.run([sys.executable,str(ROOT/'scripts/evaluate_policy_target_gate.py'),'--run',str(run)],check=True)
        findings=compare_runs(parent,run,cfg)
        for name,expected in protected_hashes.items():
            if sha(parent/name)!=expected:raise ValueError(f'原冻结产物被修改: {name}')
        # 原build核查每个缓存文件原始hash，adapter从未写回；结束时再核查全部文件。
        for row in pd.read_csv(run/'limits_download.csv').itertuples():
            if sha(ROOT/row.path)!=row.sha256:raise ValueError(f'原缓存被修改: {row.path}')
        decision=json.loads((run/'policy_target_decision.json').read_text())
        manifest.update(status='complete',findings=findings,original_cache_hashes_unchanged=True,
            gate_passed=decision['passed'],gate_action=decision['action'],model_count=8,
            output_hashes={p.name:sha(p) for p in sorted(run.iterdir()) if p.is_file() and p.name!='alias_manifest.json'})
        save();print(json.dumps(findings,ensure_ascii=False,indent=2),flush=True)
    except Exception as error:
        manifest.update(status='failed',failure_type=type(error).__name__,failure_message=str(error),
                        pandas_reader_restored=pd.read_parquet is original_reader)
        save();raise


if __name__=='__main__':main()

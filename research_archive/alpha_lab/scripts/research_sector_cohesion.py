# 历史实验脚本归档：运行依赖原研究环境及未发布输入。
#!/usr/bin/env python3
"""板块市场因素剔除、年度稳定性和龙头代理代表性诊断。"""
from datetime import datetime, timezone
from pathlib import Path
import json
import os
import shutil
import sys
import numpy as np
import pandas as pd
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
sys.path[:0] = [str(ROOT / 'src')]
from datahub import DataHub
from sector_dataset import SectorDataset, digest
from sector_cohesion import residualize, correlation_stats, series_correlation


def savefig(fig, out, name, dpi):
    fig.savefig(out / (name + '.png'), dpi=dpi, bbox_inches='tight')
    fig.savefig(out / (name + '.pdf'), bbox_inches='tight')
    plt.close(fig)


def main():
    config = ROOT / 'configs/sector_cohesion_v1.yaml'
    cfg = yaml.safe_load(config.read_text())
    dataset = PROJECT / cfg['dataset']
    ds = SectorDataset(dataset)
    if os.environ.get('DATAHUB_CONTENT_ID') != ds.manifest['market_content_id']:
        raise RuntimeError('请使用冻结版本启动')
    hub = DataHub()
    if hub.cfg.root.resolve() != (PROJECT / ds.manifest['config']['release']).resolve():
        raise RuntimeError('行情路径不符')
    out = PROJECT / cfg['output_root'] / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    out.mkdir(parents=True, exist_ok=False)
    print('OUTPUT', out, flush=True)
    font_manager.fontManager.addfont(str(PROJECT / cfg['font']))
    plt.rcParams.update({'font.family': font_manager.FontProperties(fname=PROJECT/cfg['font']).get_name(),
                        'axes.unicode_minus': False, 'pdf.fonttype': 42})
    mapping = pd.read_parquet(dataset / 'membership_daily.parquet').loc[cfg['history_start']:]
    sector_codes = json.loads((dataset/'sector_codes.json').read_text())
    catalog = pd.read_parquet(dataset/'catalog_sw_l2.parquet')
    catalog = catalog[catalog.src.eq(cfg['classification'])].drop_duplicates('index_code').sort_values('index_code')
    dates, stocks = mapping.index, mapping.columns
    raw = hub.cn.daily(start=str(dates.min().date()), end=str(dates.max().date()), fields=['hfq_close', 'total_mv_yuan'])
    assert not raw.duplicated(['date', 'ts_code']).any()
    close = raw.pivot(index='date', columns='ts_code', values='hfq_close').reindex(index=dates, columns=stocks)
    cap = raw.pivot(index='date', columns='ts_code', values='total_mv_yuan').reindex_like(close)
    del raw
    # 主档存续限制与已冻结板块数据构建口径一致，不使用当前上市状态筛选。
    basic = pd.read_parquet(hub.cfg.meta_dir/'stock_basic.parquet')
    for r in basic.itertuples():
        if r.ts_code not in close.columns:
            continue
        start = pd.to_datetime(r.list_date, format='%Y%m%d', errors='coerce')
        end = pd.to_datetime(r.delist_date, format='%Y%m%d', errors='coerce')
        invalid = (dates < start) if pd.notna(start) else np.zeros(len(dates), bool)
        if pd.notna(end):
            invalid |= dates >= end
        close.loc[invalid, r.ts_code] = np.nan
        cap.loc[invalid, r.ts_code] = np.nan
    close = close.where(close > 0)
    returns = close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
    momentum = (close/close.shift(cfg['momentum_window'])-1).shift(1)
    cap = cap.where(cap > 0).shift(1)
    del close
    index = pd.read_parquet(hub.cfg.root/'datasets/index/cn_index_daily.parquet')
    index = index[index.ts_code.isin([cfg['market_index'], cfg['small_index']])]
    index_close = index.pivot(index='date', columns='ts_code', values='close').reindex(dates)
    index_ret = index_close.pct_change(fill_method=None)
    factors = pd.DataFrame(dict(market=index_ret[cfg['market_index']],
                                size=index_ret[cfg['small_index']]-index_ret[cfg['market_index']]))
    print('RESIDUALIZE', flush=True)
    residual1, _ = residualize(returns, factors[['market']], cfg)
    residual2, models = residualize(returns, factors, cfg)
    # 真实数据上的前缀重算及独立回归核对。
    subset = returns.columns[:12]
    cut = len(dates)-100
    prefix, _ = residualize(returns.loc[:, subset].iloc[:cut], factors.iloc[:cut], cfg)
    pd.testing.assert_frame_equal(prefix, residual2.loc[:, subset].iloc[:cut])
    fit_checks = 0
    for code in returns.columns:
        if fit_checks >= 12:
            break
        if not np.isfinite(residual2[code].iloc[-1]):
            continue
        joint = pd.concat([returns[code].rename('y'), factors], axis=1).iloc[-cfg['beta_window']-1:-1].dropna()
        coef = np.linalg.lstsq(np.column_stack([np.ones(len(joint)), joint[['market','size']]]), joint.y, rcond=None)[0]
        expected = returns[code].iloc[-1]-np.dot(np.r_[1, factors.iloc[-1]], coef)
        np.testing.assert_allclose(expected, residual2[code].iloc[-1], atol=1e-10)
        fit_checks += 1
    del models
    stable = mapping.eq(mapping.shift(1)) & mapping.ge(0)
    switch = pd.Timestamp(ds.manifest['config']['version_change'])
    stable.loc[stable.index == switch] = False
    recent_dates = dates[-cfg['recent_sessions']:]
    all_methods = {'raw': returns.where(stable), 'market': residual1.where(stable), 'market_size': residual2.where(stable)}
    joint_valid = all_methods['raw'].notna() & all_methods['market'].notna() & all_methods['market_size'].notna()
    matched_methods = {k:v.where(joint_valid) for k,v in all_methods.items()}
    summary_rows, year_rows, leader_rows, daily_leaders = [], [], [], []
    original = pd.read_csv(PROJECT/cfg['prior_internal']/'summary_recent.csv').set_index('sector_code')
    for pos, row in enumerate(catalog.itertuples()):
        sector, name = row.index_code, row.industry_name
        sid = sector_codes.index(sector)
        mask = mapping.eq(sid)
        candidate = mask.any(axis=0)
        cols = stocks[candidate]
        frames = {k:v.loc[:, cols].where(mask.loc[:, cols]) for k,v in matched_methods.items()}
        # 原始口径另行复现此前报告，不将 IPO 预热造成的样本变化当成去市场效果。
        original_frame = all_methods['raw'].loc[recent_dates, cols].where(mask.loc[recent_dates, cols])
        old = correlation_stats(original_frame, cfg['minimum_pair_days'])
        assert old['pairs'] == original.loc[sector, 'valid_pairs']
        np.testing.assert_allclose(old['median'], original.loc[sector, 'median_correlation'], equal_nan=True, atol=1e-12)
        rec = dict(sector_code=sector, name=name, original_raw_median=old['median'], original_raw_pairs=old['pairs'])
        for method, frame in frames.items():
            stat = correlation_stats(frame.loc[recent_dates], cfg['minimum_pair_days'])
            rec.update({method+'_'+k:v for k,v in stat.items()})
            for year in cfg['stability_years']:
                part = frame.loc[frame.index.year == year]
                stat_year = correlation_stats(part, cfg['minimum_pair_days'])
                year_rows.append(dict(sector_code=sector, name=name, method=method, year=year, **stat_year))
        assert rec['raw_pairs'] == rec['market_pairs'] == rec['market_size_pairs']
        summary_rows.append(rec)
        # 所有代理均按前日已知成员选出，收益只取仍属同板块且有完整残差的观测。
        prior_member = mapping.loc[:, cols].shift(1).eq(sid)
        for proxy, score in [('cap_top3', cap.loc[:,cols]), ('momentum60_top3', momentum.loc[:,cols])]:
            score = score.where(prior_member)
            records = []
            for date in recent_dates:
                ranked = score.loc[date].dropna().sort_values(ascending=False, kind='stable')
                if len(ranked) < cfg['leader_count']:
                    continue
                selected = ranked.index[:cfg['leader_count']]
                rest = cols.difference(selected)
                raw_leaders = frames['raw'].loc[date, selected].dropna()
                raw_rest = frames['raw'].loc[date, rest].dropna()
                if len(raw_leaders) < cfg['minimum_leader_returns'] or len(raw_rest) < cfg['minimum_remaining_returns']:
                    continue
                assert not set(selected) & set(rest)
                record = dict(date=date, sector_code=sector, proxy=proxy, selected='|'.join(selected),
                              leader_observed=len(raw_leaders), remaining_observed=len(raw_rest),
                              leader_raw=float(raw_leaders.mean()), remaining_raw=float(raw_rest.mean()),
                              leader_residual=float(frames['market_size'].loc[date, selected].mean()),
                              remaining_residual=float(frames['market_size'].loc[date, rest].mean()))
                records.append(record)
            daily_leaders.extend(records)
            tab = pd.DataFrame(records, columns=['date','sector_code','proxy','selected','leader_observed',
                 'remaining_observed','leader_raw','remaining_raw','leader_residual','remaining_residual'])
            tab = tab.set_index('date').reindex(recent_dates)
            lrec = dict(sector_code=sector, name=name, proxy=proxy)
            for kind in ['raw','residual']:
                for lag in [0,1]:
                    corr, n = series_correlation(tab['leader_'+kind].shift(lag), tab['remaining_'+kind], cfg['minimum_pair_days'])
                    lrec[f'{kind}_lag{lag}_corr'] = corr
                    lrec[f'{kind}_lag{lag}_days'] = n
            leader_rows.append(lrec)
        if pos % 20 == 0:
            print(f'SECTORS {pos}/{len(catalog)}', flush=True)
    summary = pd.DataFrame(summary_rows)
    annual = pd.DataFrame(year_rows)
    leaders = pd.DataFrame(leader_rows)
    summary.to_csv(out/'recent_cohesion.csv', index=False)
    annual.to_csv(out/'annual_cohesion.csv', index=False)
    leaders.to_csv(out/'leader_representativeness.csv', index=False)
    pd.DataFrame(daily_leaders).to_parquet(out/'leader_daily.parquet', index=False)
    factors.to_parquet(out/'market_factors.parquet')
    valid = summary.dropna(subset=['market_size_median'])
    year_matrix = annual[annual.method.eq('market_size')].pivot(index='sector_code', columns='year', values='median')
    stable_stats = year_matrix.agg(['min','max','median','std','count'], axis=1).rename(columns={'median':'annual_median'})
    stable_stats.to_csv(out/'stability_by_sector.csv')
    rank_rows=[]
    for y1,y2 in zip(cfg['stability_years'][:-1],cfg['stability_years'][1:]):
        xy=year_matrix[[y1,y2]].dropna()
        rank_rows.append(dict(first=y1,second=y2,sectors=len(xy),rank_correlation=float(xy[y1].corr(xy[y2],method='spearman'))))
    pd.DataFrame(rank_rows).to_csv(out/'annual_rank_stability.csv',index=False)
    fig, axes=plt.subplots(1,3,figsize=(17,5),layout='constrained')
    axes[0].scatter(valid.raw_median,valid.market_median,alpha=.65,label='仅剔除沪深300')
    axes[0].scatter(valid.raw_median,valid.market_size_median,alpha=.65,label='再剔除中小盘相对走势')
    axes[0].plot([0,1],[0,1],'--',color='#999')
    axes[0].set(xlabel='同样本原始内部相关中位数',ylabel='剔除因素后的内部相关中位数',title='共同市场因素的影响')
    axes[0].legend(fontsize=8)
    for method,label in [('raw','原始'),('market','市场残差'),('market_size','市场+风格残差')]:
        axes[1].hist(valid[method+'_median'],bins=np.linspace(-.1,1,23),histtype='step',lw=2,label=label)
    axes[1].set(xlabel='板块内部相关中位数',ylabel='板块数',title='最近252日，同股票/日期/配对口径')
    axes[1].legend(fontsize=8)
    for proxy,label in [('cap_top3','市值前三'),('momentum60_top3','过去60日涨幅前三')]:
        tab=leaders[leaders.proxy.eq(proxy)]
        axes[2].scatter(tab.residual_lag0_corr,tab.residual_lag1_corr,alpha=.6,label=label)
    axes[2].axhline(0,color='#999',ls='--')
    axes[2].set(xlabel='代理组合与其余成员：同日残差相关',ylabel='代理前一日与其余成员当日：残差相关',title='同步代表性与一天领先性分开观察')
    axes[2].legend(fontsize=8)
    for ax in axes:
        ax.grid(alpha=.15);ax.spines[['top','right']].set_visible(False)
    savefig(fig,out,'cohesion_overview',cfg['dpi'])
    ordered=summary.sort_values('market_size_median',ascending=False,na_position='last')
    fig,axes=plt.subplots(1,3,figsize=(23,18),layout='constrained')
    for ax,positions in zip(axes,np.array_split(np.arange(len(ordered)),3)):
        part=ordered.iloc[positions]
        y=np.arange(len(part))
        for col,color,label in [('raw_median','#b7bcc2','原始'),('market_median','#73a6cf','市场残差'),('market_size_median','#c55635','市场+风格残差')]:
            ax.scatter(part[col],y,s=17,color=color,label=label)
        for yy,r in enumerate(part.itertuples()):
            if np.isfinite(r.raw_median):
                ax.plot([r.market_size_median,r.raw_median],[yy,yy],color='#ddd',lw=1,zorder=0)
            else:ax.text(0,yy,'数据不足',fontsize=7,color='#999',va='center')
        ax.set_yticks(y,[f'{r.name} {r.sector_code.replace(".SI","")}' for r in part.itertuples()],fontsize=8)
        ax.set_ylim(len(part)-.3,-.7);ax.set_xlim(-.1,1);ax.grid(axis='x',alpha=.2)
        ax.set_xlabel('板块内个股配对相关中位数');ax.legend(fontsize=8,loc='lower right')
    fig.suptitle('全部现行分类板块：剔除市场及中小盘风格前后\n最近252日；126日历史滚动估计且系数滞后一日；每对至少60个共同日',fontsize=16)
    savefig(fig,out,'cohesion_all_sectors',cfg['dpi'])
    matrix=year_matrix.reindex(ordered.sector_code)
    fig,ax=plt.subplots(figsize=(10,24),layout='constrained')
    cm=plt.get_cmap('RdBu_r').copy();cm.set_bad('#dedede')
    im=ax.imshow(matrix,aspect='auto',vmin=-1,vmax=1,cmap=cm)
    ax.set_xticks(range(len(matrix.columns)),matrix.columns)
    names=summary.set_index('sector_code').name
    ax.set_yticks(range(len(matrix)),[f'{names[c]} {c.replace(".SI","")}' for c in matrix.index],fontsize=7)
    ax.set_title('市场+中小盘风格剔除后的内部相关：逐年对照\n2026年截至9月8日；灰色=样本不足',fontsize=13)
    for i in range(len(matrix)):
        for j in range(len(matrix.columns)):
            v=matrix.iloc[i,j]
            if np.isfinite(v):ax.text(j,i,f'{v:.2f}',ha='center',va='center',fontsize=6)
    fig.colorbar(im,ax=ax,fraction=.025,label='内部配对相关中位数')
    savefig(fig,out,'annual_cohesion_heatmap',cfg['dpi'])
    headline=dict(recent_start=str(recent_dates.min().date()),recent_end=str(recent_dates.max().date()),sectors=len(valid),
                  raw_median=float(valid.raw_median.median()),market_median=float(valid.market_median.median()),
                  market_size_median=float(valid.market_size_median.median()),rank_stability=rank_rows,
                  proxies={p:{k:float(t[k].median()) for k in ['raw_lag0_corr','residual_lag0_corr','residual_lag1_corr']}
                           for p,t in leaders.groupby('proxy')})
    lines=['# 板块整体性：市场因素、稳定性与龙头代理', '',
           f'版本：`{ds.manifest["dataset_id"]}`；最近窗口 {recent_dates.min():%Y-%m-%d}—{recent_dates.max():%Y-%m-%d}。', '',
           '## 定义与边界', '',
           '- 逐股以过去126个交易日（至少60个有效日）估计带截距OLS，系数滞后一天。单因子为沪深300；双因子增加中证500减沪深300的收益。市场因子并不能穷尽全部风格及共同风险。',
           '- 相关统计要求两股同日属于同板块，前日归属也相同；不跨分类切换，不填补缺失收益。原始/单因子/双因子使用相同股票、日期有效掩码，每对至少60日。',
           '- 这是描述性研究，不是全新样本外检验。年度数字和排名变化只能说明这些历史窗口的稳定程度；不同配对的共同日期仍可能不同。',
           '- 龙头代理分别是前日成员中的前日总市值前三、截至前日60日涨幅前三，每日重排，并非主观识别的历史龙头。记录了逐日实际选中的代码。',
           '- 代理收益与剔除这三股后的其余成员等权收益比较；至少2只代理和2只其余成员有同样本收益。先选再检查当日数据，不因当日涨跌重新挑股。',
           '- 同日相关衡量代表性；前一日代理与下一日其余成员的相关仅作领先性描述，未做多重检验、收益成本或交易规则验证，不构成预测技能证明。', '',
           '## 最近窗口汇总', '',
           f'- 可统计 {len(valid)} / {len(summary)} 个板块；每个板块等权，再取板块内部相关中位数的中位数。',
           f'- 同样本原始：{headline["raw_median"]:.3f}；仅剔除市场：{headline["market_median"]:.3f}；再剔除中小盘相对走势：{headline["market_size_median"]:.3f}。', '',
           '| 龙头代理 | 同日原始相关中位数 | 同日残差相关中位数 | 前日→当日残差相关中位数 |',
           '|---|---:|---:|---:|']
    for p,v in headline['proxies'].items():
        lines.append(f'| {p} | {v["raw_lag0_corr"]:.3f} | {v["residual_lag0_corr"]:.3f} | {v["residual_lag1_corr"]:.3f} |')
    lines+=['','## 年度排名稳定性','','| 相邻年份 | 有效板块 | 内部残差相关排名的 Spearman 相关 |','|---|---:|---:|']
    for r in rank_rows:lines.append(f'| {r["first"]}→{r["second"]} | {r["sectors"]} | {r["rank_correlation"]:.3f} |')
    for title,part in [('最近窗口残差内部联动较高',valid.nlargest(12,'market_size_median')),
                       ('最近窗口残差内部联动较低',valid.nsmallest(12,'market_size_median'))]:
        lines+=['',f'## {title}','','| 板块 | 原始 | 市场残差 | 双因子残差 | 有效配对 | 年度最小—最大 |','|---|---:|---:|---:|---:|---|']
        for r in part.itertuples():
            st=stable_stats.loc[r.sector_code]
            lines.append(f'| {r.name} {r.sector_code} | {r.raw_median:.3f} | {r.market_median:.3f} | {r.market_size_median:.3f} | {r.market_size_pairs:,} | {st["min"]:.3f}—{st["max"]:.3f} |')
    lines+=['','## 核验与产物','','- 复现上一轮全部134个板块原始配对数及相关中位数，与旧报告一致。',
            f'- 单元测试用独立最小二乘复核缺失样本的一/二因子系数、当日收益不影响当日系数、物理前缀一致性；另对真实数据12只股票复核OLS，并重算真实数据历史前缀。',
            '- 对三种口径逐板块断言有效配对数一致。龙头与其余成员严格不重叠。',
            '- recent_cohesion.csv：原始/市场/双因子残差的均值、分位数、样本量。',
            '- annual_cohesion.csv、annual_rank_stability.csv、stability_by_sector.csv：年度结果与排名稳定性。',
            '- leader_representativeness.csv、leader_daily.parquet：两类代理的同日/滞后一日相关及每日成员、组合收益。',
            '- cohesion_overview、cohesion_all_sectors、annual_cohesion_heatmap：PNG/PDF 图表。', '']
    (out/'README.md').write_text('\n'.join(lines))
    (out/'summary.json').write_text(json.dumps(headline,ensure_ascii=False,indent=2))
    (out/'validation.json').write_text(json.dumps(dict(prior_reproduction_sectors=len(summary), real_ols_checks=fit_checks,
                                                     real_prefix_check=True,same_pair_counts=True,leader_rest_disjoint=True),indent=2))
    for p in [config,Path(__file__),ROOT/'src/sector_cohesion.py',ROOT/'tests/test_sector_cohesion.py']:
        shutil.copy2(p,out/p.name)
    (out/'manifest.json').write_text(json.dumps(dict(dataset_content_id=ds.manifest['content_id'],config=cfg,
        prior_summary_sha256=digest(PROJECT/cfg['prior_internal']/'summary_recent.csv'),
        artifacts={p.name:digest(p) for p in out.iterdir() if p.is_file()}),ensure_ascii=False,indent=2))
    print(json.dumps(dict(output=str(out),**headline),ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()

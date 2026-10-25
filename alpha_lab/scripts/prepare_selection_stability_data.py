"""只补新候选的执行元数据；旧输入逐字保留，不因数据缺失缩池。"""
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime, timezone
from pathlib import Path
import argparse
import json
import shutil
import sys

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
from build_selection_stability import digest, save
from portfolio import normalize_actions


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--run', type=Path, required=True)
    run = p.parse_args().run; cfg = yaml.safe_load((run / 'config.yaml').read_text())
    assert (run / 'signal_completion.json').exists()
    assert not (run / 'execution_manifest.json').exists()
    old = ROOT / cfg['parent'] / 'inputs'; out = run / 'inputs'; out.mkdir(exist_ok=True)
    codes = pd.read_csv(run / 'execution_codes.csv').ts_code.tolist()
    original = {k: pd.read_parquet(old / f'{k}.parquet') for k in ['market', 'limits', 'normalized_actions']}
    new_codes = sorted(set(codes) - set(original['market'].ts_code))
    raw = run / 'new_execution_raw'; raw.mkdir(exist_ok=True)
    requests = []
    for code in new_codes:
        for api in ['stk_limit', 'dividend']:
            path = raw / f'{code}_{api}.parquet'
            params = dict(ts_code=code)
            if api == 'stk_limit': params.update(start_date=cfg['start'].replace('-', ''), end_date=cfg['end'].replace('-', ''))
            if not path.exists():
                item = dict(code=code, api=api, params=params, requested_at=datetime.now(timezone.utc).isoformat())
                try:
                    with open('/dev/null', 'w') as sink, redirect_stdout(sink), redirect_stderr(sink):
                        from ts_client import get_pro
                        frame = get_pro().query(api, **params)
                    if frame is None or len(frame) >= (2000 if api == 'dividend' else 6000):
                        raise ValueError('空响应或达到分页上限')
                    frame.to_parquet(path, index=False)
                    item.update(rows=len(frame), sha256=digest(path), status='downloaded', received_at=datetime.now(timezone.utc).isoformat())
                except Exception as exc:
                    item.update(status='failed', error_type=type(exc).__name__)
                    requests.append(item); save(raw / 'receipt.json', requests)
                    raise RuntimeError(f'{code}/{api} 获取失败，未删股') from None
                requests.append(item); save(raw / 'receipt.json', requests)
                print(code, api, len(frame), flush=True)
    if new_codes:
        market = pd.read_parquet(ROOT / 'data/panel/daily_panel.parquet', columns=list(original['market'].columns),
                                filters=[('date', '>=', pd.Timestamp(cfg['start'])), ('date', '<=', pd.Timestamp(cfg['end'])), ('ts_code', 'in', new_codes)])
        limits = pd.concat([pd.read_parquet(raw / f'{c}_stk_limit.parquet') for c in new_codes], ignore_index=True)
        limits['date'] = pd.to_datetime(limits.trade_date)
        coverage = market.merge(limits[['date', 'ts_code', 'up_limit', 'down_limit']], on=['date', 'ts_code'], how='left', validate='one_to_one')
        assert not coverage[['up_limit', 'down_limit']].isna().any().any()
        actions = pd.concat([pd.read_parquet(raw / f'{c}_dividend.parquet') for c in new_codes], ignore_index=True)
        ex = pd.to_datetime(actions.ex_date, errors='coerce')
        actions = actions.loc[ex.between(cfg['start'], cfg['end'])].copy()
        ac = yaml.safe_load((old / 'corporate_action_components.yaml').read_text())
        # 原纠错配置只向实际含相关代码的数据应用；所有原条目保持在原输入。
        corrections = [x for x in ac.get('source_corrections', []) if x['match']['ts_code'] in new_codes]
        normalized = normalize_actions(actions, ac.get('components', []), ac.get('reference_adjustments', []), corrections)
        imp = actions.loc[actions.div_proc.eq('实施')].copy()
        imp['ex_date'] = pd.to_datetime(imp.ex_date); imp['imp_ann_date'] = pd.to_datetime(imp.imp_ann_date)
        assert imp.imp_ann_date.notna().all()
        known = imp.groupby(['ts_code', 'ex_date']).imp_ann_date.max().rename('known_date')
        normalized = normalized.merge(known, on=['ts_code', 'ex_date'], how='left', validate='one_to_one')
        assert normalized.known_date.notna().all() and normalized.known_date.le(normalized.record_date).all()
        added = dict(market=market, limits=limits, normalized_actions=normalized)
    else:
        added = {k: original[k].iloc[:0] for k in original}
    for key in original:
        both = pd.concat([original[key], added[key]], ignore_index=True)
        pd.testing.assert_frame_equal(both.iloc[:len(original[key])].reset_index(drop=True), original[key].reset_index(drop=True), check_exact=True)
        both.to_parquet(out / f'{key}.parquet', index=False)
    for name in ['basics.parquet', 'st.parquet', 'execution_config.yaml', 'symmetric_phase_research.yaml', 'corporate_action_components.yaml']:
        shutil.copy2(old / name, out / name)
    save(run / 'execution_input_completion.json', dict(new_codes=new_codes, old_inputs_exact=True,
        raw_provider='configured Tushare provider', downloaded_now_not_historical_retrieval=True,
        source_sha256=digest(Path(__file__)), input_sha256={str(p.relative_to(run)): digest(p) for p in out.iterdir()},
        new_raw_sha256={p.name: digest(p) for p in raw.glob('*.parquet')}))
    print('execution inputs ready', len(codes), 'codes, new', len(new_codes), flush=True)


if __name__ == '__main__': main()

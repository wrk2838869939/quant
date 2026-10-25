"""分治模型的成熟边界、日期权重、收缩与输入顺序检验。"""
from copy import deepcopy
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from divide_selection import fit_divided, phase_features, policy_predictions


def fixture():
    rng = np.random.default_rng(9)
    dates = pd.date_range('2016-01-01', periods=12, freq='MS')
    rows = []
    for month, date in enumerate(dates):
        for phase, (recent, prior) in enumerate([(0.2, -0.1), (0.2, 0.1), (0.1, 0.2)]):
            for stock in range(3):
                x = rng.uniform(-0.5, 0.5)
                rows.append(dict(date=date, ts_code=f'{phase}{stock}', trend=True,
                    mom60=recent, mom120=(1+recent)*(1+prior)-1, x=x,
                    label_end_60=date+pd.Timedelta(days=90), entry_missing_60=False,
                    ret_60=0.04+phase*0.01+x*0.02+month*0.001,
                    mae_60=-0.1+x*0.01))
    cfg = dict(features=['x'], phases=['recovery','accelerating','decelerating'],
        forecast=dict(horizon=60, alpha=1., bootstrap_reps=12, block_dates=2, seed=1, min_dates=4),
        policies=['global_return','phase_experts','shared_experts'],
        entry=dict(min_expected_return=0.005,max_expected_adverse=0.15))
    return pd.DataFrame(rows), cfg


def all_models(model):
    return [model.global_model, model.shared_model, model.constant_model, *model.local_models.values()]


class DivideSelectionTests(unittest.TestCase):
    def test_phases_use_adjacent_returns_and_known_boundary_rules(self):
        f = pd.DataFrame(dict(mom60=[.2,.2,.1,0.], mom120=[.08,.32,.32,0.], x=[1.,2.,3.,4.]))
        out, cols = phase_features(f,['x'])
        self.assertEqual(out.phase.tolist(),['recovery','accelerating','decelerating','recovery'])
        self.assertEqual(len(cols),5)
        self.assertEqual(out.loc[1,'phase_accelerating__x'],2.)
        self.assertEqual(out.loc[0,'phase_decelerating__x'],0.)

    def test_immature_labels_cannot_change_fits(self):
        frame,cfg=fixture();baseline=fit_divided(frame,cfg,'2017-12-31')
        future=frame.iloc[:9].copy();future.label_end_60=pd.Timestamp('2023-01-01')
        future.ret_60=1e10;future.mae_60=-1e10
        changed=fit_divided(pd.concat([frame,future],ignore_index=True),cfg,'2017-12-31')
        for before,after in zip(all_models(baseline),all_models(changed)):
            pd.testing.assert_frame_equal(before.coefficients,after.coefficients)
            np.testing.assert_array_equal(before.bootstrap_coefficients,after.bootstrap_coefficients)
            self.assertEqual(before.training_input_hash,after.training_input_hash)
            self.assertLessEqual(after.last_label,pd.Timestamp('2017-12-31'))

    def test_maturity_boundary_includes_equal_end_date_only(self):
        frame,cfg=fixture();cut=frame.label_end_60.sort_values().unique()[7]
        model=fit_divided(frame,cfg,cut)
        wanted=frame[frame.label_end_60.le(cut)]
        self.assertEqual(model.global_model.metadata['exclusion_counts']['fitted_rows'],len(wanted))
        self.assertEqual(model.global_model.last_label,pd.Timestamp(cut))

    def test_training_and_prediction_are_row_permutation_invariant(self):
        frame,cfg=fixture();a=fit_divided(frame,cfg,'2017-12-31')
        shuffled=frame.sample(frac=1,random_state=88)
        b=fit_divided(shuffled,cfg,'2017-12-31')
        for left,right in zip(all_models(a),all_models(b)):
            pd.testing.assert_frame_equal(left.coefficients,right.coefficients)
            self.assertEqual(left.training_input_hash,right.training_input_hash)
        pd.testing.assert_frame_equal(a.predict(frame),b.predict(shuffled).reindex(frame.index))
        pd.testing.assert_frame_equal(policy_predictions(frame,a,cfg),policy_predictions(shuffled,b,cfg).reindex(frame.index))

    def test_repeating_complete_date_preserves_all_model_weights(self):
        frame,cfg=fixture();a=fit_divided(frame,cfg,'2017-12-31')
        expanded=pd.concat([frame]+[frame[frame.date.eq(frame.date.min())]]*7,ignore_index=True)
        b=fit_divided(expanded,cfg,'2017-12-31')
        for left,right in zip(all_models(a),all_models(b)):
            np.testing.assert_allclose(left.coefficients,right.coefficients,atol=1e-14)
            np.testing.assert_allclose(left.bootstrap_coefficients,right.bootstrap_coefficients,atol=1e-14)

    def test_global_shared_and_local_fit_independent_date_weighted_equations(self):
        frame,cfg=fixture();model=fit_divided(frame,cfg,'2017-12-31');x,_=phase_features(frame,cfg['features'])
        pairs=[(model.global_model,x),(model.shared_model,x),(model.constant_model,x)]
        pairs.extend((m,x[x.phase.eq(phase)]) for phase,m in model.local_models.items())
        for fitted,data in pairs:
            xx=np.column_stack([np.ones(len(data)),data[list(fitted.feature_cols)].to_numpy()])
            yy=data[['ret_60','mae_60']].to_numpy().copy();yy[:,1]*=-1
            weights=(1/data.groupby('date').date.transform('size')).to_numpy()
            penalty=np.eye(xx.shape[1])*cfg['forecast']['alpha'];penalty[0,0]=0
            expected=np.linalg.solve(xx.T@(xx*weights[:,None])+penalty,xx.T@(yy*weights[:,None]))
            np.testing.assert_allclose(fitted.coefficients,expected,atol=1e-14)

    def test_sparse_phase_falls_back_to_same_global_forecast(self):
        frame,cfg=fixture();x,_=phase_features(frame,cfg['features'])
        frame=frame[~(x.phase.eq('recovery') & frame.date.gt('2016-02-01'))].copy()
        model=fit_divided(frame,cfg,'2017-12-31')
        self.assertTrue(model.metadata['local']['recovery']['fallback'])
        pred=model.predict(frame);group=pred.phase.eq('recovery')
        np.testing.assert_array_equal(pred.loc[group,'phase_experts__expected_return'],pred.loc[group,'global_return__expected_return'])

    def test_rule_ranking_is_not_used_as_expected_return(self):
        frame,cfg=fixture();cfg=deepcopy(cfg);cfg['policies']=['low_turnover','continuous_trend'];cfg['continuous_score']={'x':1.}
        frame['x_turnover']=frame.x;model=fit_divided(frame,cfg,'2017-12-31');pred=policy_predictions(frame,model,cfg)
        np.testing.assert_allclose(pred.low_turnover__score,-frame.x)
        np.testing.assert_allclose(pred.continuous_trend__score,frame.x)
        np.testing.assert_allclose(pred.low_turnover__expected_return,pred.constant__expected_return)
        self.assertEqual(pred.low_turnover__expected_return.nunique(),1)


if __name__=='__main__':unittest.main()

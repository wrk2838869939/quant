# 历史实验脚本归档：运行依赖原研究环境及未发布输入。
"""特征预处理、标签成熟期和非单调模糊表达测试。"""
from pathlib import Path
import unittest
import numpy as np
import pandas as pd
import torch
import yaml
from model import TrainScaler, purged_history, FuzzyNetwork, fit_network, predict_network


class ModelTests(unittest.TestCase):
    def setUp(self):
        self.cfg = yaml.safe_load((Path(__file__).parent / 'config.yaml').read_text())
        torch.set_num_threads(1)

    def test_train_only_scaling(self):
        scaler = TrainScaler(self.cfg).fit([[0, np.nan], [2, 4], [4, 6]])
        before = scaler.transform([[np.nan, 5]])
        scaler.transform([[100000, -100000]])
        np.testing.assert_array_equal(before, scaler.transform([[np.nan, 5]]))
        np.testing.assert_allclose(scaler.median, [2, 5])

    def test_label_maturity_and_embargo(self):
        calendar = pd.bdate_range('2020-01-01', periods=60)
        frame = pd.DataFrame(dict(date=calendar[:20], end_date=calendar[20:40], complete=True, target=.1))
        history, cutoff = purged_history(frame, calendar[30], calendar, self.cfg)
        self.assertEqual(cutoff, calendar[25])
        self.assertEqual(len(history), 5)
        self.assertTrue((history.end_date < cutoff).all())

    def test_non_monotonic_and_finite(self):
        x = np.array([[-1.], [0.], [1.]], dtype='float32')
        model = FuzzyNetwork(x, 3, 17, self.cfg['fnn'])
        with torch.no_grad():
            model.centers.copy_(torch.tensor([[-1.], [0.], [1.]]))
            model.raw_width.fill_(-3)
            model.consequents[:, -1] = torch.tensor([0., 1., 0.])
        pred = predict_network(model, x)
        self.assertGreater(pred[1], pred[0] + .5)
        self.assertGreater(pred[1], pred[2] + .5)
        self.assertTrue(np.isfinite(predict_network(model, np.array([[1000.]], dtype='float32'))).all())

    def test_training_updates_and_reproducible(self):
        x = np.linspace(-2, 2, 40).astype('float32')[:, None]
        y = np.exp(-x[:, 0] ** 2).astype('float32')
        # 无噪声合成函数用于检验表达能力与更新；正则化偏差不属于此断言。
        test_cfg = {**self.cfg['fnn'], 'weight_decay': 0.0}
        args = (x, y, np.ones(40), 4, .01, 17, test_cfg)
        first, _, _, _ = fit_network(*args, epochs=150)
        second, _, _, _ = fit_network(*args, epochs=150)
        a, b = predict_network(first, x), predict_network(second, x)
        np.testing.assert_allclose(a, b)
        self.assertLess(np.mean((a-y) ** 2), .02)


if __name__ == '__main__':
    unittest.main()

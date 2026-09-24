# 历史实验脚本归档：运行依赖原研究环境及未发布输入。
"""验证随机互换与每轮严格互斥，避免无意重复采样。"""
import unittest
import pandas as pd
from random_split import make_assignments


class RandomSplitTests(unittest.TestCase):
    def test_exact_halves_and_disjoint_roles(self):
        ids = list(range(330))
        result = make_assignments(ids, 2026092301)
        for _, group in result.groupby('round'):
            self.assertEqual(group.role.value_counts().to_dict(),
                             {'train': 165, 'test': 83, 'validation': 82})
            self.assertEqual(set(group.event_id), set(ids))
            self.assertFalse(group.event_id.duplicated().any())
        train1 = set(result[(result['round'] == 1) & (result.role == 'train')].event_id)
        train2 = set(result[(result['round'] == 2) & (result.role == 'train')].event_id)
        self.assertFalse(train1 & train2)
        self.assertEqual(train1 | train2, set(ids))
        tests = result[result.role == 'test'].event_id
        self.assertTrue(tests.is_unique)

    def test_reproducible_without_input_order(self):
        ids = list(range(330))
        pd.testing.assert_frame_equal(make_assignments(ids, 123), make_assignments(ids[::-1], 123))
        self.assertFalse(make_assignments(ids, 123).equals(make_assignments(ids, 124)))


if __name__ == '__main__':
    unittest.main()

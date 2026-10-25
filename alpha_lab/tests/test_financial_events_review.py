"""独立审阅发现的事件ID稳定性、原件出处和累计口径漏洞。"""

import hashlib
from pathlib import Path
import sys
import tempfile
import unittest

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from financial_events import append_events, asof_events, normalize_events, quarter_difference, verify_event_sources


def record(value, published='2021-04-30', field='cfo_yuan'):
    """完全独立的最小事件夹具，不引用root测试的构造器。"""
    return dict(ts_code='600000.SH', period_end='2021-03-31', field=field,
        scope='consolidated', published_date=published, value=value, unit='CNY',
        basis_id='issuer_original_cumulative', source_path='/fixture/report.pdf',
        source_sha256='b' * 64, pdf_page=1, source_token=str(value),
        source_url='https://issuer.example/report.pdf', event_role='disclosed_value')


class FinancialEventsReviewTests(unittest.TestCase):
    def test_fractional_append_does_not_change_existing_integer_event_id(self):
        first = normalize_events(pd.DataFrame([record(10)]))
        event_id = first.event_id.iloc[0]
        result = append_events(first, pd.DataFrame([record(10.5, '2021-05-10')]))
        self.assertEqual(len(result), 2)
        self.assertIn(event_id, result.event_id.to_list())
        request = pd.DataFrame([dict(date='2021-05-01', ts_code='600000.SH',
            period_end='2021-03-31', field='cfo_yuan', scope='consolidated')])
        original = asof_events(first, request)
        appended = asof_events(result, request)
        pd.testing.assert_frame_equal(original, appended)

    def test_whitespace_source_token_cannot_verify_an_unrelated_pdf(self):
        import pymupdf
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'source.pdf'
            with pymupdf.open() as document:
                page = document.new_page()
                page.insert_text((72, 72), 'This page does not disclose the event number.')
                document.save(path)
            event = record(123456.78)
            event.update(source_path=str(path), source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                         source_token=' \t\n ')
            with self.assertRaises(ValueError):
                verify_event_sources(pd.DataFrame([event]))

    def test_balance_sheet_stock_cannot_be_treated_as_quarterly_flow(self):
        current = dict(status='available', ts_code='600000.SH', field='total_assets_yuan',
            scope='consolidated', basis_id='original', unit='CNY', period_end='2021-09-30',
            value=100., published_date=pd.Timestamp('2021-10-29'), lineage='current')
        previous = dict(current, period_end='2021-06-30', value=80.,
                        published_date=pd.Timestamp('2021-08-20'), lineage='previous')
        with self.assertRaises(ValueError):
            quarter_difference(current, previous)


if __name__ == '__main__':
    unittest.main()

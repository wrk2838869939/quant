"""索引边界：真实指针、只读盘点、可重建元数据与安全HTML。"""
import importlib.util
import json
import os
from pathlib import Path
import sqlite3
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('workspace_index', ROOT / 'scripts/build_workspace_index.py')
INDEX = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(INDEX)


class WorkspaceIndexTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / 'alpha_lab'
        for name in ['docs', 'outputs/demo/old', 'outputs/demo/new', 'configs', 'scripts', 'experiments']:
            (self.root / name).mkdir(parents=True)
        (self.root / 'scripts/workspace_index.html').write_text((ROOT / 'scripts/workspace_index.html').read_text())
        self.config = dict(quick_links=[], pointers=[dict(label='正式结果', path='experiments/LATEST')], protected_output_groups=['demo'], topics=[])
        (self.root / 'configs/workspace_index.json').write_text(json.dumps(self.config))
        for run in ['old', 'new']:
            (self.root / 'outputs/demo' / run / 'manifest.json').write_text(json.dumps(dict(status='complete', summary=dict(asof='2026-10-22', record_origin='current_policy_scan'))))
        (self.root / 'experiments/LATEST').write_text('alpha_lab/outputs/demo/old\n')

    def test_latest_comes_from_pointer_and_rejects_unfinished_or_outside(self):
        value = INDEX.resolve_pointer(self.root, 'experiments/LATEST')
        self.assertTrue(value['valid'])
        self.assertEqual(value['target'], 'outputs/demo/old')
        (self.root / 'outputs/demo/old/manifest.json').write_text('{"status":"building"}')
        self.assertFalse(INDEX.resolve_pointer(self.root, 'experiments/LATEST')['valid'])
        (self.root / 'experiments/LATEST').write_text('../outside')
        self.assertFalse(INDEX.resolve_pointer(self.root, 'experiments/LATEST')['valid'])

    def test_inventory_does_not_follow_links_and_deduplicates_inodes(self):
        p = self.root / 'outputs/demo/old/data.bin'
        p.write_bytes(b'x' * 5000)
        os.link(p, self.root / 'outputs/demo/new/data.bin')
        outside = Path(self.temp.name) / 'outside'
        outside.mkdir()
        (outside / 'secret.bin').write_bytes(b'y' * 100_000)
        (self.root / 'outputs/demo/link').symlink_to(outside, target_is_directory=True)
        result = INDEX.inventory(self.root)
        totals = result['totals']['outputs']
        self.assertEqual(totals['files'], 4)
        self.assertEqual(totals['symlinks'], 1)
        self.assertFalse(any('secret.bin' in x['path'] for x in result['largest_files']))
        allocated = p.stat().st_blocks * 512
        allocated += sum((self.root / 'outputs/demo' / run / 'manifest.json').stat().st_blocks * 512 for run in ['old', 'new'])
        self.assertEqual(totals['allocated_bytes'], allocated)

    def test_build_preserves_originals_and_handles_html_in_titles(self):
        doc = self.root / 'docs/research.md'
        doc.write_text('# 养殖 </script><script>alert(1)</script>\n\n## 价格形态\n')
        original = doc.read_bytes()
        catalog = INDEX.build(self.root, True)
        self.assertEqual(doc.read_bytes(), original)
        page = (self.root / 'INDEX.html').read_text()
        payload = page.split('<script id="catalog" type="application/json">', 1)[1].split('</script>', 1)[0]
        parsed = json.loads(payload)
        self.assertEqual(len(parsed['records']), len(catalog['records']))
        self.assertNotIn('</script>', payload)
        with sqlite3.connect(self.root / 'catalog/index.sqlite') as con:
            self.assertEqual(con.execute('SELECT count(*) FROM entries WHERE instr(search_text, ?) > 0', ('价格形态',)).fetchone()[0], 1)
        storage_before = (self.root / 'catalog/storage.json').read_bytes()
        (self.root / 'outputs/demo/later').mkdir()
        refreshed = INDEX.build(self.root, False)
        later = next(r for r in refreshed['records'] if r['path'] == 'outputs/demo/later')
        self.assertIsNone(later['bytes'])
        self.assertEqual(storage_before, (self.root / 'catalog/storage.json').read_bytes())
        self.assertFalse(refreshed['issues'])


if __name__ == '__main__':
    unittest.main()

#!/usr/bin/env python3
"""生成可搜索目录及存储清单；不移动、删除或改写研究产物。"""
import argparse
import csv
from datetime import datetime, timezone
import fnmatch
import heapq
import html
import json
import os
from pathlib import Path
import re
import sqlite3
import tempfile
from urllib.parse import quote

DEFAULT_ROOT = Path(__file__).resolve().parents[1]
SKIP = {'.ipynb_checkpoints', '__pycache__', '.git'}
GENERATED = {'docs/INDEX.md', 'outputs/INDEX.md'}
REPORTS = ('REPORT.html', 'REPORT.md', 'report.html', 'report.md', 'TECHNICAL_REPORT.html', 'README.md', 'summary.json', 'manifest.json')


def now():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def atomic_text(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix='.' + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            stream.write(text)
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def json_read(path):
    try:
        if path.stat().st_size > 8_000_000:
            return {}
        value = json.loads(path.read_text(encoding='utf-8'))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError, UnicodeError):
        return {}


def link(path):
    return quote(str(path), safe='/')


def size(value):
    if value is None:
        return '未盘点'
    for unit in ('B', 'KiB', 'MiB', 'GiB', 'TiB'):
        if value < 1024 or unit == 'TiB':
            return f'{value:.1f} {unit}'
        value /= 1024


def report_path(root, relative):
    path = root / relative
    if path.is_file():
        return relative
    for name in REPORTS:
        if (path / name).is_file():
            return (Path(relative) / name).as_posix()
    return relative + '/'


def describe_run(path):
    manifest = json_read(path / 'manifest.json')
    summary = json_read(path / 'summary.json')
    config = manifest.get('config', {})
    summary = summary or manifest.get('summary', {})
    summary = summary if isinstance(summary, dict) else {}
    config = config if isinstance(config, dict) else {}
    date = summary.get('asof') or manifest.get('asof') or config.get('asof') or ''
    return str(manifest.get('status', '未声明')), str(date), str(summary.get('record_origin', ''))


def resolve_pointer(root, relative):
    """只从明确指针读最新记录，拒绝不存在、越界或未完成目录。"""
    path = root / relative
    try:
        value = path.read_text(encoding='utf-8').strip()
        if not value or '\n' in value:
            raise ValueError('指针内容为空或不是单一路径')
        target = Path(value)
        if not target.is_absolute():
            target = root.parent / target if target.parts[0] == root.name else root / target
        target = target.resolve()
        rel = target.relative_to(root.resolve()).as_posix()
        if not target.exists():
            raise ValueError('指向不存在的目录')
        status, date, origin = describe_run(target)
        if status not in {'complete', 'completed', 'passed', 'success'}:
            raise ValueError(f'产物完成状态未获确认：{status}')
        return dict(valid=True, target=rel, report=report_path(root, rel), status=status, asof=date, origin=origin)
    except (OSError, ValueError) as error:
        return dict(valid=False, error=str(error), target='', report='', status='', asof='', origin='')


def topic_for(relative, config):
    for item in config['topics']:
        if any(fnmatch.fnmatchcase(relative, pattern) for pattern in item['patterns']):
            return item['name']
    return '其他研究与规范'


def inventory(root):
    """仅stat文件；不读取大数据，不跟随链接。占用按inode去重。"""
    nodes, totals, seen, largest, extensions = {}, {}, set(), [], {}
    def blank():
        return dict(files=0, directories=0, logical_bytes=0, allocated_bytes=0, symlinks=0)
    for section in ('docs', 'outputs'):
        totals[section] = blank()
        stack = [root / section]
        while stack:
            directory = stack.pop()
            with os.scandir(directory) as entries:
                for entry in entries:
                    rel = Path(entry.path).relative_to(root).as_posix()
                    parts = rel.split('/')
                    keys = [section]
                    if section == 'outputs' and len(parts) >= 2:
                        keys.append('/'.join(parts[:2]))
                        if len(parts) >= 3:
                            keys.append('/'.join(parts[:3]))
                    rows = [nodes.setdefault(k, blank()) for k in keys]
                    if entry.is_symlink():
                        for row in rows:
                            row['symlinks'] += 1
                        continue
                    if entry.is_dir(follow_symlinks=False):
                        for row in rows:
                            row['directories'] += 1
                        stack.append(Path(entry.path))
                        continue
                    if not entry.is_file(follow_symlinks=False) or rel in GENERATED:
                        continue
                    stat = entry.stat(follow_symlinks=False)
                    inode = (stat.st_dev, stat.st_ino)
                    allocated = stat.st_blocks * 512 if inode not in seen else 0
                    seen.add(inode)
                    for row in rows:
                        row['files'] += 1
                        row['logical_bytes'] += stat.st_size
                        row['allocated_bytes'] += allocated
                    ext = Path(entry.name).suffix.lower() or '(none)'
                    extensions[ext] = extensions.get(ext, 0) + 1
                    item = (stat.st_size, rel)
                    if len(largest) < 30:
                        heapq.heappush(largest, item)
                    elif item > largest[0]:
                        heapq.heapreplace(largest, item)
        totals[section] = nodes.get(section, blank())
        print(f'盘点 {section}: {totals[section]["files"]:,} 文件', flush=True)
    return dict(generated_at=now(), nodes=nodes, totals=totals, extensions=extensions,
                largest_files=[dict(bytes=n, path=p) for n, p in sorted(largest, reverse=True)],
                accounting='文件逻辑大小按路径累计；分配空间按inode去重，跨目录硬链接仅首次出现计入；不跟随软链接；不含生成索引。')


def source_references(root):
    """发现源码中的静态输出族引用；没有命中不等于可安全删除。"""
    refs = {}
    pattern = re.compile(r'outputs/([A-Za-z0-9_]+)')
    for section in ('configs', 'scripts', 'src', 'experiments', 'datakit'):
        for base, dirs, files in os.walk(root / section):
            dirs[:] = [d for d in dirs if d not in SKIP]
            for name in files:
                p = Path(base) / name
                if p.suffix not in {'.py', '.yaml', '.yml', '.json', '.md'} or p.is_symlink() or p.stat().st_size > 2_000_000:
                    continue
                if p.name in {'build_workspace_index.py', 'workspace_index.json'}:
                    continue
                for group in set(pattern.findall(p.read_text(encoding='utf-8', errors='replace'))):
                    refs.setdefault(group, set()).add(p.relative_to(root).as_posix())
    return {key: sorted(value) for key, value in refs.items()}


def metadata(root, config, storage):
    records = []
    pointers = [dict(item, **resolve_pointer(root, item['path'])) for item in config['pointers']]
    current = {p['target']: p['label'] for p in pointers if p['valid']}
    refs = source_references(root)
    def add(path, kind, title='', snippet=''):
        relative = path.relative_to(root).as_posix()
        status, asof, origin = describe_run(path) if path.is_dir() else ('文档记录', '', '')
        match = re.search(r'(?<!\d)(20\d{6})(?!\d)', relative)
        date_hint = match[1] if match else ''
        stats = storage.get('nodes', {}).get(relative, {})
        group = relative.split('/')[1] if relative.startswith('outputs/') else ''
        record = dict(path=relative, kind=kind, title=title or path.name,
            topic=topic_for(relative, config), status=status, asof=asof,
            date_hint=date_hint, origin=origin, report=report_path(root, relative),
            current=current.get(relative, ''), protected=group in config['protected_output_groups'],
            file_count=stats.get('files'), bytes=stats.get('allocated_bytes'),
            references=refs.get(group, []) if kind == 'group' else [], snippet=snippet)
        records.append(record)
    for base, dirs, files in os.walk(root / 'docs'):
        dirs[:] = sorted(d for d in dirs if d not in SKIP)
        for name in sorted(files):
            path = Path(base) / name
            if path.suffix.lower() not in {'.md', '.html', '.pdf'} or path.is_symlink() or path.relative_to(root).as_posix() in GENERATED:
                continue
            if path.suffix.lower() == '.md':
                text = path.read_text(encoding='utf-8', errors='replace')
                headings = re.findall(r'^#{1,3}\s+(.+)', text, re.M)
                add(path, 'doc', headings[0] if headings else path.stem, ' '.join(headings)[:3000])
            else:
                add(path, 'doc', path.stem)
    for group in sorted((root / 'outputs').iterdir()):
        if not group.is_dir() or group.is_symlink() or group.name in SKIP:
            continue
        add(group, 'group')
        for run in sorted(group.iterdir()):
            if run.is_dir() and not run.is_symlink() and run.name not in SKIP:
                add(run, 'run', group.name + ' / ' + run.name)
    quick = [dict(item, exists=(root / item['path']).is_file()) for item in config['quick_links']]
    return dict(generated_at=now(), storage_generated_at=storage.get('generated_at', '尚未盘点'),
                records=records, pointers=pointers, quick_links=quick,
                storage_totals=storage.get('totals', {}),
                issues=[p['label'] + ': ' + p['error'] for p in pointers if not p['valid']] +
                       [q['label'] + ': 文件不存在' for q in quick if not q['exists']])


def write_database(root, catalog):
    target = root / 'catalog/index.sqlite'
    fd, name = tempfile.mkstemp(prefix='.index-', dir=target.parent)
    os.close(fd)
    try:
        with sqlite3.connect(name) as con:
            con.execute('CREATE TABLE entries(path TEXT PRIMARY KEY, kind TEXT, title TEXT, topic TEXT, status TEXT, asof TEXT, origin TEXT, current TEXT, report TEXT, file_count INTEGER, bytes INTEGER, search_text TEXT)')
            for r in catalog['records']:
                con.execute('INSERT INTO entries VALUES(?,?,?,?,?,?,?,?,?,?,?,?)', (*[r[k] for k in ['path','kind','title','topic','status','asof','origin','current','report','file_count','bytes']], ' '.join(str(r[k]) for k in ['path','title','topic','asof','current','snippet'])))
            con.execute('CREATE INDEX by_kind ON entries(kind)')
            con.execute('CREATE INDEX by_date ON entries(asof)')
            con.execute('CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT)')
            con.execute('INSERT INTO metadata VALUES(?,?)', ('generated_at', catalog['generated_at']))
        os.replace(name, target)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def md_escape(text):
    return str(text).replace('|', '\\|').replace('\n', ' ')


def write_pages(root, catalog, storage):
    def md_link(label, path, prefix=''):
        return f'[{md_escape(label)}]({link(prefix + path)})'
    front = ['# alpha_lab 工作台', '', '从这里找当前策略、最新日报、研究文档和历史产物。索引只证明文件位置，不认证研究结论。', '',
             md_link('可搜索网页', 'INDEX.html') + ' · ' + md_link('文档目录', 'docs/INDEX.md') + ' · ' + md_link('产物目录', 'outputs/INDEX.md') + ' · ' + md_link('存储清单', 'catalog/STORAGE.md'), '', '## 当前入口', '']
    for q in catalog['quick_links']:
        front.append('- ' + md_link(q['label'], q['path']))
    front += ['', '## 明确指针所指的最新记录', '', '| 入口 | 数据日期 | 记录性质 |', '|---|---|---|']
    for p in catalog['pointers']:
        front.append('| ' + (md_link(p['label'], p['report']) if p['valid'] else md_escape(p['label'] + '：' + p['error'])) + f' | {p["asof"] or "未声明"} | {p["origin"] or p["status"]} |')
    front += ['', '历史规则复核不会替代正式候选；不按目录名或修改时间猜测最新结果。', '', '## 更新与检索', '',
              '```bash', 'python alpha_lab/scripts/build_workspace_index.py build', 'python alpha_lab/scripts/build_workspace_index.py search "养殖"', 'python alpha_lab/scripts/build_workspace_index.py search "sector_live" --kind run', 'python alpha_lab/scripts/build_workspace_index.py build --inventory', '```', '',
              f'元数据更新：{catalog["generated_at"]}；容量盘点：{catalog["storage_generated_at"]}。普通build复用容量清单，新增目录容量显示“未盘点”。']
    atomic_text(root / 'INDEX.md', '\n'.join(front) + '\n')
    for section, kinds, title in [('docs', {'doc'}, '研究文档索引'), ('outputs', {'group','run'}, '运行与产物索引')]:
        lines = [f'# {title}', '', '[返回工作台](../INDEX.md) · [搜索与筛选](../INDEX.html)', '', '完成状态来自清单声明，并不等于策略有效、可交易或产物哈希已核验。历史结果保留原路径。', '']
        for topic in sorted({r['topic'] for r in catalog['records'] if r['kind'] in kinds}):
            lines += [f'## {topic}', '', '| 名称 / 路径 | 类型 | 数据日期 | 状态 |', '|---|---|---|---|']
            selected = [r for r in catalog['records'] if r['kind'] in kinds and r['topic'] == topic]
            for r in sorted(selected, key=lambda x: (x['asof'], x['path']), reverse=True):
                lines.append(f'| {md_link(r["title"], r["report"], "../")} | {r["kind"]} | {r["asof"] or "未声明"} | {md_escape(r["current"] or r["status"])} |')
            lines.append('')
        atomic_text(root / section / 'INDEX.md', '\n'.join(lines))
    groups = sorted([r for r in catalog['records'] if r['kind'] == 'group'], key=lambda r: r['bytes'] or 0, reverse=True)
    lines = ['# 存储盘点与归档候选', '', '[返回工作台](../INDEX.md) · [管理规则](../docs/WORKSPACE_ORGANIZATION.md)', '',
             f'盘点时间：{catalog["storage_generated_at"]}。{storage.get("accounting", "尚未盘点")}', '',
             '“有静态引用”仅为代码/配置文本线索；未发现引用不代表可删除。表中没有自动认定可删目录，本轮不迁移、不压缩原件。', '',
             '| 产物族 | 文件数 | 分配空间 | 当前链路保护 | 静态引用文件数 |', '|---|---:|---:|---|---:|']
    for r in groups:
        lines.append(f'| {md_link(r["title"], r["path"]+"/", "../")} | {r["file_count"] if r["file_count"] is not None else "未盘点"} | {size(r["bytes"])} | {"是" if r["protected"] else "未列入当前链路"} | {len(r["references"])} |')
    lines += ['', '## 最大单文件', '', '| 路径 | 逻辑大小 |', '|---|---:|']
    for item in storage.get('largest_files', []):
        lines.append(f'| {md_link(item["path"], item["path"], "../")} | {size(item["bytes"])} |')
    atomic_text(root / 'catalog/STORAGE.md', '\n'.join(lines)+'\n')
    payload = json.dumps(catalog, ensure_ascii=False).replace('<', '\\u003c').replace('>', '\\u003e').replace('&', '\\u0026')
    template = (root / 'scripts/workspace_index.html').read_text(encoding='utf-8')
    atomic_text(root / 'INDEX.html', template.replace('__CATALOG_JSON__', payload))


def build(root, full):
    config = json_read(root / 'configs/workspace_index.json')
    if not config:
        raise ValueError('索引配置不存在或无效')
    (root / 'catalog').mkdir(exist_ok=True)
    storage_path = root / 'catalog/storage.json'
    storage = inventory(root) if full else json_read(storage_path)
    if full:
        atomic_text(storage_path, json.dumps(storage, ensure_ascii=False, indent=2))
    catalog = metadata(root, config, storage)
    atomic_text(root / 'catalog/index.json', json.dumps(catalog, ensure_ascii=False, indent=2))
    write_database(root, catalog)
    write_pages(root, catalog, storage)
    print(json.dumps(dict(records=len(catalog['records']), issues=catalog['issues'], entry=str(root/'INDEX.html')), ensure_ascii=False))
    return catalog


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    sub = parser.add_subparsers(dest='command', required=True)
    b = sub.add_parser('build')
    b.add_argument('--inventory', action='store_true', help='重算容量和文件数，扫描全部docs/outputs；普通build只更新元数据')
    s = sub.add_parser('search')
    s.add_argument('query')
    s.add_argument('--kind', choices=['doc','run','group'])
    s.add_argument('--limit', type=int, default=20)
    args = parser.parse_args()
    if args.command == 'build':
        build(args.root.resolve(), args.inventory)
    else:
        db = args.root / 'catalog/index.sqlite'
        if not db.exists():
            parser.error('请先运行build创建索引')
        terms = args.query.split()
        where = ['instr(lower(search_text), lower(?)) > 0' for _ in terms]
        params = list(terms)
        if args.kind:
            where.append('kind=?')
            params.append(args.kind)
        sql = 'SELECT kind, asof, title, path FROM entries WHERE ' + (' AND '.join(where) or '1') + ' ORDER BY (current != "") DESC, asof DESC, path LIMIT ?'
        with sqlite3.connect(f'{db.resolve().as_uri()}?mode=ro', uri=True) as con:
            for row in con.execute(sql, [*params, max(1, args.limit)]):
                print('\t'.join(str(v) for v in row))


if __name__ == '__main__':
    main()

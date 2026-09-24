"""研究快照：复制数据与查询源码，校验内容和运行依赖，禁止覆盖已有版本。"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import tempfile
from datetime import datetime, timezone


TRACKED_DIRS = ("datasets", "src", "configs")
TRACKED_FILES = ("build_manifest.json", "pyproject.toml", "README.md",
                 "RESEARCH_READINESS_20260920.md", "FROZEN_RESEARCH.md")


def runtime_versions():
    """记录实际使用的解释器和数值计算依赖。"""
    result = {"python": platform.python_version()}
    for package in ("pandas", "numpy", "pyarrow", "joblib", "PyYAML"):
        result[package] = importlib.metadata.version(package)
    return result


def tracked_paths(root):
    """只追踪研究输入；缓存、输出和字节码不属于冻结输入。"""
    root = Path(root)
    result = []
    for name in TRACKED_DIRS:
        directory = root / name
        if directory.is_symlink():
            raise ValueError(f"冻结输入不能是软链接: {directory}")
        for path in directory.rglob("*"):
            rel = path.relative_to(root)
            if "__pycache__" in rel.parts or rel.parts[:2] == ("datasets", "cache"):
                continue
            if path.is_symlink():
                raise ValueError(f"冻结输入不能是软链接: {path}")
            if path.is_file():
                result.append(path)
    for name in TRACKED_FILES:
        path = root / name
        if path.is_symlink():
            raise ValueError(f"冻结输入不能是软链接: {path}")
        if path.is_file():
            result.append(path)
    return sorted(result)


def file_record(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"bytes": Path(path).stat().st_size, "sha256": digest.hexdigest()}


def inventory(root):
    root = Path(root)
    return {str(path.relative_to(root)): file_record(path) for path in tracked_paths(root)}


def content_id(files):
    return hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()


def create_snapshot(source, destination):
    """独立复制后核验源与副本一致，再原子发布目录；不使用可串改的硬链接。"""
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if destination.exists():
        raise FileExistsError(f"版本已存在，不能覆盖: {destination}")
    if destination == source or source.is_relative_to(destination):
        raise ValueError("目标目录不能覆盖源目录或其祖先")
    if any(destination.is_relative_to(source / name) for name in TRACKED_DIRS):
        raise ValueError("目标目录不能放在被追踪的输入目录内")
    if not (source / "datasets/cn_daily").is_dir():
        raise ValueError("缺少 A股数据目录")
    destination.parent.mkdir(parents=True, exist_ok=True)
    # 临时目录与最终目录同文件系统，失败时不留下伪完整版本。
    with tempfile.TemporaryDirectory(prefix=".freeze-", dir=destination.parent) as temp:
        stage = Path(temp) / "snapshot"
        stage.mkdir()
        before = inventory(source)
        for name in before:
            target = stage / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source / name, target)
        copied = inventory(stage)
        if copied != before or inventory(source) != before:
            raise RuntimeError("冻结过程中输入发生变化，快照未发布")
        manifest = dict(format_version=1, snapshot_id=destination.name,
                        content_id=content_id(copied),
                        created_at=datetime.now(timezone.utc).isoformat(),
                        source=str(source), runtime=runtime_versions(), files=copied)
        (stage / "snapshot_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        # 不覆盖并发创建的非空版本目录。
        stage.rename(destination)
    return manifest


def verify_snapshot(root, check_runtime=True):
    """逐文件校验 SHA256；修改、遗漏、增添输入或运行依赖漂移均报错。"""
    root = Path(root).resolve()
    manifest = json.loads((root / "snapshot_manifest.json").read_text())
    if manifest.get("format_version") != 1:
        raise ValueError("不支持的快照清单版本")
    expected = manifest["files"]
    if content_id(expected) != manifest["content_id"]:
        raise ValueError("快照清单自身不一致")
    actual = inventory(root)
    changed = sorted(name for name in set(expected) | set(actual)
                     if expected.get(name) != actual.get(name))
    if changed:
        raise ValueError("冻结输入发生变化: " + ", ".join(changed))
    if check_runtime and manifest["runtime"] != runtime_versions():
        raise ValueError("运行依赖版本与冻结记录不同，须单独验证后创建新版本")
    return manifest

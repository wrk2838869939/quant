"""datahub 配置与路径发现。

优先级: 环境变量 DATAHUB_ROOT > 向上搜索 datasets/cn_daily > 默认当前代码所在的 datahub 目录
"""
import os
from pathlib import Path


def find_hub_root() -> Path:
    env = os.environ.get("DATAHUB_ROOT")
    if env:
        return Path(env).resolve()
    # 从 cwd 向上找
    for p in [Path.cwd(), *Path.cwd().parents]:
        if (p / "datasets/cn_daily").is_dir():
            return p
    default = Path(__file__).resolve().parents[2]
    return default


class HubConfig:
    def __init__(self, root: str | Path | None = None):
        self.root = Path(root) if root else find_hub_root()
        self.datasets = self.root / "datasets"

    @property
    def cn_daily(self) -> Path:
        return self.datasets / "cn_daily"

    @property
    def us_daily(self) -> Path:
        return self.datasets / "us_daily/us_daily.parquet"

    @property
    def cn_index(self) -> Path:
        return self.datasets / "index/cn_index_daily.parquet"

    @property
    def meta_dir(self) -> Path:
        return self.datasets / "meta"

    @property
    def calendar_dir(self) -> Path:
        return self.datasets / "calendar"

    @property
    def cache_dir(self) -> Path:
        d = self.datasets / "cache"
        d.mkdir(parents=True, exist_ok=True)
        return d

"""固定三阶段的对称 Ridge；所有组偏差等罚，不指定参考组。"""

from dataclasses import dataclass
import re

import numpy as np
import pandas as pd

from divide_selection import phase_features
from value_forecast import fit_value_model


DEFAULT_PHASES = ("recovery", "accelerating", "decelerating")


def _validate_design(frame, feature_cols, phases, slopes):
    """校验固定分组、原始特征及保留列名。"""
    if not isinstance(frame, pd.DataFrame) or not frame.columns.is_unique:
        raise ValueError("frame 必须是列名唯一的 DataFrame")
    if isinstance(feature_cols, str) or isinstance(phases, str):
        raise ValueError("feature_cols 和 phases 必须为名称序列")
    features, groups = tuple(feature_cols), tuple(phases)
    if any(not isinstance(c, str) for c in features) or len(set(features)) != len(features):
        raise ValueError("特征名必须为互不重复的字符串")
    if len(groups) != len(DEFAULT_PHASES) or set(groups) != set(DEFAULT_PHASES):
        raise ValueError("phases 必须恰好包含固定三个阶段，可调整顺序")
    if not isinstance(slopes, (bool, np.bool_)):
        raise ValueError("slopes 必须为布尔值")
    for feature in features:
        if (feature in {"intercept", "date", "ts_code", "phase"}
                or feature.startswith(("sym_phase_", "phase_"))
                or re.match(r"^(ret|mae|mfe|label|label_end|entry_missing|exit_missing|target)(_|$)", feature)):
            raise ValueError(f"日期、标签或保留列不能作为预测特征: {feature}")
    missing = set(features + ("mom60", "mom120")) - set(frame.columns)
    if missing:
        raise ValueError(f"缺少阶段或预测特征: {sorted(missing)}")
    return features, groups


def phase_design(frame, feature_cols, phases=DEFAULT_PHASES, slopes=False):
    """返回原始特征加三组中心化截距（可选中心化斜率）的设计。

    每组编码为 ``1[phase=g] - 1/3``，三列全部保留。对冗余列施加
    相同 L2 惩罚，使各组系数和自然为零；没有参考组的特殊待遇。
    已有 phase 只用于一致性检查，分组始终由历史 mom60/mom120 重算。
    缺失、未知或与历史收益冲突的阶段均拒绝，不静默回退。
    """
    features, groups = _validate_design(frame, feature_cols, phases, slopes)
    try:
        raw = frame.loc[:, ["mom60", "mom120"]].to_numpy(dtype=float)
        values = frame.loc[:, list(features)].to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("阶段和特征必须为有限数值") from exc
    if not np.isfinite(raw).all() or not np.isfinite(values).all():
        raise ValueError("阶段和特征必须为有限数值")
    # 空 features 避免旧模块生成其参考组交互项；只复用既有阶段定义。
    known = frame.loc[:, ["mom60", "mom120"]].astype(float)
    actual, _ = phase_features(known, [])
    if "phase" in frame:
        supplied = frame["phase"]
        if not supplied.isin(groups).all() or not supplied.eq(actual["phase"]).all():
            raise ValueError("已有阶段缺失、未知或与历史收益不一致")
    out = frame.copy()
    out["phase"] = actual["phase"].to_numpy()
    design = list(features)
    for group in groups:
        key = "sym_phase_" + group
        out[key] = out["phase"].eq(group).astype(float) - 1.0 / len(groups)
        design.append(key)
    if slopes:
        for group in groups:
            key = "sym_phase_" + group
            for feature in features:
                name = key + "__" + feature
                out[name] = out[key] * out[feature]
                design.append(name)
    return out, design


@dataclass
class SymmetricPhaseModel:
    """预测时重建同一设计，保留底层系数与训练审计信息。"""

    underlying_model: object
    feature_cols: tuple
    phases: tuple
    slopes: bool
    metadata: dict

    def predict(self, frame):
        """返回与 ValueForecastModel 相同的四列并保留输入行顺序。"""
        design, _ = phase_design(frame, self.feature_cols, self.phases, self.slopes)
        return self.underlying_model.predict(design)


def fit_symmetric(frame, feature_cols, *, phases=DEFAULT_PHASES, slopes=False,
                  asof, **forecast_opts):
    """用固定 alpha=1 的既有日期等权、成熟标签过滤模型拟合。

    只为标签已成熟、入场有效且目标非空的行构造阶段设计；未成熟行
    即使含无效特征也不能影响估计。全量行仍交由底层记录排除数量。
    """
    features, groups = _validate_design(frame, feature_cols, phases, slopes)
    options = dict(forecast_opts)
    alpha = options.pop("alpha", 1.0)
    if isinstance(alpha, (bool, np.bool_)) or not np.isscalar(alpha) or alpha != 1.0:
        raise ValueError("对称阶段实验固定 alpha=1.0")
    horizon = options.get("horizon", 60)
    label_cols = [f"label_end_{horizon}", f"entry_missing_{horizon}",
                  f"ret_{horizon}", f"mae_{horizon}"]
    missing = set(["date"] + label_cols) - set(frame.columns)
    if missing:
        raise ValueError(f"缺少训练列: {sorted(missing)}")
    cutoff = pd.to_datetime(asof, utc=True)
    if not isinstance(cutoff, pd.Timestamp) or pd.isna(cutoff):
        raise ValueError("asof 必须为有效的单个日期")
    cutoff = cutoff.normalize()
    ends = pd.to_datetime(frame[label_cols[0]], utc=True).dt.normalize()
    eligible = (ends.notna() & ends.le(cutoff) & frame[label_cols[1]].eq(False)
                & frame[label_cols[2:]].notna().all(axis=1))
    design, columns = phase_design(frame.loc[eligible], features, groups, slopes)
    training = frame.copy()
    for column in columns[len(features):]:
        values = np.full(len(training), np.nan)
        values[eligible.to_numpy()] = design[column].to_numpy()
        training[column] = values
    model = fit_value_model(training, columns, asof=asof, alpha=1.0, **options)
    metadata = dict(model.metadata)
    metadata.update({
        "model": "symmetric_phase_ridge", "base_feature_cols": list(features),
        "phases": list(groups), "slopes": bool(slopes), "reference_phase": None,
        "phase_encoding": "indicator_minus_one_over_number_of_phases",
        "group_penalty": "equal_l2_all_groups", "group_coefficient_constraint": "sum_zero",
    })
    return SymmetricPhaseModel(model, features, groups, bool(slopes), metadata)

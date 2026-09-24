> **历史过程记录**｜以下为当时的研究判断，含可能已被修正的结论。本次归档未重跑实验。
> 原始来源：`alpha_lab_growth/docs/SIGNAL_DIRECTION_CONTRACT.md`。保留实验设计、失败与修正过程；收益及指标不能脱离窗口和后续审查单独引用。
> 成长线旧结论已在 USX 重构中标记不可信；本页只作探索过程留档。

# 信号方向契约 (SIGNAL_DIRECTION_CONTRACT)

> 2026-09-18 制定。起因：P2'/F-US 阶段连续发生符号错误（账本方向反、paper 年化公式失真、快速验证脚本组合分选错端）。根因是方向在消费层由测量 IC 事后决定，每层重掷符号骰子。本契约为约束性标准，违反=结论无效。

## 1. 方向的唯一合法来源：定义时的经济机制先验

每个因子在**定义时**必须声明 `SignalSpec`：

```python
SignalSpec(
    name="low_turnover",
    economic_direction=-1,          # +1: 值大→预期收益高; -1: 值大→预期收益低
    mechanism="换手率=注意力/投机拥挤度; 量价文献: 高换手→未来收益差 (先验于任何IC测量)",
    contested=False,                # True = 机制先验存在对立证据, 见 §4
    compute=W -> DataFrame,         # raw 因子值
)
```

`economic_direction` 的唯一合法依据是**经济机制推理**（先于任何 IC 测量）：
- 例：换手率 → 投机拥挤 → 负向；流通市值 → 壳价值/小盘溢价（Liu-Stambaugh-Yuan 2019）→ 负向（值大=大市值=预期低收益）；大单净流入强度 → 知情资金累积 → 正向（**contested**：主线 T+2 口径测得无效，见 §4）
- 禁止用 TRAIN IC 的符号反推 economic_direction。

## 2. 签名信号只生产一次

`produce_signed(spec) = raw × economic_direction`。签名信号语义恒定：**值越大越应超配**。

- 下游（paper / 账本 / 验证 / 报告）一律 `nlargest(signed_signal)`，**禁止** `nsmallest`、禁止重新乘符号、禁止按 IC 翻转。
- parquet/manifest 必须记录 `economic_direction` 与 `mechanism`。

## 3. TRAIN IC 的角色降级：一致性校验

`validate_direction(spec, raw, lab_train, dates)`：
- `sign(measured IC) == economic_direction` → 通过，记录 IC 量级；
- 不符 → **raise DirectionMismatch**。处置是调查（机制推理错？定义 bug？数据口径？），**不是翻符号继续跑**。调查结论进台账后可修订 spec（修订=新版本，留痕）。

## 4. contested 先验的处置

机制先验存在对立证据时（如大单流入：开源月频 IC +0.042 vs 主线 T+2 无效）：
- `contested=True`，预注册时**显式声明双侧检验**（两侧都跑，报告两侧），或直接不进入打分合成；
- contested 因子不得在未解决前进入账本。

## 5. 时序变量（timing）特别条款

美股动量等时序触发变量同样需机制先验（风险溢出 → 正向）。测量与先验不符（如 VALID 月频反相关）→ 该触发变量标记 contested，生产中**默认关闭**，仅在预注册双侧描述报告中出现。

## 6. 单元测试义务

每个 spec 至少一个"已知正确答案"的合成单测（构造单调数据验证方向语义）；契约模块自身的 DirectionMismatch 路径必须有测试覆盖。

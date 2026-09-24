> **历史过程记录**｜以下为当时的研究判断，含可能已被修正的结论。本次归档未重跑实验。
> 原始来源：`alpha_lab/docs/MONTHLY_RISK_MANUAL_INTEGRATION_20260914.md`。保留实验设计、失败与修正过程；收益及指标不能脱离窗口和后续审查单独引用。

# 月度风险预算候选：人工执行接入与验证

## 结论与范围

月度50%风险预算候选已接入现有收盘计划、邮件草稿/发件箱、人工核对、实际成交和SQLite账户接续接口。新入口显式使用 `configs/monthly_risk_manual.yaml`；没有修改旧入口的默认配置，没有创建真实账户账本、发送邮件或提交委托。本次是执行接入验证，不增加收益或尾部防守有效性的证据。

规则保留：最多5股、100股买入申报单位、原月度复核、排名10名之外连续两次确认退出。先定原Top20，再屏蔽确认退出股票，不向后扩池，也不因股价高剔除股票。高风险总股票预算50%，风险解除恢复正常预算；恢复买入仅限仍持有股票在本轮**实际组合减仓成交**产生的额度，并满足趋势、现金、单股上限和总仓位预算。预算上限不等于必须投满。

## 接入实现

- `src/monthly_risk_live.py`：冻结市场模型、完整月度历史、进入/解除阈值、风险轮次与恢复额度；首次空仓接入回放历史风险状态，不补做历史交易、不生成历史减仓额度。配置变更、旧风险历史修订或漏过复核均显式停止，不静默切换。
- `src/shadow_inputs.py`：全A重建排名退出及入场屏蔽、当日个股恢复证据、沪深300风险概率。模型成熟边界、交易日连续性、近期证券主表覆盖和数据获取时刻均检查。
- `src/shadow_planner.py`：组合减仓、五席位已满时恢复旧仓、恢复与新买合并限额。正式退出优先于旧减仓意图；不预支计划卖出款。
- `src/shadow_account.py`：现金、股数、成交号和恢复额度在同一事务中提交。部分成交只按实际股数计量，重复回报不重复入账；未提交、撤单和邮件均不增加额度。清仓、新风险轮次及股数公司行动令相关旧额度失效。
- `scripts/manual_execution.py prepare --config ...` 与既有 `mail_outbox.py` 可消费新计划。邮件说明风险复核日、预算上限、减仓/恢复原因；发件结果不是成交回报。

## 已完成验证

`outputs/monthly_manual_parity/attempt04/verification.json`：4个2022起历史账户，月度日期偏移1/8/15/22；**540笔策略订单，包括45笔恢复计划**，股票、方向、数量、限价、预留现金、信号日期和原因与冻结研究账本一致。订单经济字段误差容限1e-7元。

这项对照使用已结算公司行动的研究快照，不冒充完整人工成交路径。人工入口会暂停次日已知公司行动股票，另有专项测试；2022-10-26的600900缺报价影响两个账户，入口拒绝生成计划，两处研究原本也无订单。停牌/缺报价期间目前仍需人工例外处理，不能把拒绝当作正常持仓建议。

新旧账户、邮件队列及输入边界测试见 `outputs/monthly_manual_current/validation/tests.log`。覆盖部分成交、重启、重复回报、事务回滚、五席位恢复、取消不消耗额度、公司行动、历史概率修订、未来价格变异、交易日缺口和无来源指数。

`outputs/monthly_manual_current/lifecycle/` 保存可查看的合成账户SQLite、28条审计事件、减仓/恢复计划、部分成交记录与历史邮件草稿。所有数据均标为演练，未写入真实账户。

## 2026-09-14输入与异常处理

已增量获取并交叉核验5550只股票的当日日线、daily_basic、复权因子和执行元数据；观察包为 `outputs/monthly_manual_current/observation_verified/`。当前信号日仍是9月1日，13只为**候选池**，持仓上限仍为5只。候选排名与封存版本一致，单股风险折扣最大误差约2.9e-15；四个日历的市场风险概率重建误差均低于5e-15。

指数区间接口混入9月12日休市记录，原响应已隔离在 `index/`，回执改为 `complete:false`。逐日查询该日指数和股票日线均为空；重新按交易日精确查询9/9、9/10、9/11、9/14，逐日行数、日期、日历和昨收连续性通过，结果在 `index_verified/`。未从异常响应中静默删行发布。此为同一供应商内部交叉核查，**不等于独立来源认证**。

新观察投影也显式补入2021-12-13缺失的4706行价格，缺少daily_basic的60行保持未知，不补零。旧主面板、旧研究账本和根AGENTS.md保持原样；历史指标没有因这次补投影而自动获得重新认证。

当前空仓8万元仅作软件演练：9月14日不是复核日，生成0笔订单，最近月度风险概率约11.23%，风险状态未触发，股票预算上限100%。这些字段不是今日买入建议。按现有日历下一复核为**2026-10-08收盘**；本次没有新增“月中立即建仓”规则。

## 人工运行顺序

在 `alpha_lab/` 下执行。真实账户须先有用户确认的收盘现金、持股和无未结委托记录；原9月14日盘中登记不能冒充收盘核对。邮箱也需指定接收人，现有登记仍待确认。无需券商自动下单或额外账号隐私。

```bash
# 1. 收盘核对；snapshot.json来自当日人工核对，不复制演练账户
python scripts/manual_execution.py snapshot --db ACCOUNT.sqlite --input snapshot.json

# 2. 21:00后，用该账户持仓构建的当日输入准备次日清单
python scripts/manual_execution.py prepare --db ACCOUNT.sqlite \
  --config configs/monthly_risk_manual.yaml \
  --bundle observation/bundle.json --calendar observation/calendar.json --output plan.json

# 3. 生成邮件草稿；发送仍使用已有受控发件箱
python scripts/manual_execution.py draft --plan plan.json \
  --metadata observation/metadata_report.json --output mail_draft

# 4. 次日执行窗口逐单核对，手动操作后导入真实回报
python scripts/manual_execution.py review --db ACCOUNT.sqlite --input review.json
python scripts/manual_execution.py fill --db ACCOUNT.sqlite --input fill.json
python scripts/manual_execution.py terminal --db ACCOUNT.sqlite --input terminal.json
python scripts/shadow_execution.py --db ACCOUNT.sqlite status
```

未提交用 `skip`，有真实成交后再撤销剩余数量用 `terminal`，不得用skip隐去部分成交。日终重新导入快照前，所有当日委托都须有已核对的终态。

数据跨日追加：`stage_execution_market.py --after 上次日期` 获取股票增量；`stage_monthly_risk_index.py --history 上次/index_close.parquet --asof 当日 --output 新目录` 获取指数。`build_monthly_manual_observation.py --previous-observation 上次观察目录 --stage 新目录 --index-stage 新指数目录 --snapshot 当日快照 --output 新观察目录` 延续特征历史。新stage须包含独立校验后的market_stage和最新metadata。候选公司行动范围不足时会输出 `candidate_scope_request.json`，用它重取元数据后重新构建，不把缺项默认为安全。年度单股模型缺失会停止，不自动沿用过期模型。

## 复现

```bash
PYTHONPATH=src:tests python -m unittest test_shadow_execution test_shadow_inputs \
  test_manual_execution test_manual_continuation test_mail_outbox \
  test_monthly_risk_manual test_monthly_risk_inputs.MonthlyRiskInputsTests
python scripts/verify_monthly_manual_parity.py --output outputs/NEW_PARITY
python scripts/demo_monthly_manual_lifecycle.py --output outputs/NEW_DEMO
```

本轮没有新增独立智能体审查；历史订单对照、输入重建与对抗测试由本次执行完成。盈利来源、模型后期漂移及实盘成交偏差仍沿用上一轮报告的限制，后续通过已约定的小资金人工试验记录检验。

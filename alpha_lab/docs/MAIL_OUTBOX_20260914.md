# 手动执行播报的持久化发件箱

已接入SMTP发送代码和SQLite发件箱，未实际连接SMTP、未发邮件、未启动定时发信进程。现有项目的邮件配置可解析，包含两个收件地址；地址选择仍待用户回复。凭据只在运行时读取，不复制到研究配置、日志或封存。

## 发送状态

每个计划与收件人生成独立邮件标识和固定正文。同内容重复入队复用原记录；同计划正文改变则停止，不静默替换已排队清单。

| 状态 | 含义与处理 |
|---|---|
| pending | 尚未尝试发送 |
| sending | 已登记发送尝试；进程中断后不会自动重发 |
| accepted | SMTP服务器已接受，不等于邮件已到达收件箱 |
| failed | 明确发送失败，允许重试 |
| unknown | 正文提交阶段断线等导致结果不确定，先核对再决定重试 |
| expired | 已到执行截止时间，拒绝发送 |

发送进程持有锁时不能并发重发或人工改判结果。服务器接受邮件后，即使退出连接失败也保留accepted。根据[Python SMTP接口语义](https://docs.python.org/3/library/smtplib.html)，服务器明确拒绝与提交期间连接断开采用不同处理；不把异常一律当作可以安全重试。

## 使用

在 `alpha_lab/` 运行。以下读取既有脚本的配置字面量，不导入或启动旧调度器；新邮箱可改用环境变量。

```bash
python scripts/mail_outbox.py --db outputs/my_shadow/mail.sqlite \
  --legacy-settings ../live_notify_scheduler.py \
  enqueue --plan local_account/plan.json --metadata local_account/metadata_report.json

python scripts/mail_outbox.py --db outputs/my_shadow/mail.sqlite status

# 仅检查队列，不连接SMTP：
python scripts/mail_outbox.py --db outputs/my_shadow/mail.sqlite dispatch --id 邮件ID
```

确定具体收件范围和当期正文后，实际发送需给dispatch显式增加 `--send` 并提供相同的配置来源。CLI默认不发送。队列中的地址固定，不能通过更换运行配置把原邮件转发到其他地址。

环境变量配置包括 `ALPHA_SMTP_HOST/PORT/SENDER/USER/PASSWORD/TLS` 和 `ALPHA_MAIL_TO`；TLS只支持SSL或STARTTLS，验证服务器证书。不要在版本控制或聊天中保存邮箱授权码。

结果不确定时，先核对邮箱或服务器记录；发送进程已停止后可使用 `resolve --id ... --outcome accepted|retry --evidence ...` 记录处理依据。retry仅允许再次尝试，仍会检查邮件是否过期。

## 验证和边界

11项新发件箱测试加此前63项，共74项通过。测试使用注入传输与SMTP模拟对象，没有真实网络发送；覆盖去重、重启、发送中断、过期、收件人变化、并发冲突和凭据解析不执行旧脚本。

生成或发送邮件不改变账户持仓，不构成成交记录。历史演练清单不能进入实际发件队列。当前仍需有效的当期行情、真实账户起点和收件范围，之后才有可发送的真实准备清单。定时运行和送达验证尚未完成。

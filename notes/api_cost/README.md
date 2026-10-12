# Tinker API 省钱指南

起因：2026-09-13 到 2026-10-12 估算花费约 $1,600，其中约 $750 是 DSH-Inkling 评测
（10/8–10/11 用 Inkling-Small 走 OpenAI 兼容端点），$255 是 10/3 一次 256K 上下文的训练。
下面是从那次分析里总结出的做法，按省钱效果排序。

## 先看账

```bash
cd tinker-cookbook
set -a && source .env && set +a
.venv/bin/python notes/api_cost/billing_report.py --days 7
```

`billing_report.py` 调 `get_billing_usage`（单次最多 14 天，脚本会自动分段），
再乘以官方价格表（`models.json` / `serverless.json`），输出按用量类型、模型、天、
`recipe_name`、session 的拆分，以及“改走 serverless 能省多少”和当前存储量。

注意：

- API 不返回美元金额，这里是 **按当前价格** 估算；历史上的折扣变动不会反映。准确金额看 Tinker console。
- 数据滞后几个小时。
- 用 `--cache usage.json` 存一份原始数据，反复分析时不用重新拉。

## 钱花在哪：先判断大头

| 现象 | 说明 | 对应做法 |
|---|---|---|
| `prefill_cached` 占比最高、输入/输出比 ≫ 10 | 多轮 agent 每轮都重发整段上下文 | 第 1、2 条 |
| `train` 突增 | 用了 `:peft:262144` 训练，或 batch/epoch 过多 | 第 3 条 |
| `storage` 每天固定一笔 | checkpoint 没过期 | 第 4 条 |
| 大量 `(no metadata)` | 无法按实验拆账 | 第 5 条 |

上一次的画像：输入/输出 ≈ 58:1，缓存命中 95%，cached prefill 占总花费 52%。

## 1. Inkling 系列的纯采样/评测改走 serverless 端点

只做推理（评测、数据生成、agent 轨迹）时，用
`thinkingmachines/Inkling-Small:peft:262144:sampling-nvfp4`（或 `Inkling:...:sampling-nvfp4`）。

| Inkling-Small，$/M token | 训练端点采样 | serverless nvfp4 |
|---|---|---|
| prefill | 0.58 | 0.30 |
| cached prefill | 0.116 | 0.06 |
| sample | 1.44 | 1.20 |

上一次的 Inkling-Small 采样 $1,149 → 约 $645。

前提和风险：

- 仍是 beta，不建议做大规模生产流量。
- nvfp4 量化，和 bf16 结果不保证一致。**同一组对比内部不要混用端点**；切换前先在小样本上确认分数没有漂移。
- 需要采样 LoRA 权重（RL rollout、评测自己训的 checkpoint）时不适用。

## 2. 控制 agent 上下文和重复次数

cached prefill 虽然打了 2 折，但长上下文 × 多轮 × 多次重复依然是最大头。

- 设 `max_turns` / 总 token 预算上限，失败轨迹早停，不要跑满。
- 裁剪工具输出（编译日志、测试输出只保留尾部和错误行），历史超过阈值就截断或摘要。
- **先算清楚需要多少次重复。** 做 A/B 时在同一批题上配对比较，比各自独立重复全量 100 题省得多
  （见 `notes/dsh_inkling/RESULTS.md` 里关于 run-to-run variance 的讨论）。
  先在分层抽样的小子集（如 30 题）上看效应，有信号再上全量。
- 并发只影响速度不影响单价，但并发高时失控的成本涨得更快——长跑的批量任务要盯每小时用量。
- 不同模型对比（GLM-5.3、Inkling、Qwen 等）先用小子集筛，GLM-5.3 单价是 Inkling-Small 的 6–8 倍。

## 3. 训练：能用 64K 就不用 256K

`:peft:262144` 只有 **训练** 价格更高，prefill/sample 一样：

| $/M train token | 64K | 256K（`:peft:262144`） |
|---|---|---|
| Inkling-Small | 1.73 | 3.47 |
| Inkling | 5.61 | 11.23 |
| Qwen3.8-27B | 4.10 | 7.46 |

- 先统计训练数据的序列长度分布；绝大多数在 64K 内就用不带后缀的基座，超长样本截断或丢弃。
- 注意：训练客户端调用的纯 forward 也按训练价计费，评估 loss 时别频繁全量 forward。
- 正式跑之前先用小 `n_batches` 跑通，确认 loss 曲线和 checkpoint 正常再放大。

## 4. checkpoint 存储

$0.10 / GB / 月。上一次存了约 997 GB，≈ $100/月。

- cookbook 的 RL 训练默认 `ttl_seconds=604800`（7 天），rolling checkpoint 2 小时；
  自己写的脚本或把 `ttl_seconds` 设成 `None` 的会永久保留。
- 定期清理：

```bash
.venv/bin/tinker checkpoint list
.venv/bin/tinker checkpoint delete --run-id <run-id> --before 2026-10-01
.venv/bin/tinker checkpoint set-ttl tinker://<run-id>/weights/<name> --ttl 604800
```

- 要长期保留的权重：`tinker checkpoint download` 或 `push-hf` 存到别处后再删。

## 5. 给每个实验打标签

没有 metadata 的花费只能靠时间线去猜是哪个实验。

- SDK 路径：创建 `ServiceClient` 时传
  `user_metadata=recipe_user_metadata("<实验名>")`（`tinker_cookbook/utils/git_rev.py`）。
  cookbook 的 `rl/train.py`、`supervised/train.py` 已经通过 `recipe_name` 自动带上。
- OpenAI 兼容端点（`.../oai/api/v1`）的请求没有地方带 `user_metadata`，同一个 key 的流量会归到一个
  `oai-*` session 下。这类实验在启动和结束时把时间点记到实验笔记里，事后用 `billing_report.py --start/--end`
  按时间窗对账。

## 启动大任务前的检查清单

- [ ] 估算一下：题数 × 重复次数 × 平均每条轨迹 token（先跑 2–3 条量出来）× 单价。超过 $100 先想想能不能用子集。
- [ ] 纯推理且不需要 LoRA 权重 → serverless nvfp4 端点（对比组内保持一致）。
- [ ] 设了 `max_turns` / token 上限 / 早停。
- [ ] 训练序列长度 ≤ 64K → 不用 `:peft:262144`。
- [ ] checkpoint 有 TTL。
- [ ] 带了 `recipe_name` metadata，或记下了起止时间。
- [ ] 跑起来后第二天用 `billing_report.py --days 1` 看一眼实际花费是否和估算一致。

## 价格来源

[Models & Pricing](https://tinker-docs.thinkingmachines.ai/tinker/models/)（2026-10-11 查阅，
Inkling 系列当时有限时 5 折）。价格会变，以页面和 `models.json` 为准。

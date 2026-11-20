# P0 + P1 汇总：八个探针暴露的五个根因，以及应该怎么改

被测版本：`try_1/python_project/superagent_project`（`git 3a817ef` 之后的本地状态，全程未修改源码）
探针代码与记录：`Final Project/test_tasks/`
评分权重（作业要求）：Released 20% / **Unseen 10%** / **Design 30%** / Execution 10% / Presentation 30%

---

## 一、一页结论

八个盲测全部跑完，结论可以压缩成三句话：

1. **这套系统的"自主性"目前只覆盖了"在固定表征上换分类器"这一维。**
   三个数值型任务上，Planner 的诊断全部正确（过拟合、类别不平衡、内存不足），
   但动作空间里没有"换表征 / 降维 / 改编码"这类动作，所以诊断无法转化为结果。
2. **真正会导致 0 分的不是精度，是三个工程缺口**：失败配额按错误类型全局计数、
   Planner 输出被 800 token 截断、全失败时抛未捕获异常。P1 的三个运行全部因此失败，
   其中一个（P1-C）**一分钱没花、一次 LLM 都没调**就崩了。
3. **accuracy 单指标会把退化解包装成成功**：P1-A 交出的模型 accuracy 0.9924，
   但欺诈召回率 0.081，比"全预测多数类"只高 0.0007。

按性价比排序的修改顺序（详见第四节）：

| 优先级 | 改动 | 代码量 | 直接收益 |
|---|---|---|---|
| **P0-1** | 失败配额按 (实验身份, 错误类型) 计数 | ~15 行 | 消除 P1-B/P1-C 这类"两次同类错误即全灭" |
| **P0-2** | `planner_max_tokens` 800→2048，空响应重试 + `reasoning_content` 兜底 | ~30 行 | 消除 P1-A 这类"钱没花完却失败" |
| **P0-3** | 全实验失败时优雅退出，写 summary + 保留运行目录 | ~25 行 | 把"崩溃"换成"有效的部分结果"（评分明确认可） |
| **P0-4** | 放开 metric 枚举 + 选模用 metric + 退化保护 | ~80 行 | 直接兑现 10% 的 unseen 风险，且修 P1-A 类假成功 |
| **P1-1** | 树模型改回稀疏（`sparse_output=True`） | 2 行 | 消除 379 GiB 崩溃（精度仍由 P1-2 负责） |
| **P1-2** | 预处理进入 plan schema（表征积木） | ~200 行 | 唯一能同时提升 Design 30% 和实际精度的结构性改动 |
| **P1-3** | CSV 支持显式列声明（单文本 / 表格） | ~60 行 | 消除"具体任务形态一进来就死" |
| **P1-4** | 图像计划加入时间预算估算 | ~80 行 | 让默认 224/12ep 不再必然超时 |

---

## 二、八个探针的结果总表

| # | 探针 | 数据 | 结局 | 关键数字 | 花费 |
|---|---|---|---|---|---|
| P0-D | 指标非 accuracy | Bank Marketing | ❌ 启动即失败 | `only supports accuracy`，0 次实验 | ¥0 |
| P0-B | 表格型 CSV（30 列） | Breast Cancer | ❌ 数据探查失败 | `exactly two feature columns` | ¥0 |
| P0-C | 单列文本 CSV | Task 2 派生 | ❌ 数据探查失败 | 同一行代码、同一报错 | ¥0 |
| P0-A | 纯数值 npy | Wine 178×13 | ⚠️ 跑完，精度塌陷 | 测试 **0.7037** vs 缩放 LR 1.0000；999 列 / 124 行 | ¥0.003 |
| P1-A | 极端不平衡 | Credit Card 60k | ⚠️ 提前终止 | 测试 acc **0.9924**，正类 recall **0.081**，多数类 0.9918 | ¥0.010 |
| P1-B | 纯数值多类 | Covertype 40k×54 | ⚠️ 提前终止 | 测试 **0.6688** vs 缩放 LR 0.7325 vs RF 0.8478 | ¥0.009 |
| P1-C | 真实图像（灰度） | Fashion-MNIST 5k | ❌ 崩溃（exit 1） | 两基线均超时；循环 0 次迭代；**0 次 LLM 调用** | ¥0 |
| P1-D | 彩色图像 | CIFAR-10 | 未执行 | 源站带宽 49–90 KB/s；由微基准替代 | — |

累计花费 **¥0.0225**（单任务预算 ¥20 的 0.11%）。
**限制能力的从来不是钱，是决策空间和容错设计。**

四个"有效"结果的对照差距：

| 任务 | agent | 笨基线 | 常识基线 | 差距 |
|---|---:|---:|---:|---:|
| Wine | 0.7037 | 多数类 0.4074 | 缩放 LR 1.0000 | **−29.6 pt** |
| Covertype | 0.6688 | 多数类 0.4877 | RF(200) 0.8478 | **−17.9 pt** |
| Credit Card（acc） | 0.9924 | 多数类 0.9918 | 平衡 LR 0.9758 | +0.07 pt |
| Credit Card（macro-F1） | 0.5731 | 多数类 0.4979 | 平衡 LR 0.6865 | **−11.3 pt** |

最后一行是全文最重要的一行：**agent 选中的模型 accuracy 更高、macro-F1 更低**。
这正是"只会优化 accuracy"的必然结果。

---

## 三、五个根因（每条都有实测证据）

### 根因 1｜表征层是死的，搜索空间因此被压扁

- 9 个 adapter 共用 `normalize_categorical_array()` + `OneHotEncoder`：
  每个单元格 `str()` 后当作类别。
- 后果一：Wine 的 13 个浮点列 → **999 列 / 124 行训练样本**，三个实验全部 `possible_overfitting`。
- 后果二：Credit Card 的 30 个浮点列 → **1,212,208 列**，模型文件 40.4 MB。
- 后果三：Covertype 的 54 个整型列 → 12,397 列，类 4 的 recall 只有 **0.041**。
- `data_profile` 里其实已经算了 `unique_counts`，**完全有能力判断哪些列是数值型**，但没人用。

证据：P0-A 三个实验的特征数都是 999；P1-A exp_001 的特征数是 1,212,208。

### 根因 2｜树模型把稀疏矩阵稠密化，这是硬 bug

```python
# adapters/tabular_extra_trees.py:52  和  adapters/tabular_random_forest.py:35
OneHotEncoder(handle_unknown="ignore", sparse_output=False)
```

P1-A 的 exp_002 因此直接请求 **379 GiB**：

```
MemoryError: Unable to allocate 379. GiB for an array with shape (42000, 1212208)
```

同一份数据用 CSR 只需 17.4 MB（每行 30 个非零）。
Task 1 没暴露这个问题，只是因为那份数据的类别基数低——
**"在已发布任务上能跑"掩盖了"在任何高基数数据上必崩"**。

补充一个关键对照（`bench_sparse_trees.py`，同一份 P1-B 数据）：

| 变体 | 编码 | 拟合耗时 | 验证 acc |
|---|---|---:|---:|
| RandomForest(100) | One-Hot 稀疏 | 37.4s | 0.6703 |
| RandomForest(200) | **原始数值** | **3.4s** | **0.8487** |

**所以这是两个独立缺陷**：稠密化导致崩溃（改一行即可），
One-Hot 导致信息损失（要改表征层）。只修前者，会得到一个"能跑但精度 0.67"的树模型，
反而更容易被误读成"树模型不适合这个任务"。

### 根因 3｜故障处理把"可恢复"当成"致命"

三处叠加，任意一处都足以让 24 小时窗口归零：

| 机制 | 实现 | 后果 |
|---|---|---|
| `maximum_same_experiment_errors` | 计数器 key = `error_type`（全局） | P1-B 两次内存超限、P1-C 两次超时 → 自动循环**零次迭代**即返回 |
| `maximum_planner_errors: 2` | 空响应也计入 | P1-A 两次空响应 → 整轮结束，预算只用了 0.05% |
| `select_best_model` | 全失败时 `raise RuntimeError` | P1-C 退出码 1，无 `summary.json`、无锁、无检查点 |

而 `planner_max_tokens: 800` 是空响应的直接原因：两次失败调用的 `output_tokens`
**都精确等于 800**，即模型把预算全花在推理上，`content` 为空。

评分规则明确写"including valid partial results from incomplete tasks"——
**崩溃拿不到这部分分，跑完但分数低可以拿到。**

### 根因 4｜数据契约只兜住一种形态

`data_inspector.py:109` 按**列数**而不是内容路由 CSV：`≠2` 抛错，`=2` 直接写死 `text_pair`。
于是：表格 CSV 进不来、单列文本进不来、而"1 个特征列的表格"会被静默当成文本对。
README 写的是"通过数据内容判断模态"，实现是"按形状特判"——和"按任务名特判"是同一类问题。

### 根因 5｜图像链路没有时间预算，且从未在真实数据上验证

- 默认计划硬编码 `small_cnn@128/15ep` 与 `resnet18@224/12ep`。
- CPU 微基准（2 线程，batch 32）：`small_cnn@128` = 0.643 s/批 → 5000 张 15 epoch **1504s**；
  `resnet18@224` = 3.628 s/批 → **6792s**，是 1800s 上限的 3.8 倍。
- 要在 1800s 内跑完 resnet18@224，训练集不能超过约 1,300 张。
- `plan_validator` 只校验 `image_size ∈ [32,512]`、`epochs ∈ [1,100]`，
  **不知道也不告诉 Planner"224×224×12epoch 意味着 6792 秒"**。
- 灰度识别是好的（`mode_counts: {L: 5000}`、`grayscale_samples: 5000`），
  但没有任何"28px 被放大到 128px 是 20 倍算力浪费"的提示。

---

## 四、怎么改（按优先级，含代码位置与草案）

### P0-1　失败配额按"实验身份 + 错误类型"计数

**位置**：`superagent/controller.py:540-546`

```python
# 现在：同类错误出现两次就终止整轮
for previous_result in results:
    if previous_result.status == "failed":
        error_key = previous_result.error_type or "unknown"

# 改为：同一 adapter 的同一类错误，重复 N 次才放弃
for previous_result in results:
    if previous_result.status == "failed":
        signature = getattr(previous_result, "experiment_signature", None) \
                    or (previous_result.model_name, previous_result.error_type)
        experiment_error_counts[signature] = experiment_error_counts.get(signature, 0) + 1
```

配套两条：

1. **保底尝试**：若 `successful == 0`，强制允许至少 `max(3, max_experiments // 2)` 次规划，
   不受其他停止条件影响。否则一个"两个默认基线都失败"的任务永远没有第二次机会。
2. **区分错误类型**：`MemoryError`（代码必须改：稠密化）与 `ResourceLimitExceeded`
   （参数可以改）应映射到不同的 `error_type`，并在写给 Planner 的反馈里明确"这一类只能靠改参数解决"。

**验证**：重跑 P1-C，断言循环至少迭代 2 次、有 `plans.jsonl` 的 `plan_source: llm` 记录。
**预期收益**：unseen 10% 里"能否完成"这一档。

---

### P0-2　Planner 输出预算与空响应恢复

**位置**：`configs/llm.json`、`superagent/llm/client.py:159`

```json
{ "planner_max_tokens": 2048 }
```

```python
# client.py：空 content 时先看 finish_reason 与 reasoning_content
content = result["choices"][0]["message"].get("content") or ""
finish_reason = result["choices"][0].get("finish_reason")
if not content.strip():
    if finish_reason == "length":
        # 预算耗尽：翻倍重试一次，仍失败才计入错误
        return self.chat(..., max_tokens=min(max_tokens * 2, 8192), _retry_hint=True)
    content = result["choices"][0]["message"].get("reasoning_content") or ""
```

再加一条：**空响应 / JSON 解析失败各允许 3 次内部重试（指数退避），只有连续 3 次都失败才记 1 次
planner error**；同时把 `finish_reason` 写进 `llm_calls.jsonl`（现在这个字段是空的，
`input_tokens` 也是 `None`，与实际账本不一致）。

**代价**：每次调用输出上限从 800 提到 2048，最坏情况 ¥0.0048/次。
按 P0+P1 实测的调用量（4 次/8 任务），全部任务额外成本 < ¥0.1。

**验证**：把 `planner_max_tokens` 临时改回 800，断言"空响应不再直接杀死运行"。

---

### P0-3　全实验失败时优雅退出

**位置**：`superagent/controller.py:359`

```python
try:
    best = select_best_model(results)
except RuntimeError as error:
    summary = self._write_failure_summary(
        results=results, stop_reason="all_experiments_failed",
        detail=str(error), logger=logger, ...
    )
    return summary          # 退出码 0，运行目录完整，可用于"部分结果"评分
```

同时保证：`run_state.json` 的 `status` 写成 `failed`（而不是缺失），
`summary.json` 里带上每个失败实验的 `error_type` 与 `error_message`，
`final/` 目录留空但存在。

**为什么重要**：作业明确"incomplete tasks 的**有效部分结果**也计分"。
现在 P1-C 的输出是"一个 Python traceback + 半个目录"，无法被当作任何结果使用。

---

### P0-4　指标门与退化保护

三处改动，都是小改：

```python
# 1) task_config.py:206
SUPPORTED_METRICS = {"accuracy", "macro_f1", "balanced_accuracy"}
if metric not in SUPPORTED_METRICS:
    raise ValueError(f"metric must be one of {sorted(SUPPORTED_METRICS)}")

# 2) model_selector.py:19 —— 按 config.metric 排序，而不是写死 accuracy
key = {"accuracy": lambda r: r.validation_accuracy,
       "macro_f1": lambda r: r.f1_macro,
       "balanced_accuracy": lambda r: r.balanced_accuracy}[metric]

# 3) 新增退化检测（放在选模之后、写 solution_lock 之前）
def is_degenerate(metrics, profile):
    recalls = list(metrics["class_recall"].values())
    predicted_rate = metrics["positive_prediction_rate"]   # 新增
    return (min(recalls) < 0.10 and max(recalls) - min(recalls) > 0.70) \
           or (predicted_rate < 0.001 and profile["minority_rate"] > 0.01)
```

命中时**不要静默接受**：在 `summary.json` 写 `warnings: ["degenerate_model"]`，
在 `decisions.jsonl` 记一条，并在报告里打印"该模型 accuracy 高于多数类基线 0.07pt，
但少数类 recall 0.081，不建议作为最终结论"。

**即使不改 metric 枚举，第 3 条也应当立刻加**——它把 P1-A 那种"看起来 0.9924 的失败"
变成"显式的失败"，这是报告诚实度和答辩可信度的分水岭。

---

### P1-1　树模型回到稀疏（2 行）

```python
# tabular_extra_trees.py:52, tabular_random_forest.py:35
OneHotEncoder(handle_unknown="ignore", sparse_output=True)
```

`ExtraTreesClassifier` 与 `RandomForestClassifier` 都支持 CSR 输入，改完即可。
配套：当 `feature_count > 20000` 时，在 plan 生成阶段强制 `max_features="sqrt"` 且
`n_estimators ≤ 200`，并在 `data_profile` 诊断里加 `high_dimensional_representation`
的可执行建议（现在这个诊断只生成标签，不产生动作）。

**注意：这一步只解决"能跑"，不解决"跑得好"。** 实测稀疏 One-Hot 下 RF = 0.6703，
同一个 RF 用原始数值是 0.8487。所以 P1-1 必须和 P1-2 一起做，
否则会得到一个"不再崩溃、但精度和 LR 一样"的树模型，掩盖真正的问题。

**验证**：P1-A 重跑，exp_002 应在 2 分钟内完成而不是 OOM（实测 37–55s，峰值 < 810 MB）。
精度提升留给 P1-2 验证。

---

### P1-2　把预处理变成计划的一部分（结构性改动，收益最大）

这是唯一能让 30% Design 分从"9 条固定管线上的超参搜索"变成"真正的算法设计"的改动。

**1) 扩展 plan schema**（`superagent/experiment_plan.py` + `plan_validator.py`）

```python
"pipeline": {
    "numeric":     {"enum": ["standard", "minmax", "none", "quantile_bin8", "quantile_bin32"]},
    "categorical": {"enum": ["onehot", "ordinal", "target_encode"]},
    "select":      {"enum": ["none", "variance", "k_best_2000", "pca_50"]},
    "imbalance":   {"enum": ["none", "class_weight_balanced", "smote"]},
}
```

Validator 只需要加一层"积木组合合法性"（例如 `pca_50` 只能跟在 `standard` 之后，
`target_encode` 需要 `min_samples_per_category ≥ 50`），仍然是白名单，不引入代码生成。

**2) 让 Inspector 产出类型，而不是只产出统计量**（`data_inspector.py` 已有 `unique_counts`）

```python
"inferred_types": {"numeric": [0, 1, 2, ...], "categorical": [...], "high_cardinality": [...]}
```

判定规则（可解释、可审计）：dtype 数值 且 `unique_count ≥ 20` 且 `unique/n ≥ 0.02` → numeric。

**3) Adapter 改成 `build_preprocessor(plan.pipeline, profile)`**

```python
ColumnTransformer([
    ("num", {"standard": StandardScaler(), "quantile_bin8": KBinsDiscretizer(8), ...}[p.numeric], num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=True), cat_cols),
])  # 树模型与线模型共用，sparse 输出
```

**预期效果**（基于 `prepare_p*.py` 里已测的参考值）：

| 任务 | 现在 | 只修稀疏（P1-1） | 加表征层后（估计） |
|---|---:|---:|---:|
| Wine | 0.7037 | 0.7037 | **0.95–1.00** |
| Covertype | 0.6688 | ~0.67（树模型能跑但精度不变） | **0.73（LR）/ 0.83–0.85（RF）** |
| Credit Card | macro-F1 0.5731 | ~0.57 | **0.68–0.70**（balanced + 阈值） |
| 计划空间大小 | 9 种 | 9 种 | **数百种**（9 adapter × 5×3×4×3 pipeline） |

**这是把"诊断 → 动作"接通的那一步。** 没有它，Planner 关于过拟合/不平衡的诊断
永远只能写进 `reason` 字段，不能改变任何结果。

---

### P1-3　CSV 路由按显式声明

```json
"data": {"format": "csv_split", "columns": {
    "text": ["headline"], "numeric": ["price"], "categorical": ["site"], "target": "label"}}
```

有声明就按声明；无声明时才退化到启发式（每列平均 token 数、基数、数值解析成功率），
并且**把判定结果写进 `data_profile.json` 和日志**，让误判可被发现。
`data_inspector.py:109` 那条"必须恰好两列"的断言删掉。

**验证**：P0-B（表格 CSV）、P0-C（单列文本）应立即通过探查。

---

### P1-4　图像计划的时间预算

在 `plan_validator` 里加一个纯估算器（不需要跑）：

```python
SECONDS_PER_IMAGE = {  # 来自 bench_image_cpu.py，2 线程 CPU
    ("image_small_cnn", 32): 0.0016, ("image_small_cnn", 128): 0.0201,
    ("image_resnet18", 128): 0.0411, ("image_resnet18", 224): 0.1134,
}
estimated = train_samples * SECONDS_PER_IMAGE[(adapter, image_size)] * epochs
if estimated > 0.8 * experiment_timeout_seconds:
    reject(f"estimated {estimated:.0f}s exceeds 80% of the {timeout}s budget; "
           f"try image_size={suggested_size}, epochs={suggested_epochs}")
```

同时把默认值改成**原生尺寸优先**：`image_size = min(median_width, median_height)` 向上取到 32 的倍数
（Fashion 28×28 → 32，而不是 128）。仅这一条就能让 P1-C 的 1504s 降到 **121s**，
并且精度更高（无意义的 4 倍上采样会害了 28px 数据）。

**验证**：P1-C 重跑，断言 `small_cnn` 在 <300s 内完成、`resnet18@224` 的 plan 被拒并给出替代参数。

---

### P2　证据质量的三处小修（答辩前必做）

1. `training_time_seconds=0.0` 在所有 sklearn adapter 里是**硬编码**（`tabular_baseline.py:120` 等）。
   改成 `time.perf_counter()` 实测，否则报告里任何"训练耗时"都是假的。
2. `llm_calls.jsonl` 的 `input_tokens/output_tokens/finish_reason` 是 `None`，
   而 `budget_ledger.jsonl` 有值 —— 两份审计日志不一致，抽查时会被问。
3. `maximum_same_experiment_errors` 的语义要写进 README（改前/改后各是什么），
   否则修改记录本身不可审计。

---

### 关于停止策略（可选，改动小）

`no_improvement_patience: 2` 在小数据多类任务上过激（P0-A 在第 3 个实验就停了）。
建议：

- `patience ≥ 3`；
- **失败实验不计入** `no_improvement`（现在只统计 success，这点是对的，保持）；
- 增加 `min_experiments_before_stop: 3`，避免"两个基线一差一崩就收工"。

---

## 五、改完之后怎么证明改对了（回归表）

建议把下面这张表直接放进报告，用同一套 `test_tasks/` 重跑，一格一格对：

| 探针 | 触发点 | 改前 | 改后的通过判据 |
|---|---|---|---|
| P0-D | `metric: macro_f1` | 启动失败 | 正常完成，`summary.metric == "macro_f1"` |
| P0-B | 表格 CSV | 探查失败 | 正常完成 |
| P0-C | 单列文本 | 探查失败 | 正常完成 |
| P0-A | Wine | 0.7037 | **≥ 0.95**，特征数 ≤ 50 |
| P1-A | Credit Card | acc 0.9924 / 少数类 recall 0.081 | **macro-F1 ≥ 0.65 且 recall ≥ 0.6**，或显式标记 `degenerate_model` |
| P1-B | Covertype | 0.6688，树模型 OOM | **≥ 0.80**，且至少一个树模型实验成功（只修 P1-1 只能到 ~0.67） |
| P1-C | Fashion | 崩溃 | **完成**，`small_cnn` 在 300s 内，val ≥ 0.85 |
| 全部 | 失败处理 | 3/4 运行中途死 | **0 次 `maximum_same_experiment_errors_reached`**（除非真的重复同一配置 3 次） |
| 全部 | 审计 | `training_time_seconds=0.0` | 非零，且与实测耗时吻合 |

建议再加一个"缺陷注入"回归（成本很低）：
把 `planner_max_tokens` 临时设为 100 制造空响应，断言运行仍然完成；
把 `experiment_timeout_seconds` 设为 30 制造超时，断言循环仍然继续规划。

---

## 六、这些结果在报告/答辩里怎么用

1. **把"缺陷"写成"边界"**：报告里主动列"已知不支持：单列文本、表格 CSV、非 accuracy 指标"，
   比被抽查发现要好。P0-B/C/D 的失败日志是最诚实的证据。
2. **用对照差值代替绝对分数**：`0.6688` 本身没有信息量，
   "比 3 行 `StandardScaler+LR` 低 6.4 个点、比 RF 低 17.9 个点"才有。
   P1-A 的"accuracy 更高但 macro-F1 更低"是最有说服力的一张幻灯片。
3. **诚实交代 cost/benefit**：八个盲测总共花了 ¥0.0225，
   说明预算不是瓶颈，**决策空间才是**。这句话可以直接回答"为什么不做更多实验"。
4. **如果来不及改代码**：至少做 P0-1/P0-2/P0-3（合计 < 80 行），
   把"工程性归零"的概率压下去；P1-2 作为"下一步工作"写在 limitations 里，
   并用 P0-A / P1-B 两张对照表说明"为什么需要它"。

---

## 七、附：本次新增文件

```
Final Project/test_tasks/
  P0_RESULTS.md                       P0 四个探针的完整记录
  P1_RESULTS.md                       P1 三个探针的完整记录 + 微基准
  P0_P1_SUMMARY_AND_FIX_PLAN.md        本文档
  prepare_p0.py / prepare_p1.py        数据与配置生成（无特征选择、无调参）
  bench_image_cpu.py                   CPU 单步耗时微基准
  bench_sparse_trees.py                稠密化 vs 表征损失 的贡献分离实验
  p0*/ p1*/                            各任务的 config.json / split_manifest.json / data
  results/logs/*.log                   每次运行的完整 stdout
  results/runs/*/                      experiments.jsonl / plans.jsonl / errors.jsonl /
                                       budget_ledger.jsonl / final/test_metrics.json
  results/bench_image_cpu.json         五种 模型×尺寸 的实测耗时
  results/bench_sparse_trees.json      稀疏/原始数值 × ET/RF 的耗时与精度
```

`python_project` 下未修改任何源码或配置；`runs/` 之外没有新增文件。

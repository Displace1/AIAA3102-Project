# SuperAgent 更新说明（2026-11-23）

面向对象：项目组其他成员
改动范围：`superagent_project/` 下 45 个文件（+4476 / −238），含 3 个新模块、7 个新测试文件
同步位置：GitHub `Displace1/AIAA3102-Project` 的 `main` 分支

---

## 0. 三句话版本

1. **图像实验不再"拍脑袋估时间"**：估算器从「一张参考表 + 面积平方外推」换成「加性代价模型 + 机器指纹」，
   再加一层**闭环止损**（跑不完就在 epoch 边界停，交出可用模型）和一层**实测缓存**（换机器跑一次就自举）。
2. **表格类型推断多了"整数序数"通道**：低基数整数列不再一律 one-hot。Covertype 上 ExtraTrees **+2.37 点**，
   特征维数 97 → 54，训练时间 **62.1s → 8.3s**。
3. **指标 3 → 7**（新增 `precision` / `recall` / `roc_auc` / `log_loss`，其中 `log_loss` 是唯一"越小越好"的，
   需要全链路方向感知），**字符串标签**可直接使用；3 个长期失败的测试（配置里残留旧绝对路径）一并修掉。
   **测试从 174 例 3 失败 → 196 例全绿。**

---

## 1. 改动一览

| # | 主题 | 一句话 | 主要文件 | 新增测试 |
|---|---|---|---|---|
| 1 | 图像运行时长自适应 | 估算器重标定 + 机器指纹 + 闭环止损 + 实测缓存 | `image_runtime.py`、`runtime_environment.py`(新)、`throughput_cache.py`(新)、`executor.py`、`image_training.py`、`plan_validator.py` | 14 + 14 + 16 例 |
| 2 | 整数序数通道 | 低基数整数列按"单调性证据"决定编码，而非一律 one-hot | `tabular_data.py`、`data_inspector.py`、`plan_validator.py`、`experiment_registry.py`、三个 tabular adapter | 25 例 |
| 3 | 指标枚举 3 → 7 | `metrics.py` 成为唯一真源，含方向性 | `metrics.py`、`evaluation.py`、`model_selector.py`、`stop_policy.py`、`predictor.py`、`final_evaluator.py` | 13 + 19 例 |
| 4 | 字符串标签 | 画像阶段建立映射，全链路统一编码/解码 | `label_encoding.py`(新)、`data_inspector.py`、`tabular_data.py`、三个 text adapter、`final_evaluator.py` | 15 例 |
| 5 | 配置可搬运 | 相对路径按**配置文件目录**解析 | `task_config.py`、`configs/task1.json`、`task2.json`、`image_smoke.json`、`tests/__init__.py` | 3 例 |

---

## 2. 图像任务：训练时长从"经验常数"到"闭环 + 实测"

### 2.1 旧估算器长什么样，错在哪

旧实现是 `plan_validator.py` 顶部的一张表：

```python
REFERENCE_SECONDS_PER_IMAGE = {
    "cpu":  {small_cnn: (32, 0.0016), mobilenet: (128, 0.0320), resnet18: (128, 0.0411)},
    "cuda": {small_cnn: (128, 0.00030), mobilenet: (224, 0.00080), resnet18: (224, 0.00100)},
}
scale = (image_size / reference_size) ** 2
return 10.0 + samples * epochs * seconds_per_image * scale * 1.25
```

四个部件：**设备二值分桶**（cpu/cuda）、**锚点分辨率**、**面积平方外推**、**1.25 系数 + 10s 固定项**。

问题不在"常数不准"，而在**结构不对**：

| 现象 | 实测 | 结论 |
|---|---|---|
| 锚点处很准 | 1.02× / 0.88× / 1.02× | 说明标定本身没错 |
| 离开锚点系统性偏 | resnet18@32px 只给 203s，实际约 933s（**低估 4.6×**） | 面积平方外推在"小图 + 大模型"方向失效 |
| 反向也错 | small_cnn@128px 高估 **2.2 倍** | 同一个指数同时拟合两端不可能 |
| `batch_size` 不入公式 | 改 batch 不影响估计 | 维度缺失 |
| GPU 桶是手填的 | resnet18 224px 一轮实际 5.0s，表里给 3–5 倍乐观 | 没有来源记录，不可复核 |

根因：**每批成本有截距**。Windows 上 `num_workers=0`，PNG 解码和 resize 在关键路径上，
这部分随像素增长但不随模型 FLOPs 增长。单锚点 + 纯 `size²` 无法同时拟合 32px 和 224px
（实测面积 ×16 时，每批耗时 small_cnn ×7.4、resnet18 ×4.4）。

后果是最坏的那种：**resnet18 + 较大训练集** 计划通过校验、实际超时，
整个 experiment 被硬杀，已经算完的 epoch 全部丢弃（状态文档里的 R1）。

### 2.2 现在的三层

| 层 | 机制 | 需要标定吗 | 解决什么 |
|---|---|---|---|
| **L1 闭环** | 执行前把**绝对截止时间**写进 profile；worker 每轮记 `epoch_seconds`，剩余时间 < 最慢轮 ×1.25 就在 epoch 边界停，`stopped_reason="time_budget"` | 不需要 | 估算完全失效时也不丢模型 |
| **L2 实测** | `throughput_cache` 按 `(adapter, size, batch, train/val 样本数, os_family, device, torch_threads)` 缓存"最慢轮耗时"，命中即 `SETUP + epochs × 最慢轮 × 1.10` | 不需要 | 换机器跑一次就自举，跨 OS/跨硬件 |
| **L3 先验** | 加性代价模型 + 机器指纹，指纹不符返回 `None` | 需要 | 首次运行就有可用的守门值 |

新的代价模型（`image_runtime.py` 顶部有完整推导与来源）：

```
T = SETUP(12s) + epochs × ( warm_per_epoch + COLD_IO_PER_IMAGE × train_samples )
warm_per_epoch = n_train × (a_tr + b_tr × size²) + n_val × (a_val + b_val × size²)
COLD_IO_PER_IMAGE = 0.00185 s/图     # 5000 张 PNG 冷页缓存 17.5s/epoch vs 热 8.25s
```

关键设计是**冷 IO 按"每文件加性"而不是"乘性"**：它是重读刚写入的图片的开销，
只和每轮读多少文件有关，和网络多重无关。早期版本把它当乘数（×2.1），
结果小数据集高估 3–5 倍、大模型又低估——正是那次误拒了一个本来只花 1/10 超时的 resnet18 计划。

系数来自 12 组 `(adapter, image_size)` 实测，每组重复 3 次取中位数
（`test_tasks/calibrate_image_runtime.py`），拟合残差：small_cnn **1.9%**、resnet18 **3.0%**、
mobilenet **13.4%**（所以 mobilenet 外推额外乘 1.10 低置信上浮）。

**机器指纹**：`CALIBRATED_OS_FAMILY = "windows"`、`CALIBRATED_TRAINING_DEVICE = "cpu"`。
不符就返回 `None`——守门静默关闭，只剩硬超时，而不是拿 Windows 系数去 Linux 上"看起来权威地错"。
（`os_family` 缺失视为旧 profile，仍放行，保持向后兼容。）
GPU 桶故意留空：解码/计算的比重和 CPU 差几个数量级，宁可没有估计，也不复用 CPU 系数。

### 2.3 代码落点

| 文件 | 改动 |
|---|---|
| `superagent/runtime_environment.py`（新） | `describe_runtime()` 记录 `os_family / training_device / device_name / torch_threads / cpu_count / npu_available`；`attach_experiment_deadline()`；`remaining_training_budget()`（executor 留 10s，训练尾部留 20s） |
| `superagent/throughput_cache.py`（新） | key 构造、`measurement_from_result`、`lookup_measured_epoch_seconds`、JSON 持久化（上限 200 条，文件损坏不致命） |
| `superagent/executor.py` | 子进程启动前写入绝对 `wall_clock_deadline` |
| `superagent/image_training.py` | `time_budget_seconds`；逐轮 `epoch_seconds`；`stopped_reason`；早停指标支持 `log_loss` 且方向感知 |
| `superagent/image_runtime.py` | 新代价模型、指纹、`estimate_..._with_source()`、`recommended_image_size()` |
| `superagent/plan_validator.py` | 用带来源的估计；拒绝时**一并列出本机真正跑得动的配置**（如 `image_small_cnn@64px/3 epochs about 42s`），减少 LLM 反复试错 |
| `superagent/controller.py` | 启动时把缓存注入 `execution_context.measured_throughput`；成功实验后回灌；确定性基线计划触犯硬限制时**跳过并记 `decisions.jsonl`**，而不是整轮中止 |
| `superagent/image_inspector.py` | 画像写完整指纹 |

### 2.4 实测

- **闭环止损**（`test_tasks/verify_time_budget_guard.py`，把 `os_family` 改成 `linux` 模拟陌生机器）：
  估算返回 `None` → 30 轮 / 120s 上限 → **第 4 轮停下、exit 0、wall 90.5s、验证精度 0.55**（有模型可用）。
- **实测缓存**（`test_tasks/verify_throughput_cache.py`，1200/400 张、small_cnn@64px、6 轮）：
  解析模型 55.5s，实际 73.2s（**0.76×，偏乐观 24%**）；缓存给出 79.8s（**1.09×**）。
  把 profile 改成陌生 OS 后：第一次无估计，跑完一次即自举出 79.8s。
- **真实端到端**：small_cnn 8 轮 0.745、resnet18 6 轮 0.8025；resnet18 逐轮耗时
  `[3.0, 3.1, 9.3, 7.8, 7.6, 11.6]`（同一轮内 **4 倍波动**，这就是缓存还要乘 1.10 的原因）。
- **缓存命中后误拒案例**：第二次运行 resnet18@32px/12 轮，缓存判 356s > 240s → 跳过，
  错误信息里明确写 `from a measured run on this data and machine`，开发者能立刻判断可信度。

### 2.5 代价与限制（必须知道）

- 缓存按**最慢轮**外推，比均值保守约 1.4 倍；波动大的任务上会误拒
  （实测一次选中精度 0.8175 → 0.7625，−5.5 点）。可调 `MEASURED_THROUGHPUT_UPLIFT`。
- 标定值仍绑定**本机**。跨 OS 第一次实验没有守门（只有硬超时）。
- 项目只认 `torch.cuda`：昇腾 NPU 会被当成 `cpu`，指纹因此会"误判为已知机器"。
  这条不是本次改动引入的，但用 NPU 时要留意。

---

## 3. 表格：整数序数通道

### 3.1 问题

原判定只有两路：`可解析成数字 且 基数 ≥ 阈值` → numeric，否则 categorical → one-hot。
盲区是**整数但基数不大的列**（`k` 小于阈值），它们一律被 one-hot。

代价在真实数据上量过（covertype 的 43 个 0/1 列，ExtraTrees 200 棵）：

| 编码 | 维数 | accuracy | macro-F1 |
|---|---:|---:|---:|
| 全部 one-hot（旧） | 97 | 0.7948 | — |
| 序数通道（新） | 54 | **0.8185** | **+2.92 点** |

原因是 `max_features="sqrt"` 每次分裂只抽约 10 列：把二值列拆成两列高度共线的 one-hot 后，
抽到"真正那个特征"的概率几乎减半。

### 3.2 判据为什么不是 Spearman（一次自证伪）

第一版用 `|Spearman(列, 目标)| ≥ 0.10` 门控，被两个探针同时推翻：

| 反例 | 现象 | rho | 后果 |
|---|---|---:|---|
| `p2d_nominal_codes` | 效应是 `[高,0,低,0,高,0,低,0]` 的交替模式 | **0.281** | 判成序数，logistic **−19.5 点** |
| `p2e_imbalanced_macro_f1` | 3% 正类，等级效应真实单调 | **0.055** | 判成名义，通道失效 |

秩相关只刻画"平均响应的单调性"，测不出"曲线中途剧烈反向"；正类 3% 时方差极小，
相关系数又被稀释。**rho 的大小同时受效应强度和类别比例支配，不能当阈值用。**

### 3.3 最终判据：一个显著性 + 一个形状

只对 `3 ≤ k < 阈值` 的整数列生效，两个条件**同时**满足才进序数通道：

```
gain = deviance(常数比例) - deviance(保序单调拟合),  df = 单调块数 - 1
p    = chi2.sf(gain, df)            # 多分类按 one-vs-rest 做 Bonferroni
形状分 = 单调拟合解释的（按等级计数加权的）方差占比
```

- 显著性：`df` 与样本量都进入统计量尺度，所以**同一个 1% 水平对所有数据集成立**，
  不用按 `k` 调阈值。零分布实测（随机标签 500 次重抽，k=4..30 / n=200..8000）在 **0.006–0.014**，标定一致。
- 形状分 ≥ **0.80**：`p2d` 的曲线第 1 级极端低（0.031），单调拟合能吃掉巨大偏差（`p=1.3e-63`），
  但形状分只有 **0.511**，因此被拒。其余真实单调列形状分 0.899–1.000。
- 两个检验互补：显著性负责"噪声不通过"，形状负责"非单调不通过"。单用形状也不行
  （k=4 时纯噪声中位数就有 0.64–0.73，固定 0.80 会有 36–42% 假阳性）。

**仍无条件走序数**：`k == 2`（两个水平下 0/1 与 one-hot 只差符号，线性模型可自行翻转）、
显式声明 `data.columns.ordinal`、以及 `ordinal_preprocessing="onehot"` 显式退回旧行为（供 A/B 与误判兜底）。

### 3.4 实测（六个探针 + 两个真实任务）

| 任务 | 通道 | logistic one-hot→ordinal | trees one-hot→ordinal | 维数 |
|---|---|---|---:|---:|
| `p2a_ordinal_levels` | num2 / **ord5** / cat1 | 0.7245→0.7310 | 0.6705→**0.7005** (+3.00) | −35 |
| `p2b_binary_block` | num5 / **ord40** | 0.9172→0.9189 | 0.7011→**0.7306** (+2.94) | −40 |
| `p2c_ordinal_declared` | num2 / **ord1**(声明) / cat1 | 0.6300→0.6267 | 0.6267→0.6383 | −11 |
| `p2d_nominal_codes` | num2 / ord0 / cat1 | **0.8312→0.8312** | **0.8362→0.8362** | 0 |
| `p2e_imbalanced_macro_f1` | num2 / **ord3** | 0.4922→0.4922 | 0.5324→0.5418 | −19 |
| `p2f_ordinal_undeclared` | num1 / **ord1** / cat1 | 0.6433→0.6483 | 0.6192→0.6250 | −13 |

关键两行：`p2d` 修复前 **−19.52 点**，现在 **0.0000**（通道为空，两个变体完全一致）；
`p2e` 修复前通道为空，现在 3 列。

真实数据：covertype（28000×54）**ord43**，ExtraTrees 0.7948 → **0.8185**、F1 +2.92、训练 **62.1s → 8.3s**；
Wine 对照（124×13，全连续）通道 0 列，Δ **精确为 0** —— 证明改动作用域没有波及连续量。

### 3.5 三个实现陷阱（踩过）

1. **不能用 `OrdinalEncoder`**：它按字符串字典序排（`"10" < "2"`），而 csv 是按字符串读入的；
   `unknown_value=-1` 还会造出"比最小值更小"的假水平。序数组改为复用 `_coerce_numeric`，数值序天然单调。
2. **插补必须用中位数**：中位数在序数尺度上有定义（均值没有），与 numeric 分支一致，
   于是序数组可以直接复用 numeric 的 pipeline，不需要新算子。
3. **`ColumnTransformer(remainder="drop")` 会静默丢列**：构造函数里加了断言，
   numeric/ordinal/categorical 三组索引互斥且并集覆盖全部列。

### 3.6 画像与提示词接口

`data_profile` 新增：`ordinal_column_indices`、`ordinal_columns`(csv)、`ordinal_feature_count`、
`ordinal_evidence`（含 `levels` / `monotonicity` / `shape_score` / `monotone_p_value` / `reason`）、
`ordinal_evidence_truncated`、`column_type_counts`。

`reason` 能区分 `no_monotone_association`（没关联）与 `non_monotone_shape`（有关联但形状不单调），
复盘时直接看出是哪一关拒的。三个 tabular adapter 与 `plan_validator` 增加
`ordinal_preprocessing: {ordinal, onehot}`，确定性基线计划默认 `"ordinal"`。

---

## 4. 指标：3 → 7，含 log_loss

### 4.1 现在支持

`accuracy`、`macro_f1`、`balanced_accuracy`、`precision`、`recall`、`roc_auc`、`log_loss`。

`superagent/metrics.py` 是唯一真源，其它模块不再各自写 if 分支。

### 4.2 log_loss 不是"加一行枚举"

它跟其余六个有两处结构性冲突，只加枚举会**稳定选中效果最差的模型**：

**① 方向相反（唯一"越小越好"）**
全系统原本默认越大越好：`model_selector` 用 `max()`，`stop_policy` 用 `value - best`，
图像早停用 `-inf` 打底。新增 `METRIC_DIRECTIONS` + `metric_direction()` / `metric_improvement()` / `is_metric_better()`：

- 选择器改为 `max(key=(direction × value, -time))`；
- 停止策略改 `direction × (value - best)`；
- 图像早停 `best_score` 按方向以 ±inf 打底；
- `solution_lock.selection_rule` 按方向写 `Lowest validation log_loss`（否则审计文本会说谎）。

**② 只接受真概率，不接受 margin**
`roc_auc` 只看排序，`LinearSVC` 的 margin 可以直接用；`log_loss` 对未归一化的数取对数会得到
"看起来合理但毫无意义"的值。所以 `PROBABILITY_METRICS = {"log_loss"}` 只走 `predict_proba`/softmax，
且校验比"看形状"严格：元素必须在 `[0,1]`（margin 常有负数）、行和必须在 `(0, 1+1e-6]`（多分类 margin 行和≈0），
通过后才 clip 重归一。

这条必须做：去掉值域检查时，`LinearSVC` 的 3 列 margin 会被 clip 归一化，算出 **0.221 的假 log_loss**。

**版本坑**：`log_loss(eps=...)` 在 sklearn 1.5 已从签名移除，显式传直接 `TypeError`。现在不传，clip 自己做。

### 4.3 接入点（9 处）

`metrics.py` / `evaluate_classifier` / `ExperimentResult.log_loss` / `model_selector` / `stop_policy` /
`image_training`（softmax 概率 + 早停方向 + `ImageTrainingResult.log_loss`）/
`predictor.PredictionBatch.probabilities` / `final_evaluator`（`test_log_loss`）/
planner 的 history 白名单；`controller` 与 `main.py` 的打印对 `None` 做防护
（否则"某模型不报 log_loss"会让整轮崩在 f-string 上）。

**顺手修的真 bug**：`stop_policy` 在"有改进"分支写的是 `best_before = result.validation_accuracy`，
用 accuracy 覆盖了当前指标。以前 metric 只有 accuracy 族所以没暴露，换成 `roc_auc`/`log_loss` 就串量纲。

### 4.4 实测（三个确定性任务，均过 `--final-evaluate-run`）

| 任务 | 模态 | 实验（验证集） | 选中 | 测试 |
|---|---|---|---|---:|
| `p4a_log_loss` | 表格 npy | logistic **0.0460** / extra_trees 0.1550 | logistic | **0.0573** |
| `p4b_log_loss_string_labels` | 表格 csv | logistic **0.5493** / extra_trees 0.5886 | logistic | **0.5430** |
| `p4c_log_loss_text` | 文本 csv | word-tfidf 0.6623 / char-tfidf **0.6570** | char-tfidf | **0.6666** |

两个值得记的发现：

- **p4a 是"为什么需要 log_loss"的最好例子**：两个实验的 accuracy 和 roc_auc 都是 **1.0000** 完全并列，
  只有 log_loss 分得开（3.4 倍差距）。按 accuracy 这时只能靠训练时间 tie-break，等于随机。
- **p4c 是反例，得如实说**：log_loss 选的 char-tfidf 的 accuracy 反而更低（0.5904 vs 0.6043）。
  它优化的是概率质量而不是硬正确率，在这里牺牲了约 1.4 点 accuracy。

### 4.5 代价

没有概率校准流程：`RandomForest`/`ExtraTrees` 的 `predict_proba` 偏向 0/1，log_loss 会重罚，树模型天然吃亏；
`LinearSVC` 这类无概率适配器在 log_loss 下会被直接排除（选择器给出指向 `predict_proba` 的专门错误信息）。

---

## 5. 字符串标签

新增 `superagent/label_encoding.py`：`describe_labels()` / `encode_labels()`。

- 全整数标签**保持原值不重排**（与交付约定一致，不影响既有任务）；
- 需要映射时按 `sorted()` 建立，**与行序无关**（同一份数据的任意切分映射一致）；
- `["1","01","2"]` 这类会退回映射（避免把两个不同标签合到一个码上）；未知标签**显式报错**。

映射写入 profile 的 `label_encoding` / `label_mapping` / `class_labels`，四处共用：
`load_tabular_split`、`load_tabular_csv`、`load_text_csv`（三个文本 adapter）、`final_evaluator._load_test_labels`。

输出约定：`predictions.npy` **保持整数码**（下游训练/评测用），`test_predictions.csv` 与
`class_recall` 的键**解码回原标签**（人读的报表用）。

实测：`p3a`（npy，`no`/`yes`，按 roc_auc 选模）logistic 0.7989 vs trees 0.7778，测试 0.7569；
`p3b`（csv，`>50K`/`<=50K`，按 macro_f1）验证 0.7013 / 测试 0.6983。两次 `--final-evaluate-run` 均通过。

---

## 6. 配置可搬运 + 那 3 个长期失败的测试

三例是 `test_controller_feedback_loop`、`test_controller_resumes_incomplete_experiment_without_rerunning_done`、
`test_loads_task1_execution_settings`，报的都是同一个错：

```
FileNotFoundError: Configured file does not exist: D:\python_project\...\C_train.npy
```

**不是代码 bug，是 11-20 迁移残留**：项目从 `D:\python_project` 挪到 `try_1/superagent_project`，
但 `configs/*.json` 里还是旧的绝对路径；而 `load_task_config` 会校验文件存在。
真正的数据一直在 `try_1/Released_Task1-...` 下。

修法让配置**可搬运**，而不是改测试绕过：

1. `_resolve_configured_path(value, base_dir)`：相对路径按**配置文件所在目录**解析
   （以前按进程 cwd，同样不可靠——不能"在哪个目录敲命令"就换一份数据）；
2. `task1` / `task2` / `image_smoke` 三个 config 全改相对路径（`../../Released_Task1-...`、`../smoke_data/...`）；
3. 两个 controller 测试要把配置复制到临时目录，复制前用 `tests/__init__.absolutize_config_paths()` 固化绝对路径，
   否则副本会去临时目录旁边找数据。

**给组员的直接影响**：新写 config 时**建议用相对路径**（相对该 json 所在目录），
这样整包换机器/换路径都能跑；老配置用绝对路径也仍然有效。

---

## 7. 文件清单

**新增（10）**

| 文件 | 作用 |
|---|---|
| `superagent/runtime_environment.py` | 机器指纹 + 绝对截止时间 |
| `superagent/throughput_cache.py` | 实测每轮耗时缓存 |
| `superagent/label_encoding.py` | 字符串标签映射与编解码 |
| `tests/test_time_budget_and_fingerprint.py` | 14 例 |
| `tests/test_throughput_cache.py` | 14 例 |
| `tests/test_image_runtime_estimator.py` | 16 例（加性模型/指纹/外推） |
| `tests/test_column_type_ordinal.py` | 25 例 |
| `tests/test_metric_enumeration.py` | 13 例 |
| `tests/test_log_loss_metric.py` | 19 例 |
| `tests/test_string_labels.py` | 15 例 |

**主要修改（按模块）**

- 图像：`image_runtime.py`、`image_training.py`、`image_inspector.py`、`plan_validator.py`、`executor.py`、`adapters/image_base.py`
- 表格：`tabular_data.py`（类型推断与预处理器，改动最大）、`data_inspector.py`、`experiment_registry.py`、
  `adapters/tabular_{baseline,extra_trees,random_forest}.py`
- 指标/标签：`metrics.py`、`evaluation.py`、`model_selector.py`、`stop_policy.py`、`predictor.py`、
  `final_evaluator.py`、`adapters/base.py`、`adapters/text_pair_*.py`
- 流程：`controller.py`、`task_config.py`、`llm/experiment_planner.py`、`llm/factory.py`、`main.py`
- 配置：`configs/{task1,task2,image_smoke}.json`（相对路径）、`configs/llm.json`（新增 `planner_plan_repairs: 2`）

---

## 8. 怎么验证（可直接复制）

> 下面第 2–4 步的脚本在**本地开发目录** `Final Project/test_tasks/`（21 个脚本，共 164KB）。
> 这个目录同时存着 4.1GB 的探针数据集，所以**没有纳入仓库**；需要的话我可以单独归档脚本和报告。

```bash
cd "Final Project/try_1/superagent_project"

# 1) 全量回归：应为 196 例 OK（约 105 秒）
python -m unittest discover -s tests

# 2) 图像闭环止损（模拟陌生机器，30 轮 / 120s → 第 4 轮停下）
cd ../../test_tasks && python verify_time_budget_guard.py

# 3) 实测缓存自举（换机器跑一次即产生估计）
python verify_throughput_cache.py

# 4) 序数通道 A/B（六个探针，确定性，无 LLM）
python verify_ordinal_channel.py
python verify_ordinal_real_tasks.py          # covertype + Wine 对照

# 5) 端到端（表格 / 文本 / 图像任选）
cd ../try_1/superagent_project
python main.py --config ../../test_tasks/p4a_log_loss/config.json
python main.py --config ../../test_tasks/p4a_log_loss/config.json --final-evaluate-run runs/<上面的 run 目录>
```

产物位置：`superagent_project/runs/<task>_<timestamp>/`，其中
`data_profile.json`（含指纹与序数证据）、`experiments.jsonl`（逐实验指标）、
`solution_lock.json`（选模规则）、`final/test_metrics.json`（测试集指标，含 `test_log_loss`）。

---

## 9. 对小组其他成员的影响（必读）

1. **metric 名可以用 7 个了**，但 `log_loss` 需要模型有概率：树模型会被重罚、`LinearSVC` 直接不可用。
   不要用 log_loss 去比较"有概率"和"无概率"的模型。
2. **`ordinal_preprocessing` 默认 `"ordinal"`**：同一份配置在不同版本上结果可能不同。
   需要复现旧结果时，在那个实验的 `hyperparameters` 里显式写 `"ordinal_preprocessing": "onehot"`。
3. **画像字段有新增**，如果组内有脚本读 `data_profile.json`，注意 `numeric_column_indices` 的语义没变，
   但**列不再全部落在 numeric/categorical 两组**——请改用 `column_type_counts` 或 `ordinal_column_indices`。
4. **配置路径**：新配置建议相对路径（相对 json 所在目录）；`--config` 从任意目录传都可以。
5. **图像任务的"跳过"是正常行为**：如果确定性基线计划超限，会打印 `skipped; ...` 并继续，
   不再整轮失败；`decisions.jsonl` 里能查到原因。
6. **`predictions.npy` 仍是整数码**，别拿它和原始字符串标签直接比；用 `test_predictions.csv`。

---

## 10. 已知限制 / 建议的下一步

| 限制 | 现状 | 建议 |
|---|---|---|
| 无概率校准 | 树模型在 log_loss 下吃亏 | 加 `CalibratedClassifierCV`（sigmoid/isotonic）作为一个 adapter 变体 |
| 缓存按最慢轮外推 | 波动大时误拒（本轮 −5.5 点案例） | 改存"中位数 + 分位数"，或按 `epoch_seconds` 的 p80 而不是 max |
| 标定值绑定本机 | 跨 OS 首次实验无守门 | 在 HPC 节点跑一次 `calibrate_image_runtime.py` 并写入新的指纹常量 |
| 只认 `torch.cuda` | 昇腾 NPU 被当成 cpu | `runtime_environment.describe_runtime()` 已能报 `npu_available`，适配器选择设备时接上即可 |
| 高基数名义整数列 | zipcode/ID 仍判 numeric | 需 target encoding，属独立改动 |
| 容器格式 | 只支持 npy_split / csv_split / image_directory_split | parquet、单文件未切分 CSV、图片 + labels.csv 仍是缺口 |

---

维护：本次改动的设计推导与全部实测数据在本地 `Final Project/test_tasks/` 下
（`COLUMN_TYPE_ORDINAL_REPORT.md`、`METRICS_AND_LABELS_REPORT.md`、
`IMAGE_RUNTIME_CALIBRATION_GUIDE.md` §九/§十、`CURRENT_STATUS_2026-11-23.md`），
该目录未纳入仓库（含 4.1GB 探针数据），需要时可单独归档。

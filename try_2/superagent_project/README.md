# Budgeted Autonomous Classification SuperAgent

一个可审计、可恢复、受预算与资源限制的自主分类实验系统。

给定一个陌生分类任务、已经划分好的数据集和评价指标，SuperAgent 会自动：

1. 检查数据并判断任务模态（表格、文本对或图像）；
2. 运行确定性的初始基线；
3. 将数据画像和验证集实验历史交给 LLM Experiment Planner；
4. 校验 LLM 提出的模型与超参数，隔离运行实验；
5. 根据验证集结果继续试验、调参或停止；
6. 自动选择验证集上最好的成功模型并锁定方案；
7. 在锁定后单独执行一次最终测试集评估；
8. 导出可以独立加载并对无标签数据预测的最终模型。

在一次自主运行开始后，不需要、也不应该由学生手动修改代码、指定下一次实验或挑选最终模型。所有决策、失败和停止原因都会写入运行目录，方便复现和检查。

> 当前版本支持分类任务，验证集选模指标可设为 `accuracy`、`macro_f1` 或 `balanced_accuracy`。数据提供者需要预先准备 train/validation/test 三个 split；Agent 会在正式运行中自行完成数据探查，但不会替数据提供者收集数据或决定原始数据如何划分。

## 1. 系统架构

```text
Task JSON + Dataset
        |
        v
 TaskConfig validation
        |
        v
 DataInspector / ImageInspector ----> data_profile.json
        |
        v
 deterministic baseline experiments
        |
        v
+---------------- autonomous loop ----------------+
| validation history -> LLM Planner                |
|                         |                         |
|                         v                         |
|                 PlanValidator                     |
|                         |                         |
|                         v                         |
|          isolated ExperimentExecutor              |
|        (timeout / CPU / RAM / GPU monitoring)     |
|                         |                         |
|                         v                         |
|               validation metrics ----------------+
+--------------------------------------------------+
        |
        v
 automatic best-model selection + solution_lock.json
        |
        v
 one-time FinalEvaluator -> exported model + test predictions
        |
        v
 FinalModelRunner -> predictions for new unlabeled data
```

核心原则：

- **验证集驱动决策**：Planner 和模型选择器都看不到测试集指标。
- **测试集隔离**：只有产生 `solution_lock.json` 后才能进行最终测试。
- **方案白名单**：LLM 只能选择系统注册且参数通过校验的 Adapter。
- **完整日志**：成功、失败、LLM 错误、费用和停止原因全部保留。
- **资源保护**：每个实验在独立进程中执行，并限制时间、CPU 线程和总内存。
- **可恢复**：中断后从 `run_state.json` 恢复，不重复已经完成的实验。
- **产物校验**：最终模型和预测文件使用 SHA-256 检查完整性。

## 2. 支持范围

| 数据模态 | 输入格式 | 当前候选模型 |
| --- | --- | --- |
| 表格分类 | `npy_split` 或 `csv_split` | 类型感知 Logistic Regression、Extra Trees、Random Forest |
| 单文本或文本对分类 | `csv_split` | Word TF-IDF + Logistic Regression、Character TF-IDF + Logistic Regression / Linear SVM |
| 图像分类 | `image_directory_split` | Small CNN、ResNet18、MobileNetV3 Small |

图像目录必须使用以下类别子目录结构，并且三个 split 的类别名称必须一致：

```text
dataset/
├── train/
│   ├── cat/
│   └── dog/
├── validation/
│   ├── cat/
│   └── dog/
└── test/
    ├── cat/
    └── dog/
```

图像检查器会统计图片尺寸、颜色模式、类别和类别分布，自动将灰度图转换为 RGB，并报告损坏图片。图像训练会记录训练/验证准确率、macro F1、泛化差距、最佳 epoch、训练时间和 GPU 峰值显存；CUDA 不可用时会回退到 CPU。

## 3. 环境安装

项目在以下环境验证：

- Windows PowerShell
- Python 3.14.6
- NVIDIA CUDA 12.8 对应的 PyTorch 2.11.0

在项目父目录创建并激活虚拟环境：

```powershell
cd D:\python_project
py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install --index-url https://pypi.org/simple `
  -r .\superagent_project\requirements.txt
```

如果已有虚拟环境，只需激活并安装依赖。不要在 URL 中写 `https\://`，也不要在包名末尾添加反斜杠；PowerShell 的换行符是反引号 `` ` ``。

安装后检查：

```powershell
cd D:\python_project\superagent_project
python -c "import sklearn, torch, torchvision; print(sklearn.__version__); print(torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

精确依赖版本记录在 `requirements.txt`。当前 PyTorch 是 CUDA 12.8 构建；没有 NVIDIA GPU 时仍可运行表格、文本和小型 CPU 图像测试，但正式图像训练会更慢。

## 4. 配置 API Key

只有 `planner_mode: "llm"` 的自主搜索需要 API Key。测试、最终评估和最终模型推理不需要调用 LLM。

当前 PowerShell 窗口中设置：

```powershell
$env:AIGC_API_KEY = "你的 API Key"
```

如果 Key 保存在本机文件中：

```powershell
$env:AIGC_API_KEY = (
  Get-Content -LiteralPath "D:\python_project\api_key.txt" -Raw
).Trim()

if ([string]::IsNullOrWhiteSpace($env:AIGC_API_KEY)) {
  throw "AIGC_API_KEY is missing"
}
```

不要把 API Key 写进 JSON、Python 文件、README、Git 提交或运行日志。

可先做一次连接测试：

```powershell
cd D:\python_project\superagent_project
python -m scripts.test_llm_connection
```

LLM 地址和模型名在 `configs/llm.json`，官方价格快照和每个任务的预算在 `configs/model_pricing.json`。

## 5. 任务配置

### 5.1 通用配置

建议复制现有配置后修改，不要直接覆盖正式 Task 1/Task 2 配置：

```json
{
  "run_type": "development_run",
  "task": {
    "name": "my_unseen_classification_task",
    "metric": "accuracy"
  },
  "data": {},
  "execution": {
    "random_seed": 42,
    "evaluate_test": false,
    "planner_mode": "llm",
    "initial_deterministic_experiments": 2,
    "max_experiments": 8,
    "max_runtime_minutes": 180,
    "no_improvement_patience": 3,
    "minimum_improvement": 0.001,
    "maximum_planner_errors": 2,
    "maximum_same_experiment_errors": 2,
    "experiment_timeout_seconds": 1800,
    "maximum_memory_mb": 4096,
    "maximum_cpu_threads": 2,
    "resource_poll_interval_seconds": 0.2
  }
}
```

- `development_run`：开发、冒烟测试或陌生数据能力测试。
- `official_report_run`：最终报告使用的冻结版本正式运行；该值会进入 manifest、summary 和 solution lock。
- `evaluate_test` 必须保持 `false`；训练阶段禁止直接评估测试集。
- `planner_mode` 可用 `llm` 或 `deterministic`。
- `minimum_improvement` 和 `no_improvement_patience` 控制无提升停止。
- `experiment_timeout_seconds`、`maximum_memory_mb` 和 `maximum_cpu_threads` 是单次实验限制。

### 5.2 NPY 表格任务

```json
"data": {
  "format": "npy_split",
  "train_features": "D:/data/task/C_train.npy",
  "train_labels": "D:/data/task/y_train.npy",
  "validation_features": "D:/data/task/C_val.npy",
  "validation_labels": "D:/data/task/y_val.npy",
  "test_features": "D:/data/task/C_test.npy",
  "test_labels": "D:/data/task/y_test.npy"
}
```

### 5.3 CSV 表格、单文本或文本对任务

```json
"data": {
  "format": "csv_split",
  "train": "D:/data/task/train.csv",
  "validation": "D:/data/task/val.csv",
  "test": "D:/data/task/test.csv",
  "target_column": "label_pairwise",
  "columns": {
    "text": ["headline_a", "headline_b"]
  }
}
```

推荐用可选的 `columns` 显式声明一至两列 `text`，或者完整声明 `numeric` 和 `categorical` 列。没有声明时，系统根据列内容判断模态，而不是根据任务名称判断。

### 5.4 图像目录任务

```json
"data": {
  "format": "image_directory_split",
  "train_directory": "D:/data/images/train",
  "validation_directory": "D:/data/images/validation",
  "test_directory": "D:/data/images/test"
}
```

## 6. 运行完整自主流程

以下示例以 Task 1 为例。Task 2 只需把配置改成 `configs/task2.json`。

### 第一步：运行自主实验

```powershell
cd D:\python_project\superagent_project
python .\main.py --config .\configs\task1.json
```

也可以把输出写到指定目录：

```powershell
python .\main.py `
  --config .\configs\task1.json `
  --runs-dir D:\experiment_results\task1
```

运行结束会显示真实的运行目录，例如：

```text
Run directory: D:\experiment_results\task1\released_task_1_<timestamp>
```

请复制程序实际输出的目录名。命令中的 `<timestamp>`、`<run_id>` 或 `<运行目录>` 都只是占位符，不能原样输入 PowerShell。

### 第二步：查看自动锁定结果

```powershell
$run = "D:\experiment_results\task1\released_task_1_实际时间戳"
Get-Content -LiteralPath "$run\solution_lock.json"
Get-Content -LiteralPath "$run\stop_reason.json"
Get-Content -LiteralPath "$run\summary.json"
```

### 第三步：仅执行一次最终测试评估

```powershell
python .\main.py `
  --config .\configs\task1.json `
  --final-evaluate-run $run
```

此模式不会调用 LLM、不会运行新实验、不会重新选模型。以下情况会被拒绝：方案未锁定、配置不匹配、模型被手动替换，或该运行已经做过最终测试评估。

### 第四步：校验最终产物

```powershell
python -m scripts.verify_final_artifacts `
  --run-directory $run
```

### Task 2

```powershell
python .\main.py --config .\configs\task2.json
```

Task 2 完成后，使用它自己输出的运行目录和 `configs/task2.json` 重复第二至第四步。两个任务必须使用相同的冻结 Git commit，且运行过程中不得修改代码或配置。

## 7. 恢复中断的运行

每次运行都会原子更新 `run_state.json`。已经完成的实验从各自的 `plan.json` 和 `experiment.json` 恢复；已经生成计划但尚未完成的实验会重新执行，不会再次向 LLM 请求另一个计划。

```powershell
python .\main.py `
  --config .\configs\task1.json `
  --resume-run D:\experiment_results\task1\released_task_1_实际时间戳
```

已经包含 `solution_lock.json` 的运行已经完成，不能恢复。恢复记录写入 `resume_events.jsonl`，中断期间不计入 Agent 的有效运行时间。

## 8. 对新数据进行推理

最终推理不需要任务配置、标签、API Key 或 LLM。必须先完成最终测试评估并生成 `final/` 目录。

### 表格 NPY

```powershell
python .\main.py `
  --predict-run $run `
  --input-data D:\new_data\features.npy `
  --prediction-output-dir D:\predictions\task1_batch
```

### 文本对 CSV

```powershell
python .\main.py `
  --predict-run $run `
  --input-data D:\new_data\headline_pairs.csv `
  --prediction-output-dir D:\predictions\task2_batch
```

CSV 应包含导出模型保存的两列文本特征，但不需要标签列。

### 图片目录

```powershell
python .\main.py `
  --predict-run $run `
  --input-data D:\new_images `
  --prediction-output-dir D:\predictions\images
```

图片推理支持单张图片或图片目录。图像 CSV 输出列为：

```text
row_id,input_path,prediction,class_name
```

输出目录必须为空或尚不存在，系统不会覆盖旧预测。成功后目录包含：

- `predictions.csv`
- `predictions.npy`
- `prediction_manifest.json`

加载模型前会先检查最终模型 SHA-256。

## 9. 运行产物和审计日志

每次自主运行创建独立目录，主要文件如下：

| 文件 | 用途 |
| --- | --- |
| `run_manifest.json` | 任务、配置、版本和运行身份 |
| `data_profile.json` | Agent 自动生成的数据画像 |
| `plans.jsonl` | 所有经过验证的实验计划 |
| `planner_decisions.jsonl` | LLM 的建议、理由和响应记录 |
| `experiments.jsonl` | 各实验状态和验证结果 |
| `decisions.jsonl` | Controller 的循环决策 |
| `errors.jsonl` | Planner 或实验错误（如果发生） |
| `budget_ledger.jsonl` | Token 使用量和模型费用 |
| `run_state.json` | 中断恢复检查点 |
| `stop_reason.json` | 自动停止原因 |
| `summary.json` | 自动选择的最佳验证集模型 |
| `solution_lock.json` | 最终方案锁定证据及模型路径 |
| `experiments/exp_*/` | 每次实验的计划、指标和模型 |

最终测试评估成功后还会生成：

```text
final/
├── final_model.joblib      # 表格或文本模型
├── final_model.pt          # 图像模型；与 joblib 二选一
├── test_predictions.csv
├── test_predictions.npy
├── test_metrics.json
└── artifact_hashes.json
```

日志不会因为实验失败而删除对应记录。资源超时、内存超限、子进程崩溃和 CUDA OOM 会被分类并反馈给 Planner；图像 Planner 可以据此降低 batch size、image size 或模型规模。

## 10. 测试

完整测试套件：

```powershell
cd D:\python_project\superagent_project
python -m unittest discover -s tests -v
```

测试使用 Fake Planner，不消耗 API 额度。覆盖 Controller 自主闭环、计划校验、预算、恢复、资源限制、最终评估、最终推理，以及图像检查/训练/产物重新加载。

离线测试 Experiment Planner：

```powershell
python -m scripts.test_experiment_planner --task-config .\configs\task1.json
python -m scripts.test_experiment_planner --task-config .\configs\task2.json
```

允许一次真实、受预算限制的 Planner 请求：

```powershell
python -m scripts.test_experiment_planner `
  --task-config .\configs\task1.json `
  --live
```

创建并运行一个不依赖外部数据的图像冒烟任务：

```powershell
python -m scripts.create_synthetic_image_dataset --force
python .\main.py --config .\configs\image_smoke.json
```

`image_smoke.json` 默认使用 deterministic Planner 和一个小型 CNN，适合确认图像链路是否能在本机运行。

## 11. 正式运行规范

正式报告运行前冻结同一个代码版本：

```powershell
git status --porcelain
git add .
git commit -m "Freeze final SuperAgent implementation"
git tag final-superagent-v1
git rev-parse HEAD
git rev-parse "final-superagent-v1^{commit}"
```

开始 Task 1/Task 2 正式运行前应满足：

- 工作区干净；
- 两个任务使用同一 commit；
- 配置中的 `run_type` 为 `official_report_run`；
- 运行期间不改代码、不改配置、不提示 Planner 下一步做什么；
- 不因结果不好而重跑后只报告更好的一次；
- 不手动删除失败记录或挑选最终模型；
- 只在 `solution_lock.json` 生成后执行一次最终测试评估。

## 12. 已完成的陌生任务验证

冻结版本 `3a817efe8c598feb56ca45a4795569b809add620` 曾在 UCI Bank Marketing 陌生表格分类任务上进行独立的 `development_run`：

- 样本数：45,211；16 个输入特征；固定随机种子 42；
- Agent 自动运行 4 个实验，并因 `no_validation_improvement` 自动停止；
- 自动选择 One-Hot Logistic Regression；
- 验证集 accuracy：`0.897375`；
- 锁定后一次性测试 accuracy：`0.894132`；
- 最终模型 SHA-256：`d36b7f06a441422c68ba2fe7b3682b44307d08319c68457f85df7385354c9de7`；
- 最终产物校验通过。

这说明系统能够在未针对该数据集编写专用训练逻辑的情况下完成数据探查、实验迭代、自动选择、方案锁定、最终评估和模型导出。它不代表所有陌生任务都会获得最佳结果，也不等于 Released Tasks 的正式成绩。

## 13. 已知限制

- 当前支持 `accuracy`、`macro_f1` 和 `balanced_accuracy` 三种主评价指标。
- 陌生数据必须预先划分成支持的 train/validation/test 格式。
- Planner 只能在 Adapter 白名单内组合模型和参数，不会任意生成并执行 Python 代码。
- 表格管线偏向类别型/字符串特征；对连续数值特征的专用变换仍可扩展。
- 类别不平衡任务只按 accuracy 自动选模时，少数类 recall 可能不理想；报告中应同时查看 macro F1、balanced accuracy、per-class recall 和混淆矩阵。
- ResNet18/MobileNet 的 `pretrained: true` 可能在首次使用时下载权重；无网络测试应使用非预训练设置。
- CPU 内存限制按实验进程树监控；GPU 显存由 PyTorch 峰值统计与 CUDA OOM 分类处理。

## 14. 项目结构

```text
superagent_project/
├── main.py                       # 统一命令行入口
├── requirements.txt              # 固定版本依赖
├── configs/                      # 任务、LLM、费用配置
├── adapters/                     # 表格、文本、图像模型 Adapter
├── superagent/
│   ├── controller.py             # 自主循环控制器
│   ├── data_inspector.py         # 表格/文本数据探查
│   ├── image_inspector.py        # 图像数据探查
│   ├── experiment_registry.py    # Adapter 白名单与默认实验
│   ├── plan_validator.py         # LLM 计划安全校验
│   ├── executor.py               # 隔离实验执行
│   ├── resource_monitor.py       # 资源监控
│   ├── recovery.py               # 中断恢复
│   ├── final_evaluator.py        # 一次性最终测试评估
│   ├── final_model_runner.py     # 无标签推理
│   └── llm/                      # API、预算和 Planner
├── scripts/                      # 验证和冒烟脚本
├── tests/                        # 自动测试
└── runs/                         # 本地运行产物（Git 忽略）
```

## 15. 常见问题

**`ModuleNotFoundError: No module named 'superagent'`**

请先 `cd` 到含有 `main.py` 和 `superagent/` 的项目根目录，再用 `python .\main.py ...` 或 `python -m scripts.xxx` 运行。

**`AIGC_API_KEY is not set`**

当前配置使用了 `planner_mode: "llm"`，请在同一个 PowerShell 窗口设置环境变量。不要把 Key 写入仓库。

**`Get-Content .\runs\<run_id>\solution_lock.json` 报路径含非法字符**

尖括号内容是占位符。请从程序输出中复制真实运行目录，或使用：

```powershell
Get-ChildItem .\runs -Directory | Sort-Object LastWriteTime -Descending
```

**预测时报 `Prediction output directory is not empty`**

换一个新目录，或把旧结果移动到其他位置。系统故意拒绝覆盖已有预测。

**`pip` 显示 `No matching distribution found`**

确认 URL 写作 `https://...`，而不是 `https\://...`；确认包名末尾没有 `\`；必要时显式使用官方源：

```powershell
python -m pip install --index-url https://pypi.org/simple -r .\requirements.txt
```

---

提交给老师前，建议同时保留：冻结 commit、任务配置、完整运行目录、最终产物校验输出和实验记录。README 说明系统如何运行；运行日志证明系统确实自主完成了运行。

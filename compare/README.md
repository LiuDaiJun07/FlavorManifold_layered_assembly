# 对照实验研究档案

本目录按“研究方法—实验结果—数值数据—计算缓存—复现记录”组织。原项目的 `src/`、`Output/` 保持原位。先阅读 [结果总览](results/RESULTS.md) 或 [分领域实验索引](results/README.md)，方法调整见 [施行性核查](methodology/feasibility.md)，原始要求见 [对照实验设计](../对照实验设计.md)。

| 目录 | 学术用途 | 内容 |
| --- | --- | --- |
| `methodology/` | 研究设计与方法依据 | 可行性核查、文献与阈值来源候选 |
| `methods/` | 方法实现 | 预测建模、关键因子分析、生成与迁移三类研究代码 |
| `results/` | 实验结果与展示 | C01–C21 报告、统计表、图及跨实验综合比较 |
| `data/` | 分析数据与协议 | 折外预测、生成样本、统计量、折划分、参考表 |
| `cache/` | 可续算的计算中间量 | 分折预测、嵌入、特征选择与重抽样断点 |
| `provenance/` | 计算溯源 | 环境版本、日志、执行状态、哈希清单、历史档案 |
| `verification/` | 质量控制 | 协议核查、输入核查、验证记录与缩小规模试运行 |
| `workflow/` | 实验工作流 | 执行调度、结果汇总与统一路径规则 |

三个研究领域统一命名为 `q1_prediction`（感官预测）、`q2_factor_analysis`（关键因子分析）、`q3_generation_transfer`（风格生成与迁移）。`results/` 内按实验编号和研究内容分目录，每项实验分别存放 `reports/`、`tables/`、`figures/`；C15 与 C16 共用生成验证目录。A01 为额外的复现审计。各领域目录内有实验索引。

`data/` 保存可供后续分析的数值结果，`cache/` 保存续算断点，两者均已保留。目录迁移记录见 [文件映射与哈希清单](provenance/manifests/artifact_reorganization.json)。历史档案内的旧路径用于追溯。

## 运行与复核

在项目根目录运行：

```powershell
$env:NJOB='6'
$env:TREE_SCALE='1'
$env:FOLDS='10'
.venv/Scripts/python.exe -B compare/run_all.py
```

默认跳过已完成且脚本未变化的实验。`--only C1,C2` 指定实验，`--rerun` 重新执行；改变方法或模型参数时还须核查对应缓存是否适用。正式设置为十折、全量树数，缩小设置的产物进入 `verification/smoke/`。默认使用六个工作进程，模型与 BLAS 内部单线程。

```powershell
.venv/Scripts/python.exe -B compare/workflow/summarize.py
.venv/Scripts/python.exe -B compare/verification/verify_protocol.py
.venv/Scripts/python.exe -B compare/verification/verify_optional_inputs.py
.venv/Scripts/python.exe -B compare/verification/verify_layout.py
```

执行状态保存在 `provenance/execution/run_status.json`，科学完成情况见 [实验登记表](results/experiment_registry.csv)。C8 缺真实独立评分，C14 缺核验单位和独立阈值，C13 尚缺选择频率的原数据抽样 BCa 区间。C10 的收敛警告、统计检验的解释边界、历史 BASE5 复现漂移均保留在报告中。

# 复现说明

本目录是 MathorCup 2026 赛道 A（啤酒酿造化合物与风味预测）的完整实现，三个问题的代码、实验记录与答案齐全。

## 环境

项目级 Python 环境在 `main/.venv`，由 anaconda 的 Python 3.13 以 `--system-site-packages` 方式创建，不需要联网安装任何包。

```powershell
& "main\.venv\Scripts\python.exe" -X utf8 "main\src\q1\05_train_et.py"
```

依赖：numpy / pandas / scipy / scikit-learn / matplotlib / networkx / openpyxl，全部来自 anaconda 基础环境。

两处刻意的工程决定：

一、不用 torch 或 geoopt。250 个节点的双曲嵌入用纯 numpy 手写黎曼梯度下降即可，梯度用有限差分校验到 1e-10 量级，省掉一个 200MB 的依赖。

二、estimator 的并行度由环境变量 `NJOB` 控制，默认 `1`。joblib 在 Windows 沙盒里建线程池要打开命名管道，会被拒绝（`PermissionError: [WinError 5]`），所以受限环境里只能 `NJOB=1`；在普通桌面上设 `NJOB=-1` 就能吃满全部核心，这也是补丁里唯一值得做的提速手段。

## 目录

```
main/src/q1/     问题一：0-6 分制感官评分
main/src/q2/     问题二：Top-3 关键因素化合物
main/src/q3/     问题三：风格条件生成与跨风格迁移
main/Output/     每一步的实验记录（md 报告、csv 表、png 图、npz 中间量）
_v1_backup/      打补丁前的旧产物备份（Output 三个子目录各一份，只增不删）
答案/            最终交付的答案表与说明
```

## 怎么跑（v2 补丁后，推荐用一键脚本）

补丁后问题一的主线是「五成员基座 → LSF 局部收缩 → ZITS 零膨胀 → 叠加」，脚本从 9 个变成 13 个，逐步手敲容易漏，所以提供了一键脚本 `main/run_all.ps1`（Windows PowerShell 与 pwsh 都能跑，文件带 BOM）：

```powershell
# 全量正式跑：10 折 + 全部树数 + 全部核心（预计 1.5~2 小时，ZITS 最慢）
powershell -NoProfile -ExecutionPolicy Bypass -File main\run_all.ps1

# 只跑问题一（问题二、三不动）
powershell -NoProfile -ExecutionPolicy Bypass -File main\run_all.ps1 -SkipQ2 -SkipQ3

# 指定 8 个 worker、10 折
powershell -NoProfile -ExecutionPolicy Bypass -File main\run_all.ps1 -Nj 8

# 冒烟测试：5 折 + 树数打 8%（几分钟，用来确认链路没断，指标必然偏低）
powershell -NoProfile -ExecutionPolicy Bypass -File main\run_all.ps1 -TreeScale 0.08 -Folds 5
```

三个参数都落到环境变量上，也可以不用脚本、直接手工设变量再逐个跑：

```powershell
$env:NJOB='-1'        # sklearn n_jobs，-1 = 全部核心
$env:TREE_SCALE='1.0' # 树数缩放，1.0 = 全量；0.08 用于冒烟
$env:FOLDS='10'       # 折数，正式 10，复核 25
& "main\.venv\Scripts\python.exe" -X utf8 "main\src\q1\06b_members.py"
```

脚本用 `$ErrorActionPreference='Stop'`，任何一步非零退出就立刻停，不会带着坏产物往下跑。

## 运行顺序

问题一（10 折、全量树数，多核约 60~90 分钟；ZITS 约占一半）：

```
01_audit.py           数据审计：缺失、左删失、风格分布、行对齐
02_destandardize.py   三口径反标准化（σ=1 / 训练折 sd / min-max），主口径 = 训练折 sd
03_features.py        特征构建模块（折内 prune + 哨兵填充 + 秩变换 + kNN 邻域，被后面各步 import）
04_hyperbolic.py      双曲嵌入模块（同上）
05_train_et.py        单模型消融 E0/E1/E2/E5，10 折折外预测
06_stacking.py        二层 Stacking（ET_E0/ET_E1 复用 05 的折外预测，RF/RD 现场训练）
06b_members.py        五成员基座 M1–M5，顺带缓存每折零占比、折内 μ/σ、kNN 邻域索引
06c_lsf.py            LSF：弱维局部收缩 + 残差邻域校正 + 孤立度分层偏差，内层 5 折选参
06d_zits.py           ZITS：零占比 ≥0.40 的维走「分类器 × 正样本回归」两阶段（最慢的一步）
06e_final.py          按 LSF→ZITS 顺序叠加，G3 双列对照，三条门禁，出最终 OOF
07_ablation.py        消融矩阵（单模型 / 二层 / 基座 / LSF / ZITS / 最终）
08_answers.py         生成交付答案表与精度图
09_scale_audit.py     对数底数与标度一致性核实（题目明确要求）
export_answers.py     把三个问题的产物汇总进 答案/
```

问题二（约 40 分钟，主要是 bootstrap 与 Lasso 交叉验证；补丁 §7 要求保持现状）：

```
01_graph.py       2.csv 共表达图、社区发现、中心性
02_top3.py        早期版本：三条证据链（置换重要性 + 图扩散 + bootstrap），保留作对照
03_stability.py   早期版本的全样本 rho 复算，保留作对照
04_final.py       最终版：全 231 列候选 + LassoCV 条件重要性 + 图扩散 + bootstrap
```

最终答案取 `04_final.py` 的结果。早期版本对候选集做了近常数列裁剪（沿用问题一的特征口径），
会把 `X4VG` 这种几乎全在检出限以下的列剔掉，导致 `X4vg` 维度选不出同名化合物，因此重做。

问题三（约 1 分钟）：

```
01_generative.py        风格条件潜因子生成模型与分布检验
02_transfer.py          跨风格迁移、缺陷化合物守护、感官画像对照
03_hyperbolic_path.py   22 风格双曲层级 + Pils/Lager→Tripel 测地线 5 个 t 的权重表与图
```

最后跑 `main/src/export_answers.py` 把结果汇总进 `答案/`。

## 跑完看哪里

一、问题一的门禁与结论在 `Output/Q1/06e_final.md`：三条硬门禁（宏 R²≥0.274、12 弱维均值≥0.075、5 强维下降≤0.005）逐条打印通过与否；G3 的两列（内层选参 / 全固定默认值）宏 R² 并列，差超过 0.010 时判选择过拟合并改以默认值列为准；若叠加后还不如基座，脚本按退路直接退回基座交付并在报告里写明。

二、消融矩阵在 `Output/Q1/07_ablation.csv`，一眼能看出每一层加了多少：单模型 → 二层 → 五成员基座 → LSF → ZITS → 最终。

三、交付答案在 `答案/`：`问题1_感官评分_分级0到6.csv`（250×53，3 列 id + 50 维等级）与 `问题1_感官评分_连续分.csv`。

四、问题三的层级图是 `答案/图_风格层级.png`（左：Poincaré 盘上 22 个风格与测地线；右：5 个 t 变化最大的 14 维），权重表在 `Output/Q3/03_path_weights.csv`（列 t/style/weight/kappa/n_guard_fix，5×22=110 行）。

## 关键口径（换数据重跑时必须注意）

一、1.csv 的底数是 log10，但标度并不统一。完整证据在 `09_scale_audit.md`：16 个已知物理量做 10 的幂反变换后落在合理区间的平均命中率是 0.9585，以 e 为底只有 0.8553；决定性的是 pH，底 10 给出 4.13，底 e 给出 1.85，低于啤酒 pH 的物理下限。另有 56 个全程非负的列给的是原始浓度而不是 log10（11 列最大值上万、15 列集中在 4 到 9），树模型不受影响，但算 OAV 时必须按列判断。`CO2.PSI.` 列更像二氧化碳体积数而不是 psi。

二、缺失不等于零。9 个含缺失列建指示列，树路径用哨兵值 `列最小值 - 10*(列跨度+1)` 填充，需要完整矩阵的图构建用折内 kNN 填充。所有统计量只在训练折上估计。

三、4.csv 的分级口径改成三口径并列（补丁 §3）：主口径是 σ=训练折该维标准差、μ=-σ·z_min，用于交付；同时报 σ=1 与 min-max（σ=6/(zmax-zmin)）两组作敏感性对照。R² 一律在 z 空间逐维算再平均，MAE 三口径都打出来，不许只报一个。旧版取 σ=1 的论证是错的，已在 `02_destandardize.md` 里改写。

四、2.csv 的行列名是可读化学名，与 1.csv 字段名不同，映射靠 8.xlsx 的 `field_name_R` 列，231 个全部对上。8.xlsx 里还有 mojibake（α 存成 `Î±`），`common.norm_name` 负责修复。

五、6.csv 是 250 个样本的全样本化合物-感官相关阵，只能用于事后方向核对与化学解释，不能参与特征选择，否则泄漏。

六、12 个弱维（clove / F_esters_flower / A_esters_flower / orange / stale_hops / A_esters_fruity / A_malt_grain / barnyard / dms / diacetyl / t2n / F_malt_grain）是**事后子集**，由旧版逐维 R² 挑出来的，不是先验定义；所以报告里同时给整体宏 R²，不只报这个子集口径。

`main/src/q1/` 下还有若干以下划线或 `_tune/_screen` 命名的脚本，是调参阶段的探索记录（表示形式对比、
超参网格、样本级聚合特征、kNN 目标编码等），不参与最终结果，保留只为可追溯。它们的负面结论已经
写进 `Output/Q1/` 的对应记录里，例如 CLR 变换、PCA 投影、双曲块、Stacked 二层在这份数据上都降低了指标。

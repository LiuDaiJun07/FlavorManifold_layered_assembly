# 一键复现脚本（v2 补丁后）
# 本机只有 Windows PowerShell 5.1，没有 pwsh；两种写法都可以：
# 用法示例：
#   powershell -NoProfile -ExecutionPolicy Bypass -File main\run_all.ps1            # 全量 10 折，所有核心
#   powershell -NoProfile -ExecutionPolicy Bypass -File main\run_all.ps1 -Nj 4       # 指定 4 进程
#   powershell -NoProfile -ExecutionPolicy Bypass -File main\run_all.ps1 -TreeScale 0.1 -Folds 5   # 冒烟（几分钟）
#   powershell -NoProfile -ExecutionPolicy Bypass -File main\run_all.ps1 -SkipQ2 -SkipQ3            # 只跑问题一
param(
    [int]$Nj = -1,               # sklearn n_jobs：-1 用全部核心，1 单核（沙盒环境只能 1）
    [double]$TreeScale = 1.0,    # 树数缩放：1.0 全量，0.1 冒烟
    [int]$Folds = 10,            # 折数：正式 10，复核 25，冒烟 5
    [switch]$SkipQ2,
    [switch]$SkipQ3
)

$ErrorActionPreference = 'Stop'
$env:NJOB = "$Nj"
$env:TREE_SCALE = "$TreeScale"
$env:FOLDS = "$Folds"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$py = Join-Path $root '.venv\Scripts\python.exe'
if (-not (Test-Path $py)) { throw "找不到虚拟环境解释器：$py" }

Write-Host "== 参数：NJOB=$Nj TREE_SCALE=$TreeScale FOLDS=$Folds ==" -ForegroundColor Cyan
Write-Host "== 解释器：$py ==" -ForegroundColor Cyan

# 依次运行一个脚本，失败立即停
function Step($name, $script) {
    $p = Join-Path $root "src\$script"
    Write-Host ""
    Write-Host ">>> $name  ($script)" -ForegroundColor Yellow
    $t0 = Get-Date
    & $py -X utf8 $p
    if ($LASTEXITCODE -ne 0) { throw "步骤失败：$script（exit $LASTEXITCODE）" }
    $dt = (Get-Date) - $t0
    Write-Host ("<<< {0} 完成，用时 {1:N1} 秒" -f $name, $dt.TotalSeconds) -ForegroundColor Green
}

# 问题一主线：反标准化 → 单模型消融 → 二层 → 五成员基座 → LSF → ZITS → 叠加 → 汇总 → 答案
Step '01 数据审计' 'q1\01_audit.py'
Step '20 反标准化（三口径）' 'q1\02_destandardize.py'
Step '05 单模型消融' 'q1\05_train_et.py'
Step '06 二层 Stacking' 'q1\06_stacking.py'
Step '06b 五成员基座' 'q1\06b_members.py'
Step '06b2 内层基座（不含外层验证折）' 'q1\06b2_inner_base.py'
Step '06c LSF 局部收缩' 'q1\06c_lsf.py'
Step '06d ZITS 零膨胀' 'q1\06d_zits.py'
Step '06e 最终叠加与门禁' 'q1\06e_final.py'
Step '07 消融矩阵' 'q1\07_ablation.py'
Step '08 答案与图表' 'q1\08_answers.py'

if (-not $SkipQ2) {
    Write-Host ""
    Write-Host ">>> 问题二（按补丁 §7 保持现状，候选仍为全 231 列）" -ForegroundColor Yellow
    Step 'Q2-1 相关图' 'q2\01_graph.py'
    Step 'Q2-2 Top3 候选' 'q2\02_top3.py'
    Step 'Q2-3 稳定性' 'q2\03_stability.py'
    Step 'Q2-4 最终名单' 'q2\04_final.py'
}

if (-not $SkipQ3) {
    Step 'Q3-1 生成模型' 'q3\01_generative.py'
    Step 'Q3-2 迁移' 'q3\02_transfer.py'
    Step 'Q3-3 风格层级与双曲路径' 'q3\03_hyperbolic_path.py'
}

Step '09 规模审计' 'q1\09_scale_audit.py'
Step '99 答案汇总' 'export_answers.py'

Write-Host ""
Write-Host "== 全部完成，产物在 main\Output\Q1|Q2|Q3 与 答案\ ==" -ForegroundColor Cyan

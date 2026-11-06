# 修复后的一键执行：备份旧产物 -> 冒烟 -> 删旧 -> 多核正式重跑
# 需要 danger-full-access（沙盒下 main\Output\Q1 与 temp 不可写，n_jobs=-1 会撞命名管道）
$ErrorActionPreference = 'Stop'

$root = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))
$py = Join-Path $root 'main\.venv\Scripts\python.exe'
$src = Join-Path $root 'main\src\q1'
$q1 = Join-Path $root 'main\Output\Q1'
$ans = Join-Path $root '答案'
$bk = Join-Path $root 'temp\_old_Q1'

if (-not (Test-Path $py)) { throw "解释器不存在：$py" }
if (-not (Test-Path $q1)) { throw "产物目录不存在：$q1" }

function Step($name, $file) {
    Write-Host ">>> $name" -ForegroundColor Cyan
    $t0 = Get-Date
    & $py -X utf8 (Join-Path $src $file)
    if ($LASTEXITCODE -ne 0) { throw "步骤失败：$file（exit $LASTEXITCODE）" }
    Write-Host ("<<< {0} 用时 {1:N0}s" -f $name, ((Get-Date) - $t0).TotalSeconds) -ForegroundColor Green
}

# ---------- 1 备份旧产物到 temp ----------
Write-Host '=== 1 备份旧产物到 temp ===' -ForegroundColor Yellow
New-Item -ItemType Directory -Force -Path $bk | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $bk '答案') | Out-Null
Copy-Item (Join-Path $q1 '*') $bk -Force -Recurse
foreach ($pat in '问题1_*.csv', '问题1_*.md', 'fig08_*.png') {
    $hit = @(Get-ChildItem $ans -Filter $pat -File -ErrorAction SilentlyContinue)
    if ($hit.Count -gt 0) { $hit | Copy-Item -Destination (Join-Path $bk '答案') -Force }
}
$n = @(Get-ChildItem $bk -Recurse -File | Measure-Object).Count
Write-Host "备份完成：$n 个文件 -> $bk"
Write-Host '备份清单：'
Get-ChildItem $bk -Recurse -File | Sort-Object Name | ForEach-Object { Write-Host ('  ' + $_.FullName.Replace($bk, '')) }

# ---------- 2 冒烟（小规模，验逻辑） ----------
Write-Host '=== 2 冒烟测试 FOLDS=2 TREE_SCALE=0.02 NJOB=1 ===' -ForegroundColor Yellow
$env:Q1_OUT = Join-Path $root '_smoke_q1'
$env:Q1_ANS = Join-Path $root '_smoke_ans'
New-Item -ItemType Directory -Force -Path $env:Q1_OUT | Out-Null
New-Item -ItemType Directory -Force -Path $env:Q1_ANS | Out-Null
Copy-Item (Join-Path $q1 '*') $env:Q1_OUT -Force -Recurse
$env:FOLDS = '2'
$env:TREE_SCALE = '0.02'
$env:NJOB = '1'
foreach ($s in '06b_members.py', '06b2_inner_base.py', '06c_lsf.py', '06d_zits.py', '06e_final.py', '08_answers.py') {
    Step "冒烟 $s" $s
}
Write-Host '冒烟通过' -ForegroundColor Green

# ---------- 3 删除旧产物 ----------
Write-Host '=== 3 删除旧产物 ===' -ForegroundColor Yellow
foreach ($pat in '06*', '07*', '08*', 'fig07*', 'fig08*') {
    $hit = @(Get-ChildItem $q1 -Filter $pat -File -ErrorAction SilentlyContinue)
    if ($hit.Count -gt 0) { $hit | Remove-Item -Force; Write-Host ("  删除 " + $hit.Count + " 个 " + $pat) }
}
foreach ($pat in '问题1_*.csv', '问题1_说明.md', 'fig08_*.png') {
    $hit = @(Get-ChildItem $ans -Filter $pat -File -ErrorAction SilentlyContinue)
    if ($hit.Count -gt 0) { $hit | Remove-Item -Force; Write-Host ("  删除答案 " + $hit.Count + " 个 " + $pat) }
}
Write-Host '剩余输入产物：'
Get-ChildItem $q1 -File | Sort-Object Name | ForEach-Object { Write-Host ('  ' + $_.Name) }

# ---------- 4 多核正式重跑 ----------
Write-Host '=== 4 多核正式重跑 FOLDS=10 TREE_SCALE=1 NJOB=-1 ===' -ForegroundColor Yellow
Remove-Item Env:Q1_OUT -ErrorAction SilentlyContinue
Remove-Item Env:Q1_ANS -ErrorAction SilentlyContinue
$env:FOLDS = '10'
$env:TREE_SCALE = '1.0'
$env:NJOB = '-1'
$t_all = Get-Date
foreach ($s in '06b_members.py', '06b2_inner_base.py', '06c_lsf.py', '06d_zits.py', '06e_final.py', '07_ablation.py', '08_answers.py') {
    Step "正式 $s" $s
}
Write-Host ("=== 全部完成，总用时 {0:N0}s ===" -f ((Get-Date) - $t_all).TotalSeconds) -ForegroundColor Green

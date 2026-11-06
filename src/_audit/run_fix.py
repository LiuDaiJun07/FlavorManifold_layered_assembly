# -*- coding: utf-8 -*-
# 修复后的执行驱动：备份旧产物 -> 冒烟 -> 删旧 -> 多核正式重跑
# 用 Python 而不是 .ps1：本机 pwsh 是 5.1，读无 BOM 的 UTF-8 .ps1 会把中文变成乱码而解析失败
import glob
import os
import shutil
import subprocess
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
PY = os.path.join(ROOT, 'main', '.venv', 'Scripts', 'python.exe')
SRC = os.path.join(ROOT, 'main', 'src', 'q1')
Q1 = os.path.join(ROOT, 'main', 'Output', 'Q1')
ANS = os.path.join(ROOT, '答案')
BK = os.path.join(ROOT, 'temp', '_old_Q1')
SMOKE = os.path.join(ROOT, '_smoke_q1')
SMOKE_ANS = os.path.join(ROOT, '_smoke_ans')


# 打印一行并立刻刷出去（后台任务只有刷出去才看得到）
def log(s=''):
    print(s, flush=True)


# 用一个临时环境跑一个 q1 脚本，失败直接终止整个流程
def run(file, adds):
    env = os.environ.copy()
    env.update(adds)
    log(f'>>> {file}  env={adds}')
    t0 = time.time()
    r = subprocess.run([PY, '-X', 'utf8', os.path.join(SRC, file)], cwd=ROOT, env=env)
    if r.returncode != 0:
        log(f'!!! {file} 失败，exit {r.returncode}')
        raise SystemExit(1)
    log(f'<<< {file} 用时 {time.time() - t0:.0f}s')


if not os.path.isfile(PY):
    raise SystemExit(f'解释器不存在：{PY}')

# ---------- 1 备份 ----------
log('=== 1 备份旧产物到 temp ===')
if os.path.isdir(BK):
    shutil.rmtree(BK)
os.makedirs(BK)
n = 0
for name in sorted(os.listdir(Q1)):
    s = os.path.join(Q1, name)
    if os.path.isfile(s):
        shutil.copy2(s, os.path.join(BK, name))
        n += 1
sub = os.path.join(BK, '答案')
os.makedirs(sub)
for pat in ('问题1_*.csv', '问题1_*.md', 'fig08_*.png'):
    for s in glob.glob(os.path.join(ANS, pat)):
        shutil.copy2(s, os.path.join(sub, os.path.basename(s)))
        n += 1
log(f'备份完成：{n} 个文件 -> {BK}')
for name in sorted(os.listdir(BK)):
    log('  ' + name)

# ---------- 2 冒烟 ----------
log('=== 2 冒烟 FOLDS=2 TREE_SCALE=0.02 NJOB=1 ===')
SENV = {'Q1_OUT': SMOKE, 'Q1_ANS': SMOKE_ANS, 'FOLDS': '2', 'TREE_SCALE': '0.02', 'NJOB': '1'}
# 上次冒烟已经跑出 06b–06d 时，只为验证下游改动重跑 06e/08，省掉几分钟
resume = os.environ.get('SMOKE_RESUME') == '1' and os.path.exists(os.path.join(SMOKE, '06d_zits.npz'))
if resume:
    log('复用上次冒烟的 06b/06b2/06c/06d 产物，只重跑 06e 与 08')
    smoke_files = ('06e_final.py', '08_answers.py')
else:
    for d in (SMOKE, SMOKE_ANS):
        if os.path.isdir(d):
            shutil.rmtree(d)
        os.makedirs(d)
    for name in sorted(os.listdir(Q1)):
        s = os.path.join(Q1, name)
        if os.path.isfile(s):
            shutil.copy2(s, os.path.join(SMOKE, name))
    smoke_files = ('06b_members.py', '06b2_inner_base.py', '06c_lsf.py', '06d_zits.py',
                   '06e_final.py', '08_answers.py')
for f in smoke_files:
    run(f, SENV)
log('冒烟通过')

# ---------- 3 删除旧产物 ----------
log('=== 3 删除旧产物 ===')
for pat in ('06*', '07*', '08*', 'fig07*', 'fig08*'):
    hit = sorted(glob.glob(os.path.join(Q1, pat)))
    for s in hit:
        if os.path.isfile(s):
            os.remove(s)
    if hit:
        log(f'  删除 {len(hit)} 个 {pat}')
for pat in ('问题1_*.csv', '问题1_说明.md', 'fig08_*.png'):
    hit = sorted(glob.glob(os.path.join(ANS, pat)))
    for s in hit:
        os.remove(s)
    if hit:
        log(f'  删除答案 {len(hit)} 个 {pat}')
log('剩余输入产物：')
for name in sorted(os.listdir(Q1)):
    if os.path.isfile(os.path.join(Q1, name)):
        log('  ' + name)

# ---------- 4 多核正式重跑 ----------
log('=== 4 正式 FOLDS=10 TREE_SCALE=1.0 NJOB=-1 ===')
FENV = {'FOLDS': '10', 'TREE_SCALE': '1.0', 'NJOB': '-1'}
t_all = time.time()
for f in ('06b_members.py', '06b2_inner_base.py', '06c_lsf.py', '06d_zits.py',
          '06e_final.py', '07_ablation.py', '08_answers.py'):
    run(f, FENV)
log(f'=== 全部完成，总用时 {time.time() - t_all:.0f}s ===')

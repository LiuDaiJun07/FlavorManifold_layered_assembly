# -*- coding: utf-8 -*-
# probe9 漏洞.md 三条指控的实测核对（只读产物 + 单折计时，不改任何输出）
# ① M4/M5 hist 是否缺、hist_mix 行和是否 0.6
# ② ZITS 的「零组」定义：lab=(y<=zmin) 命中的行数 vs 真实 0 级行数
# ③ zits_hist 是否恒为 [1,0,0,0,0,0,0]
# ④ 单折五成员拟合耗时（用于估算 06b2 内层基座的成本）
import importlib
import os
import sys
import time

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor

HERE = os.path.dirname(os.path.abspath(__file__))
Q1 = os.path.join(os.path.dirname(HERE), 'q1')
sys.path.insert(0, Q1)
FE = importlib.import_module('03_features')
from common import (OUT, SEED, TARGET_COLS, calib_params, load_chem, load_sensory,
                    load_sensory_grade, score_to_grade, z_to_score)

Z = np.load(os.path.join(OUT, '06b_members.npz'), allow_pickle=False)
Y = Z['y']
G = Z['g']
fold_id = Z['fold_id']

print('=== ① 直方图完备性 ===')
for k in ['M1', 'M2', 'M3', 'M4', 'M5']:
    h = Z[f'{k}_hist']
    s = h.sum(axis=2)
    print(f'  {k}_hist 行和 min={s.min():.4f} max={s.max():.4f} 全零比例={np.mean(s < 1e-12):.4f}')
mix = np.mean([Z[f'{k}_hist'] for k in ['M1', 'M2', 'M3', 'M4', 'M5']], axis=0)
print(f'  hist_mix 行和 min={mix.sum(axis=2).min():.6f} max={mix.sum(axis=2).max():.6f}')

print()
print('=== ② ZITS 零组定义 ===')
mu_all, sg_all = calib_params(Y, 'sd')
zero_mean = np.mean([Z[f'zero_{f}'] for f in range(int(fold_id.max()) + 1)], axis=0)
hit = np.where(zero_mean >= 0.40)[0]
print(f'  零占比≥0.40 的维数 = {len(hit)}')
print('  维度 | 0级行数 | 等于zmin行数 | z的不同取值数 | 内层判定(lab==0)=正样本行数')
for j in hit[:8]:
    z = Y[:, j]
    zmin = float(z.min())
    n_zero = int((G[:, j] == 0).sum())
    n_min = int((z <= zmin + 1e-9).sum())
    n_uniq = int(len(np.unique(z)))
    print(f'  {TARGET_COLS[j]:18s} | {n_zero:3d} | {n_min:3d} | {n_uniq:3d} | {len(z) - n_min:3d}')


# 复刻 zits_hist，观察是否退化
def zits_hist_old(T, p1):
    n = T.shape[0]
    ge = np.ones((n, 7))
    for k in range(1, 7):
        ge[:, k] = np.clip(p1 * np.mean(T >= (k - 0.5), axis=1), 0.0, 1.0)
    ge = np.minimum.accumulate(ge, axis=1)
    hist = np.diff(np.column_stack([np.zeros(n), ge]), axis=1)
    return np.clip(hist, 0.0, 1.0)


print()
print('=== ③ zits_hist 退化复现（用真实量级的 z 与 p1） ===')
rs = np.random.RandomState(0)
T_demo = rs.normal(-0.5, 0.7, size=(200, 150))
p1_demo = np.full(200, 0.55)
h = zits_hist_old(T_demo, p1_demo)
print(f'  首行 hist = {np.round(h[0], 4)} 行和={h[0].sum():.4f}')
print(f'  全部行都等于首行 = {bool(np.allclose(h, h[0]))}')
mg = np.argmax(np.cumsum(h, axis=1) >= 0.5, axis=1)
print(f'  中位数等级取值 = {np.unique(mg, return_counts=True)}')

print()
print('=== ④ 单折五成员拟合耗时 ===')
N_EST, N_EST_DIM, MF1, MF2, PCA_T, TOP_K = 300, 200, 0.5, 0.8, 30, 60
chem = load_chem()
sens = load_sensory()
style = sens['tasting_category_fine'].astype(str).to_numpy()
from sklearn.model_selection import KFold
tr = np.arange(0, 225)
te = np.arange(225, 250)
keep, _ = FE.prune_cols(chem, rows=tr)
Xtr, Xte, _ = FE.build(chem, style, tr, te, blocks=('chem', 'miss'), feat_cols=keep)
n_chem = len(keep)
print(f'  n_feat={Xtr.shape[1]}')
t0 = time.time()
m1 = ExtraTreesRegressor(n_estimators=N_EST, max_features=MF1, min_samples_leaf=1,
                         random_state=SEED, n_jobs=1).fit(Xtr, Y[tr])
t1 = time.time()
print(f'  M1（300 棵多输出）={t1 - t0:.1f}s')
Xc = Xtr[:, :n_chem]
t2 = time.time()
for j in range(3):
    ExtraTreesRegressor(n_estimators=N_EST_DIM, max_features=MF1, min_samples_leaf=1,
                        random_state=SEED + j, n_jobs=1).fit(Xc[:, :TOP_K], Y[tr, j])
t3 = time.time()
print(f'  M4 单维（200 棵）={(t3 - t2) / 3:.2f}s → 50 维约 {(t3 - t2) / 3 * 50:.0f}s')
print(f'  单折五成员估计 ≈ {3 * (t1 - t0) + 2 * (t3 - t2) / 3 * 50:.0f}s')

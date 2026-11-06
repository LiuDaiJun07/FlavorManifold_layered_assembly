# -*- coding: utf-8 -*-
# 临时调参脚本：在分层 5 折交叉验证下比较特征表示与 ExtraTrees 超参数
# 用折外拼接预测计算宏平均 R2 与等级 MAE，比单次留出稳定
import importlib
import itertools
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
FE = importlib.import_module('03_features')
from common import CORE_COLS, OUT, SEED, TARGET_COLS, load_chem, load_sensory

# 理化列与化合物列的划分，用于对数比变换与分块消融
PHYS_COLS = ['SO2.mg.L.', 'acetic_acid.g.L.', 'ammonia.mg.L.', 'color.EBC.', 'betaglucan.mg.L.',
             'lactic_acid.mg.L.', 'glycerol', 'iron', 'pH', 'protein.g.L.', 'bitterness', 'GFS',
             'ethanol..v.v.', 'CO2.PSI.', 'unfermentables', 'kcalperc']

chem = load_chem()
sens = load_sensory()
style = sens['tasting_category_fine'].to_numpy()
y = sens[TARGET_COLS].to_numpy(float)
M = pd.read_csv(os.path.join(OUT, '02_destandardize.csv'))
ZMIN = M['z_min'].to_numpy()
G = pd.read_csv(os.path.join(OUT, '02_target_grade.csv'))[TARGET_COLS].to_numpy(int)
core_idx = [TARGET_COLS.index(c) for c in CORE_COLS]

cols = [c for c in chem.columns if c not in ('beer', 'beer_id', 'tasting_category_fine')]
COMP_COLS = [c for c in cols if c not in PHYS_COLS]
ci = [cols.index(c) for c in COMP_COLS]
Xraw = chem[cols].to_numpy(float)


# 哨兵填充：缺失用比最小值再低 10 个量程的常数占位
def fill(X):
    X = X.copy()
    for j in range(X.shape[1]):
        col = X[:, j]
        if np.isnan(col).any():
            mn, mx = np.nanmin(col), np.nanmax(col)
            X[np.isnan(col), j] = mn - 10.0 * (mx - mn + 1.0)
    return X


Xf = fill(Xraw)


# 中心化对数比：对化合物子集逐样本减去其均值，消除总体浓度这一共同因子
def clr(X):
    Xc = X[:, ci]
    return Xc - Xc.mean(axis=1, keepdims=True)


# 宏平均 R2
def macro_r2(yt, yp):
    out = []
    for j in range(yt.shape[1]):
        ss_res = np.sum((yt[:, j] - yp[:, j]) ** 2)
        ss_tot = np.sum((yt[:, j] - yt[:, j].mean()) ** 2)
        out.append(1.0 - ss_res / max(ss_tot, 1e-12))
    return np.array(out)


# 交叉验证：返回折外预测的标准化值与各项指标
def cv_eval(X, mf, msl, md, n_est=300, n_pca=None):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    pred = np.zeros_like(y)
    for tr, te in skf.split(X, style):
        A, B = X[tr], X[te]
        if n_pca:
            pca = PCA(n_components=n_pca, random_state=SEED).fit(A)
            A, B = pca.transform(A), pca.transform(B)
        m = ExtraTreesRegressor(n_estimators=n_est, max_features=mf, min_samples_leaf=msl,
                                max_depth=md, n_jobs=1, random_state=SEED)
        m.fit(A, y[tr])
        pred[te] = m.predict(B)
    r2 = macro_r2(y, pred)
    s = np.clip(pred - ZMIN[None, :], 0.0, 6.0)
    g = np.clip(np.round(s), 0, 6).astype(int)
    return r2.mean(), r2[core_idx].mean(), float(np.abs(s - G).mean()), float(np.abs(g - G).mean()), pred


print('化合物列 %d 个，理化列 %d 个' % (len(COMP_COLS), len(PHYS_COLS)))
print()
print('%-26s %8s %8s %8s %8s' % ('config', 'macroR2', 'coreR2', 'MAE_lin', 'MAE_round'))
reps = {
    'log': Xf,
    'log+clr': np.column_stack([Xf, clr(Xf)]),
    'clr': clr(Xf),
}
best = None
for name, X in reps.items():
    for mf in [0.1, 0.3, 0.5]:
        t0 = time.time()
        r2, cr2, ml, mr, _ = cv_eval(X, mf, 1, None)
        print('%-26s %8.4f %8.4f %8.4f %8.4f  (%.0fs)'
              % ('%s mf=%.1f' % (name, mf), r2, cr2, ml, mr, time.time() - t0))
        if best is None or r2 > best[0]:
            best = (r2, '%s mf=%.1f' % (name, mf), X, mf)
print()
print('最优表示：', best[1], 'macroR2=%.4f' % best[0])
Xbest = best[2]
for mf, msl, md in itertools.product([0.3, 0.5], [1, 3], [None, 12]):
    t0 = time.time()
    r2, cr2, ml, mr, _ = cv_eval(Xbest, mf, msl, md)
    print('%-26s %8.4f %8.4f %8.4f %8.4f  (%.0fs)'
          % ('mf=%.1f msl=%d md=%s' % (mf, msl, md), r2, cr2, ml, mr, time.time() - t0))
print()
for n_pca in [30, 60, 100]:
    t0 = time.time()
    r2, cr2, ml, mr, _ = cv_eval(Xbest, 0.5, 1, None, n_pca=n_pca)
    print('%-26s %8.4f %8.4f %8.4f %8.4f  (%.0fs)'
          % ('pca%d mf=1.0' % n_pca, r2, cr2, ml, mr, time.time() - t0))

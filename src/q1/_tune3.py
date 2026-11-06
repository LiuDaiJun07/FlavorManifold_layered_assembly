# -*- coding: utf-8 -*-
# 临时调参脚本第三轮：样本级聚合特征、风格 one-hot、近邻目标编码（stacking 的单层近似）
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import kurtosis
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CORE_COLS, OUT, SEED, TARGET_COLS, load_chem, load_sensory

PHYS_COLS = ['SO2.mg.L.', 'acetic_acid.g.L.', 'ammonia.mg.L.', 'color.EBC.', 'betaglucan.mg.L.',
             'lactic_acid.mg.L.', 'glycerol', 'iron', 'pH', 'protein.g.L.', 'bitterness', 'GFS',
             'ethanol..v.v.', 'CO2.PSI.', 'unfermentables', 'kcalperc']
SUM_COLS = ['esters_sum', 'aroma_hops_sum', 'acids_sum', 'sulfur_sum']

chem = load_chem()
sens = load_sensory()
style = sens['tasting_category_fine'].to_numpy()
y = sens[TARGET_COLS].to_numpy(float)
M = pd.read_csv(os.path.join(OUT, '02_destandardize.csv'))
ZMIN = M['z_min'].to_numpy()
G = pd.read_csv(os.path.join(OUT, '02_target_grade.csv'))[TARGET_COLS].to_numpy(int)
core_idx = [TARGET_COLS.index(c) for c in CORE_COLS]
cols = [c for c in chem.columns if c not in ('beer', 'beer_id', 'tasting_category_fine')]
X0 = chem[cols].to_numpy(float)


def fill(X):
    X = X.copy()
    for j in range(X.shape[1]):
        col = X[:, j]
        if np.isnan(col).any():
            mn, mx = np.nanmin(col), np.nanmax(col)
            X[np.isnan(col), j] = mn - 10.0 * (mx - mn + 1.0)
    return X


def macro_r2(yt, yp):
    out = []
    for j in range(yt.shape[1]):
        ss_res = np.sum((yt[:, j] - yp[:, j]) ** 2)
        ss_tot = np.sum((yt[:, j] - yt[:, j].mean()) ** 2)
        out.append(1.0 - ss_res / max(ss_tot, 1e-12))
    return np.array(out)


def to_grade(z):
    return np.clip(np.round(np.clip(z - ZMIN[None, :], 0.0, 6.0)), 0, 6).astype(int)


def expect_grade(model, Xte):
    hist = np.zeros((Xte.shape[0], len(TARGET_COLS), 7))
    for est in model.estimators_:
        g = to_grade(est.predict(Xte))
        for k in range(7):
            hist[:, :, k] += (g == k)
    hist /= np.maximum(hist.sum(axis=2, keepdims=True), 1)
    exp = np.tensordot(hist, np.arange(7), axes=([2], [0]))
    return np.clip(np.round(exp), 0, 6).astype(int)


# 样本级聚合：反映总体香气负载强度与离散度
def samp_agg(X):
    Xi = X[:, [cols.index(c) for c in cols if c not in PHYS_COLS and c not in SUM_COLS]]
    s = np.sort(Xi, axis=1)
    return np.column_stack([Xi.mean(1), Xi.std(1), np.median(Xi, 1), Xi.max(1), Xi.min(1),
                            (Xi > 0).sum(1).astype(float), s[:, -10:].mean(1),
                            kurtosis(Xi, axis=1, bias=False)])


# 近邻目标编码：训练行用留一法近邻均值，测试行用全部训练行近邻均值
def knn_target(Ztr, Zte, ytr, k=15):
    mu, sd = Ztr.mean(0), Ztr.std(0)
    mu[sd == 0] = 0
    sd[sd == 0] = 1
    A = (Ztr - mu) / sd
    B = (Zte - mu) / sd
    Dtt = ((A[:, None, :] - A[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(Dtt, np.inf)
    idx_tr = np.argsort(Dtt, axis=1)[:, :k]
    Ttr = ytr[idx_tr].mean(1)
    Dte = ((B[:, None, :] - A[None, :, :]) ** 2).sum(-1)
    idx_te = np.argsort(Dte, axis=1)[:, :k]
    Tte = ytr[idx_te].mean(1)
    return Ttr, Tte


def run(Xcols, use_agg=False, use_style=False, use_knn=0, mf=0.5, n_est=300, msl=1):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    pred = np.zeros_like(y)
    ge = np.zeros_like(G)
    AGG = samp_agg(fill(X0))
    for tr, te in skf.split(Xcols, style):
        A, B = Xcols[tr], Xcols[te]
        if use_knn:
            Ttr, Tte = knn_target(A, B, y[tr], k=use_knn)
            A = np.column_stack([A, Ttr])
            B = np.column_stack([B, Tte])
        if use_agg:
            A = np.column_stack([A, AGG[tr]])
            B = np.column_stack([B, AGG[te]])
        if use_style:
            cats = sorted(pd.unique(style[tr]))
            A = np.column_stack([A, np.array([[1.0 if s == c else 0.0 for c in cats] for s in style[tr]])])
            B = np.column_stack([B, np.array([[1.0 if s == c else 0.0 for c in cats] for s in style[te]])])
        m = ExtraTreesRegressor(n_estimators=n_est, max_features=mf, min_samples_leaf=msl,
                                n_jobs=1, random_state=SEED).fit(A, y[tr])
        pred[te] = m.predict(B)
        ge[te] = expect_grade(m, B)
    r2 = macro_r2(y, pred)
    return dict(R2=r2.mean(), core=r2[core_idx].mean(),
                lin=float(np.abs(np.clip(pred - ZMIN, 0, 6) - G).mean()),
                rnd=float(np.abs(to_grade(pred) - G).mean()),
                exp=float(np.abs(ge - G).mean()))


keep2 = [i for i, c in enumerate(cols)
         if c not in SUM_COLS and pd.Series(X0[:, i]).value_counts(dropna=True).iloc[0] / len(X0) < 0.5]
Xb = fill(X0)[:, keep2]
print('基础列 %d 个' % Xb.shape[1])
print('%-32s %8s %8s %8s %8s %8s' % ('config', 'R2', 'coreR2', 'MAElin', 'MAErnd', 'MAEexp'))
for tag, kw in [('base', {}),
                ('base+agg', dict(use_agg=True)),
                ('base+style', dict(use_style=True)),
                ('base+knn15', dict(use_knn=15)),
                ('base+knn25', dict(use_knn=25)),
                ('base+knn15+style', dict(use_knn=15, use_style=True)),
                ('base+knn15+agg+style', dict(use_knn=15, use_agg=True, use_style=True)),
                ('base+knn30+agg', dict(use_knn=30, use_agg=True))]:
    t0 = time.time()
    d = run(Xb, **kw)
    print('%-32s %8.4f %8.4f %8.4f %8.4f %8.4f (%.0fs)'
          % (tag, d['R2'], d['core'], d['lin'], d['rnd'], d['exp'], time.time() - t0))

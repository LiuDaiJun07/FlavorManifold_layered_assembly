# -*- coding: utf-8 -*-
# 临时调参脚本第四轮：用 2.csv 图做化合物值平滑、模块均值特征，以及目标端的秩高斯变换
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import importlib
FE = importlib.import_module('03_features')
from common import ATT, CORE_COLS, OUT, SEED, TARGET_COLS, load_chem, load_scale, load_sensory, macro_r2

chem = load_chem()
sens = load_sensory()
style = sens['tasting_category_fine'].to_numpy()
Y = sens[TARGET_COLS].to_numpy(float)
G = pd.read_csv(os.path.join(OUT, '02_target_grade.csv'))[TARGET_COLS].to_numpy(int)
MU, SG = load_scale()
core_idx = [TARGET_COLS.index(c) for c in CORE_COLS]
keep, _ = FE.prune_cols(chem)
allc = [c for c in chem.columns if c not in ('beer', 'beer_id', 'tasting_category_fine')]
Xall = chem[allc].to_numpy(float)

R = pd.read_csv(os.path.join(ATT, '2.csv'), index_col=0) if False else None
from common import load_graph_corr
allc, RR = load_graph_corr()
A = np.abs(RR)
np.fill_diagonal(A, 0.0)
# 图平滑算子：自身权重 1，邻居权重按相似度归一化
W = A.copy()
W = W / np.maximum(W.sum(axis=1, keepdims=True), 1e-9)
np.fill_diagonal(W, 1.0)
W = W / W.sum(axis=1, keepdims=True)
ki = [allc.index(c) for c in keep]


def fill(X):
    X = X.copy()
    for j in range(X.shape[1]):
        col = X[:, j]
        if np.isnan(col).any():
            mn, mx = np.nanmin(col), np.nanmax(col)
            X[np.isnan(col), j] = mn - 10.0 * (mx - mn + 1.0)
    return X


Xf = fill(Xall)
XF = Xf[:, ki]
WS = W[ki][:, ki]
GRAPH = XF @ WS.T


def to_grade(z):
    return np.clip(np.round(np.clip(MU[None, :] + SG[None, :] * z, 0, 6)), 0, 6).astype(int)


def exp_grade(model, Xte):
    hist = np.zeros((Xte.shape[0], len(TARGET_COLS), 7))
    for est in model.estimators_:
        g = to_grade(est.predict(Xte))
        for k in range(7):
            hist[:, :, k] += (g == k)
    hist /= np.maximum(hist.sum(axis=2, keepdims=True), 1)
    return to_grade(np.tensordot(hist, np.arange(7, dtype=float), axes=([2], [0])))


# 秩高斯变换：把每维目标换成训练折上的正态分位数，拟合后再反变换
def rank_gauss(ytr, yte):
    out_tr = np.zeros_like(ytr)
    out_te = np.zeros_like(yte)
    for j in range(ytr.shape[1]):
        v = ytr[:, j]
        r = rankdata(v) / (len(v) + 1.0)
        out_tr[:, j] = norm.ppf(r)
        srt = np.sort(v)
        out_te[:, j] = srt[np.clip((rankdata(yte[:, j]) - 1).astype(int), 0, len(srt) - 1)]
    return out_tr, out_te


def run(Xcols, use_style=False, rg=False, mf=0.5, n_est=300):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    pred = np.zeros_like(Y)
    ge = np.zeros_like(G)
    for tr, te in skf.split(Xcols, style):
        Atr, Btr = Xcols[tr], Xcols[te]
        if use_style:
            cats = sorted(pd.unique(style[tr]))
            Atr = np.column_stack([Atr, np.array([[1.0 if s == c else 0.0 for c in cats] for s in style[tr]])])
            Btr = np.column_stack([Btr, np.array([[1.0 if s == c else 0.0 for c in cats] for s in style[te]])])
        if rg:
            yr, _ = rank_gauss(Y[tr], Y[tr])
            m = ExtraTreesRegressor(n_estimators=n_est, max_features=mf, n_jobs=1, random_state=SEED).fit(Atr, yr)
            pr = m.predict(Btr)
            for j in range(len(TARGET_COLS)):
                srt = np.sort(Y[tr, j])
                q = np.clip(norm.cdf(pr[:, j]) * (len(srt) - 1), 0, len(srt) - 1)
                lo = np.floor(q).astype(int)
                hi = np.clip(lo + 1, 0, len(srt) - 1)
                w = q - lo
                pred[te, j] = srt[lo] * (1 - w) + srt[hi] * w
        else:
            m = ExtraTreesRegressor(n_estimators=n_est, max_features=mf, n_jobs=1, random_state=SEED).fit(Atr, Y[tr])
            pred[te] = m.predict(Btr)
        ge[te] = exp_grade(m, Btr)
    r2 = macro_r2(Y, pred)
    return dict(R2=r2.mean(), core=r2[core_idx].mean(),
                lin=float(np.abs(np.clip(MU + SG * pred, 0, 6) - G).mean()),
                rnd=float(np.abs(to_grade(pred) - G).mean()),
                exp=float(np.abs(ge - G).mean()))


Xs = np.column_stack([XF, GRAPH])
print('%-28s %8s %8s %8s %8s %8s' % ('config', 'R2', 'coreR2', 'MAElin', 'MAErnd', 'MAEexp'))
for tag, X, kw in [('base', XF, {}),
                   ('base+style', XF, dict(use_style=True)),
                   ('base+graph', Xs, {}),
                   ('base+graph+style', Xs, dict(use_style=True)),
                   ('base+rankgauss', XF, dict(rg=True)),
                   ('base+style+rankgauss', XF, dict(use_style=True, rg=True))]:
    t0 = time.time()
    d = run(X, **kw)
    print('%-28s %8.4f %8.4f %8.4f %8.4f %8.4f (%.0fs)'
          % (tag, d['R2'], d['core'], d['lin'], d['rnd'], d['exp'], time.time() - t0))

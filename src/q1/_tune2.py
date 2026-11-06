# -*- coding: utf-8 -*-
# 临时调参脚本第二轮：比较多输出与逐维建模、等级还原方式（直接取整 vs 森林等级分布期望）、特征裁剪
import os
import sys
import time

import numpy as np
import pandas as pd
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


# 哨兵填充
def fill(X):
    X = X.copy()
    for j in range(X.shape[1]):
        col = X[:, j]
        if np.isnan(col).any():
            mn, mx = np.nanmin(col), np.nanmax(col)
            X[np.isnan(col), j] = mn - 10.0 * (mx - mn + 1.0)
    return X


# 宏平均 R2
def macro_r2(yt, yp):
    out = []
    for j in range(yt.shape[1]):
        ss_res = np.sum((yt[:, j] - yp[:, j]) ** 2)
        ss_tot = np.sum((yt[:, j] - yt[:, j].mean()) ** 2)
        out.append(1.0 - ss_res / max(ss_tot, 1e-12))
    return np.array(out)


# 由标准化值还原到 0-6 并取整
def to_grade(z):
    return np.clip(np.round(np.clip(z - ZMIN[None, :], 0.0, 6.0)), 0, 6).astype(int)


# 森林等级经验分布期望：逐棵树还原等级后取期望，再四舍五入
def expect_grade(models, Xte, n_out):
    hist = np.zeros((Xte.shape[0], n_out, 7))
    for m in models:
        ests = m.estimators_ if hasattr(m, 'estimators_') else [m]
        for est in ests:
            pz = est.predict(Xte)
            if pz.ndim == 1:
                pz = pz[:, None]
            g = to_grade(pz)
            for k in range(7):
                hist[:, :, k] += (g == k)
    hist /= np.maximum(hist.sum(axis=2, keepdims=True), 1)
    exp = np.tensordot(hist, np.arange(7), axes=([2], [0]))
    return np.clip(np.round(exp), 0, 6).astype(int)


# 一次完整交叉验证，mode=multi 或 dim
def run(X, mf=0.5, mode='multi', n_est=300, msl=1):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    pred = np.zeros_like(y)
    pred2 = np.zeros_like(y)
    ge = np.zeros_like(G)
    for tr, te in skf.split(X, style):
        A, B = X[tr], X[te]
        if mode == 'multi':
            m = ExtraTreesRegressor(n_estimators=n_est, max_features=mf, min_samples_leaf=msl,
                                    n_jobs=1, random_state=SEED).fit(A, y[tr])
            pred[te] = m.predict(B)
            ge[te] = expect_grade([m], B, len(TARGET_COLS))
        else:
            models = []
            for j in range(len(TARGET_COLS)):
                m = ExtraTreesRegressor(n_estimators=n_est, max_features=mf, min_samples_leaf=msl,
                                        n_jobs=1, random_state=SEED).fit(A, y[tr, j])
                pred[te, j] = m.predict(B)
                models.append(m)
            ge[te] = expect_grade(models, B, len(TARGET_COLS))
    r2 = macro_r2(y, pred)
    gr = to_grade(pred)
    return dict(macroR2=r2.mean(), coreR2=r2[core_idx].mean(),
                MAE_lin=float(np.abs(np.clip(pred - ZMIN, 0, 6) - G).mean()),
                MAE_round=float(np.abs(gr - G).mean()),
                MAE_exp=float(np.abs(ge - G).mean()),
                core_exp=float(np.abs(ge[:, core_idx] - G[:, core_idx]).mean()))


def show(tag, d, dt):
    print('%-30s R2=%.4f coreR2=%.4f | MAE_lin=%.4f round=%.4f exp=%.4f coreExp=%.4f (%.0fs)'
          % (tag, d['macroR2'], d['coreR2'], d['MAE_lin'], d['MAE_round'], d['MAE_exp'], d['core_exp'], dt))


Xf = fill(X0)
print('%-30s %s' % ('config', 'metrics'))
t0 = time.time(); show('multi mf=0.5', run(Xf), time.time() - t0)

# 特征裁剪：去掉 4 个汇总列
keep = [i for i, c in enumerate(cols) if c not in SUM_COLS]
t0 = time.time(); show('multi mf=0.5 -sum', run(Xf[:, keep]), time.time() - t0)

# 特征裁剪：去掉众数占比超过 5 成的列，这些列信息量几乎为零
keep2 = [i for i, c in enumerate(cols)
         if c not in SUM_COLS and pd.Series(X0[:, i]).value_counts(dropna=True).iloc[0] / len(X0) < 0.5]
print('众数占比<0.5 保留 %d 列' % len(keep2))
t0 = time.time(); show('multi mf=0.5 -sum -mode', run(Xf[:, keep2]), time.time() - t0)

t0 = time.time(); show('dim  mf=0.3', run(Xf, mf=0.3, mode='dim', n_est=200), time.time() - t0)

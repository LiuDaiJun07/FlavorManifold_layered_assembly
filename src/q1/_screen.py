# -*- coding: utf-8 -*-
# 临时筛选脚本：用单次分层留出快速比较各特征块与几何设定对 R2 / MAE 的影响
import importlib
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
F = importlib.import_module('03_features')
from common import CORE_COLS, OUT, SEED, TARGET_COLS, load_chem, load_sensory

chem = load_chem()
sens = load_sensory()
style = sens['tasting_category_fine'].to_numpy()
y = sens[TARGET_COLS].to_numpy(float)
G = pd.read_csv(os.path.join(OUT, '02_target_grade.csv'))[TARGET_COLS].to_numpy(int)
M = pd.read_csv(os.path.join(OUT, '02_destandardize.csv'))
MU = M['mu'].to_numpy()
SG = M['sigma'].to_numpy()

idx = np.arange(len(sens))
tr, te = train_test_split(idx, test_size=0.30, random_state=SEED, stratify=style)


# 由标准化预测值还原成 0-6 连续分
def to_score(z):
    return np.clip(MU[None, :] + SG[None, :] * z, 0.0, 6.0)


# 逐棵树还原等级并取期望，等价于用森林的等级经验分布求期望
def expect_grade(model, Xte):
    n = Xte.shape[0]
    hist = np.zeros((n, len(TARGET_COLS), 7))
    for est in model.estimators_:
        g = np.clip(np.round(to_score(est.predict(Xte))), 0, 6).astype(int)
        for k in range(7):
            hist[:, :, k] += (g == k)
    hist /= len(model.estimators_)
    exp = np.tensordot(hist, np.arange(7), axes=([2], [0]))
    return np.clip(np.round(exp), 0, 6).astype(int), exp


# 宏平均 R2，逐维算后取均值
def macro_r2(yt, yp):
    out = []
    for j in range(yt.shape[1]):
        ss_res = np.sum((yt[:, j] - yp[:, j]) ** 2)
        ss_tot = np.sum((yt[:, j] - yt[:, j].mean()) ** 2)
        out.append(1.0 - ss_res / max(ss_tot, 1e-12))
    return np.array(out)


core_idx = [TARGET_COLS.index(c) for c in CORE_COLS]
configs = [
    ('E0 chem+miss', ('chem', 'miss'), 1.0),
    ('E1 +style', ('chem', 'miss', 'style'), 1.0),
    ('E2 +hyp', ('chem', 'miss', 'style', 'hyp'), 1.0),
    ('E5 +hyp c->0', ('chem', 'miss', 'style', 'hyp'), 1e-8),
]
rows = []
for name, blocks, c in configs:
    t0 = time.time()
    Xtr, Xte, names = F.build(chem, style, tr, te, blocks=blocks, c=c)
    m = ExtraTreesRegressor(n_estimators=300, max_features=0.5, n_jobs=1, random_state=SEED)
    m.fit(Xtr, y[tr])
    zp = m.predict(Xte)
    r2 = macro_r2(y[te], zp)
    grade_round = np.clip(np.round(to_score(zp)), 0, 6).astype(int)
    ge, gexp = expect_grade(m, Xte)
    mae_r = float(np.abs(grade_round - G[te]).mean())
    mae_e = float(np.abs(ge - G[te]).mean())
    mae_core = float(np.abs(ge[:, core_idx] - G[te][:, core_idx]).mean())
    mae_lin = float(np.abs(to_score(zp) - G[te]).mean())
    print('%-14s p=%3d  %.1fs  macroR2=%.4f coreR2=%.4f | MAE_lin=%.4f MAE_round=%.4f MAE_exp=%.4f coreMAE=%.4f'
          % (name, Xtr.shape[1], time.time() - t0, r2.mean(), r2[core_idx].mean(),
             mae_lin, mae_r, mae_e, mae_core))
    rows.append(dict(config=name, p=Xtr.shape[1], macroR2=r2.mean(),
                     coreR2=r2[core_idx].mean(), mae_lin=mae_lin, mae_round=mae_r,
                     mae_exp=mae_e, core_mae=mae_core))
print('SEED=%d' % SEED)

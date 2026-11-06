# -*- coding: utf-8 -*-
# 审查用探针 2：换模型类别能不能突破 0.25
# 依次测量：逐维梯度提升 / 逐维序数分类取期望 / 目标链式（用第一阶段预测当特征）/ 与 ET 融合
import importlib
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.model_selection import StratifiedKFold

HERE = os.path.dirname(os.path.abspath(__file__))
Q1 = os.path.join(os.path.dirname(HERE), 'q1')
sys.path.insert(0, Q1)
FE = importlib.import_module('03_features')
from common import CORE_COLS, OUT, SEED, TARGET_COLS, load_chem, load_sensory, macro_r2

chem = load_chem()
sens = load_sensory()
style = sens['tasting_category_fine'].to_numpy()
Y = sens[TARGET_COLS].to_numpy(float)
G = pd.read_csv(os.path.join(OUT, '02_target_grade.csv'))[TARGET_COLS].to_numpy(int)
MU = np.array([0.0] * len(TARGET_COLS))
keep, _ = FE.prune_cols(chem)
core_idx = [TARGET_COLS.index(c) for c in CORE_COLS]
n, m = Y.shape
folds = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED).split(np.zeros(n), style))
RES = {}
PRED = {}
CACHE = [(tr, te) + tuple(FE.build(chem, style, tr, te, blocks=('chem', 'miss'), feat_cols=keep)[:2])
         for tr, te in folds]
print(f'缓存 {CACHE[0][2].shape[1]} 列', flush=True)


def score(tag, P):
    r2 = macro_r2(Y, P)
    RES[tag] = dict(macro_r2=float(r2.mean()), core_r2=float(r2[core_idx].mean()),
                    extra_r2=float(np.delete(r2, core_idx).mean()),
                    mae_z=float(np.abs(P - Y).mean()), n_dim_bad=int((r2 < 0.10).sum()),
                    r2_dim=r2.tolist())
    PRED[tag] = P.copy()
    print(f'{tag:28s} R2={RES[tag]["macro_r2"]:.4f} core={RES[tag]["core_r2"]:.4f} '
          f'MAEz={RES[tag]["mae_z"]:.4f} dim<0.10={RES[tag]["n_dim_bad"]}', flush=True)


# 逐维梯度提升：EXACT 版，200 行上单线程可接受
def gbr_per_dim(n_est=300, lr=0.05, depth=3):
    def f(A, Ytr, B, tr):
        out = np.zeros((B.shape[0], Ytr.shape[1]))
        for j in range(Ytr.shape[1]):
            m = GradientBoostingRegressor(n_estimators=n_est, learning_rate=lr, max_depth=depth,
                                          subsample=0.8, random_state=SEED).fit(A, Ytr[:, j])
            out[:, j] = m.predict(B)
        return out
    return f


# 逐维序数分类：把 0-6 等级当 7 类，取概率期望，再换回 z 空间用于 R2
def ord_per_dim(n_est=300, mf=0.5):
    def f(A, Ytr, B, tr):
        out = np.zeros((B.shape[0], Ytr.shape[1]))
        for j in range(Ytr.shape[1]):
            y = G[tr, j]
            m = ExtraTreesClassifier(n_estimators=n_est, max_features=mf, min_samples_leaf=1,
                                     n_jobs=1, random_state=SEED).fit(A, y)
            p = m.predict_proba(B)
            cls = m.classes_.astype(float)
            out[:, j] = p @ cls
        return out
    return f


# 目标链式：外层折内再做 5 折，得到无泄漏的第一阶段折外预测，把它当作第二阶段的新特征
def chain(n_est=300, mf=0.5):
    def f(A, Ytr, B, tr):
        inner = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED + 7).split(A, style[tr]))
        S1 = np.zeros_like(Ytr)
        for itr, ite in inner:
            mi = ExtraTreesRegressor(n_estimators=n_est, max_features=mf, min_samples_leaf=1,
                                    n_jobs=1, random_state=SEED).fit(A[itr], Ytr[itr])
            S1[ite] = mi.predict(A[ite])
        m1 = ExtraTreesRegressor(n_estimators=n_est, max_features=mf, min_samples_leaf=1,
                                 n_jobs=1, random_state=SEED).fit(A, Ytr)
        S1t = m1.predict(B)
        A2 = np.hstack([A, S1])
        B2 = np.hstack([B, S1t])
        out = np.zeros((B.shape[0], Ytr.shape[1]))
        for j in range(Ytr.shape[1]):
            m2 = ExtraTreesRegressor(n_estimators=n_est, max_features=mf, min_samples_leaf=1,
                                     n_jobs=1, random_state=SEED).fit(A2, Ytr[:, j])
            out[:, j] = m2.predict(B2)
        return out
    return f


def run_cv(tag, fit_predict):
    t0 = time.time()
    P = np.zeros_like(Y)
    for tr, te, A, B in CACHE:
        P[te] = fit_predict(A, Y[tr], B, tr)
    score(tag, P)
    print(f'  用时 {time.time() - t0:.0f}s', flush=True)


run_cv('B1_gbr_perdim', gbr_per_dim())
run_cv('B2_ordinal_et', ord_per_dim())
run_cv('B3_chain_et', chain())

# 融合：与探针一的 ET 结果按逐维最小二乘（非负）组合，权重用折外再折一次得到，避免自评
P1 = np.load(os.path.join(HERE, 'probe_pred.npz'))
base = {k: P1[k] for k in P1.files}
base.update(PRED)
np.savez_compressed(os.path.join(HERE, 'probe2_pred.npz'), **PRED)


# 逐维非负最小二乘融合，权重在 5 折里交叉拟合
def nnls_fuse(keys, tag):
    from scipy.optimize import nnls
    Pw = np.zeros_like(Y)
    for tr, te, _, _ in CACHE:
        for j in range(Y.shape[1]):
            M = np.column_stack([base[k][:, j] for k in keys])
            M = np.column_stack([M, np.ones(len(M))])
            w, _ = nnls(M[tr], Y[tr, j])
            Pw[te, j] = M[te] @ w
    score(tag, Pw)


for keys in (['A0_multi_raw_mf0.5', 'B1_gbr_perdim'], ['A0_multi_raw_mf0.5', 'B2_ordinal_et'],
             ['B1_gbr_perdim', 'B2_ordinal_et'], ['A0_multi_raw_mf0.5', 'B1_gbr_perdim', 'B2_ordinal_et'],
             ['A0_multi_raw_mf0.5', 'A3_multi_norm_mf0.8', 'A4_pca30_norm', 'B1_gbr_perdim', 'B2_ordinal_et']):
    ok = [k for k in keys if k in base]
    if len(ok) < 2:
        continue
    nnls_fuse(ok, 'C_nnls_' + '+'.join(k.split("_")[0] for k in ok))

with open(os.path.join(HERE, 'probe2.json'), 'w', encoding='utf-8') as f:
    json.dump(RES, f, ensure_ascii=False, indent=1)
print('完成', flush=True)

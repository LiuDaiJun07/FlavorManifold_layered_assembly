# -*- coding: utf-8 -*-
# 审查用探针：只做测量，不改动任何管线产物
# 目的：量化问题一的可达上限，为补丁文档提供实测依据
# 依次测量：基线 / 目标按列归一化 / 目标主成分去噪 / 逐维建模 + 折内特征筛选 / 融合 / A-F 双通道信度
import importlib
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
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
keep, dropped = FE.prune_cols(chem)
core_idx = [TARGET_COLS.index(c) for c in CORE_COLS]
n, m = Y.shape
folds = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED).split(np.zeros(n), style))
RES = {}


# 记录一组折外预测的全部指标
def score(tag, P):
    r2 = macro_r2(Y, P)
    RES[tag] = dict(macro_r2=float(r2.mean()), core_r2=float(r2[core_idx].mean()),
                    extra_r2=float(np.delete(r2, core_idx).mean()),
                    mae_z=float(np.abs(P - Y).mean()),
                    mae_z_core=float(np.abs(P[:, core_idx] - Y[:, core_idx]).mean()),
                    r2_dim=r2.tolist(), n_dim_bad=int((r2 < 0.10).sum()))
    print(f'{tag:28s} R2={RES[tag]["macro_r2"]:.4f} core={RES[tag]["core_r2"]:.4f} '
          f'MAEz={RES[tag]["mae_z"]:.4f} dim<0.10={RES[tag]["n_dim_bad"]}', flush=True)


# 每一折的特征矩阵先算一次并缓存，所有实验复用
CACHE = []
t0 = time.time()
for tr, te in folds:
    A, B, names = FE.build(chem, style, tr, te, blocks=('chem', 'miss'), feat_cols=keep)
    CACHE.append((tr, te, A, B))
print(f'特征缓存完成 {len(CACHE)} 折，{CACHE[0][2].shape[1]} 列，用时 {time.time() - t0:.0f}s', flush=True)


PRED = {}


# 通用折外框架：fit_fn(A, Ytr, tr_idx) 返回 predict(B) 的函数
def run_cv(tag, fit_predict):
    P = np.zeros_like(Y)
    for tr, te, A, B in CACHE:
        P[te] = fit_predict(A, Y[tr], B, tr)
    PRED[tag] = P.copy()
    score(tag, P)
    return P


# 多输出极端森林，目标可先按列归一化
def et_multi(mf, norm=False, n_est=300):
    def f(A, Ytr, B, tr):
        mu, sd = Ytr.mean(0), Ytr.std(0, ddof=1)
        sd = np.where(sd < 1e-12, 1.0, sd)
        T = (Ytr - mu) / sd if norm else Ytr
        m = ExtraTreesRegressor(n_estimators=n_est, max_features=mf, min_samples_leaf=1,
                                n_jobs=1, random_state=SEED).fit(A, T)
        Q = m.predict(B)
        return Q * sd + mu if norm else Q
    return f


# 目标主成分去噪：在归一化目标空间取前 k 个主成分建模，再投影回 50 维
def et_pca(k, mf=0.5, n_est=300):
    def f(A, Ytr, B, tr):
        mu, sd = Ytr.mean(0), Ytr.std(0, ddof=1)
        sd = np.where(sd < 1e-12, 1.0, sd)
        Z = (Ytr - mu) / sd
        pca = PCA(n_components=k, random_state=SEED).fit(Z)
        P = pca.transform(Z)
        m = ExtraTreesRegressor(n_estimators=n_est, max_features=mf, min_samples_leaf=1,
                                n_jobs=1, random_state=SEED).fit(A, P)
        return pca.inverse_transform(m.predict(B)) * sd + mu
    return f


# 逐维建模：每维先按折内秩相关筛出 topK 个特征，再单独建模
def et_per_dim(mf, topk=None, n_est=300, est='et'):
    def f(A, Ytr, B, tr):
        out = np.zeros((B.shape[0], Ytr.shape[1]))
        for j in range(Ytr.shape[1]):
            y = Ytr[:, j]
            if topk is not None and topk < A.shape[1]:
                col = np.array([abs(spearmanr(A[:, i], y).statistic) for i in range(A.shape[1])])
                col = np.nan_to_num(col)
                idx = np.argsort(-col)[:topk]
            else:
                idx = np.arange(A.shape[1])
            if est == 'et':
                m = ExtraTreesRegressor(n_estimators=n_est, max_features=mf, min_samples_leaf=1,
                                        n_jobs=1, random_state=SEED)
            else:
                m = RandomForestRegressor(n_estimators=n_est, max_features=mf, min_samples_leaf=1,
                                          n_jobs=1, random_state=SEED)
            m.fit(A[:, idx], y)
            out[:, j] = m.predict(B[:, idx])
        return out
    return f


# 基线：现有配置的原样复现
run_cv('A0_multi_raw_mf0.5', et_multi(0.5, norm=False))
run_cv('A1_multi_norm_mf0.5', et_multi(0.5, norm=True))
run_cv('A2_multi_norm_mf0.3', et_multi(0.3, norm=True))
run_cv('A3_multi_norm_mf0.8', et_multi(0.8, norm=True))
for k in (10, 20, 30):
    run_cv(f'A4_pca{k}_norm', et_pca(k))
run_cv('A5_perdim_mf0.5_all', et_per_dim(0.5, topk=None))
for K in (40, 60, 100):
    run_cv(f'A6_perdim_mf0.5_top{K}', et_per_dim(0.5, topk=K))
run_cv('A7_perdim_mf1.0_top60', et_per_dim(1.0, topk=60))
run_cv('A8_perdim_rf0.5_top60', et_per_dim(0.5, topk=60, est='rf'))

# 融合：把几组折外预测做等权平均，看线性融合还能挤出多少
groups = {
    'F1_norm_perdim': ['A1_multi_norm_mf0.5', 'A6_perdim_mf0.5_top60'],
    'F2_norm_pca_perdim': ['A1_multi_norm_mf0.5', 'A4_pca20_norm', 'A6_perdim_mf0.5_top60'],
    'F3_all5': ['A1_multi_norm_mf0.5', 'A4_pca20_norm', 'A6_perdim_mf0.5_top60',
                'A7_perdim_mf1.0_top60', 'A8_perdim_rf0.5_top60'],
    'F4_et_rf_perdim': ['A6_perdim_mf0.5_top60', 'A8_perdim_rf0.5_top60'],
}
for g, ks in groups.items():
    ks = [k for k in ks if k in PRED]
    if len(ks) < 2:
        continue
    Pm = np.mean([PRED[k] for k in ks], axis=0)
    score(g + '_mean', Pm)
    # 逐维按各成员折外 R2 加权，权重只取正值
    R2 = np.array([macro_r2(Y, PRED[k]) for k in ks])
    w = np.clip(R2, 1e-3, None)
    w = w / w.sum(0, keepdims=True)
    Pw = np.sum([PRED[k] * w[i][None, :] for i, k in enumerate(ks)], axis=0)
    score(g + '_r2w', Pw)

# 保存中间结果
with open(os.path.join(HERE, 'probe_frontier.json'), 'w', encoding='utf-8') as f:
    json.dump(RES, f, ensure_ascii=False, indent=1)
np.savez_compressed(os.path.join(HERE, 'probe_pred.npz'), **PRED)

# A-F 双通道信度：同名概念的香气维与风味维的相关，可作为单通道可靠度上限
print('\n===== A-F 双通道信度 =====', flush=True)
rows = []
for c in CORE_COLS:
    if not c.startswith('A_'):
        continue
    f = 'F_' + c[2:]
    if f not in TARGET_COLS:
        continue
    a = Y[:, TARGET_COLS.index(c)]
    b = Y[:, TARGET_COLS.index(f)]
    r = float(spearmanr(a, b).statistic)
    rp = float(np.corrcoef(a, b)[0, 1])
    rows.append(dict(concept=c[2:], rho_AF=r, pearson_AF=rp, sd_A=float(a.std(ddof=1)), sd_F=float(b.std(ddof=1))))
af = pd.DataFrame(rows)
print(af.to_string(index=False), flush=True)
print(f'均值 rho={af["rho_AF"].mean():.4f}  pearson={af["pearson_AF"].mean():.4f} '
      f'pearson^2={np.mean(af["pearson_AF"] ** 2):.4f}', flush=True)
af.to_csv(os.path.join(HERE, 'probe_af_reliability.csv'), index=False, encoding='utf-8-sig')
print('列标准差：中位 %.4f 均值 %.4f 最小 %.4f 最大 %.4f' % (
    np.median(Y.std(0, ddof=1)), Y.std(0, ddof=1).mean(), Y.std(0, ddof=1).min(), Y.std(0, ddof=1).max()), flush=True)
print('完成', flush=True)

# -*- coding: utf-8 -*-
# 06b_member_core 五成员定义（06b 外层基座与 06b2 内层基座共用同一份实现）
# 目的：内外层成员必须完全同构，否则内层选出的参数换到外层就不是同一族模型
# 另一目的：等级直方图由统一入口 member_hist 从「逐树 z 空间预测」生成，
#           任何成员都不会再出现「只写点预测、忘写直方图」的缺项（漏洞 01）
import os
import sys

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import SEED, TARGET_COLS, score_to_grade, trees

# 成员超参：与开发补丁 §4.1 的成员池一致
N_EST = 300
N_EST_DIM = 200
MF1 = 0.5
MF2 = 0.8
PCA_T = 30
TOP_K = 60
NJOB = int(os.environ.get('NJOB', '1'))
MEMBERS = ('M1', 'M2', 'M3', 'M4', 'M5')


# 折内 |Pearson| 前 top_k 列：逐维独立选，相关系数只用传入的训练集
def top_corr_cols(Xc, Ytr, top_k=TOP_K):
    n_chem = Xc.shape[1]
    C = np.zeros((n_chem, len(TARGET_COLS)))
    for j in range(len(TARGET_COLS)):
        a = Xc[:, :]
        yj = Ytr[:, j]
        ok = np.isfinite(a).all(axis=0) & (a.std(axis=0) > 1e-12)
        cc = np.zeros(n_chem)
        if ok.any():
            aa = a[:, ok]
            yz = (yj - yj.mean()) / (yj.std() + 1e-12)
            az = (aa - aa.mean(axis=0)) / (aa.std(axis=0) + 1e-12)
            cc[ok] = np.abs(az.T @ yz) / max(len(yj) - 1, 1)
        C[:, j] = cc
    return np.argsort(-C, axis=0)[:top_k, :]


# 在 (Xtr, Ytr) 上拟合五成员，返回对 Xte 的点预测与（可选）逐树 z 空间预测
# 逐树预测的最后一维是树数：M1–M3 为 trees(N_EST)，M4/M5 为 trees(N_EST_DIM)
# seed_off 让外层折与内层折用不同随机数，同时保持「同折内五成员同构」
def fit_members(Xtr, Xte, Ytr, n_chem, seed_off=0, keep_trees=False):
    n_te = Xte.shape[0]
    Xc_tr = Xtr[:, :n_chem]
    Xc_te = Xte[:, :n_chem]
    sd = np.where(Ytr.std(axis=0, ddof=1) > 1e-9, Ytr.std(axis=0, ddof=1), 1.0)
    point = {}
    tz = {}

    # M1：标准 ET，目标不归一化
    m1 = ExtraTreesRegressor(n_estimators=trees(N_EST), max_features=MF1, min_samples_leaf=1,
                             random_state=SEED + seed_off, n_jobs=NJOB).fit(Xtr, Ytr)
    point['M1'] = m1.predict(Xte)
    if keep_trees:
        tz['M1'] = np.stack([e.predict(Xte) for e in m1.estimators_], axis=2)

    # M2：目标按训练折列 sd 归一化，预测再乘回
    m2 = ExtraTreesRegressor(n_estimators=trees(N_EST), max_features=MF2, min_samples_leaf=1,
                             random_state=SEED + 100 + seed_off, n_jobs=NJOB).fit(Xtr, Ytr / sd)
    point['M2'] = m2.predict(Xte) * sd
    if keep_trees:
        tz['M2'] = np.stack([e.predict(Xte) for e in m2.estimators_], axis=2) * sd[None, :, None]

    # M3：目标打 30 维 PCA，森林回归主成分分数，再逐树逆变换回 50 维
    ncomp = min(PCA_T, Ytr.shape[0] - 1, len(TARGET_COLS))
    pca = PCA(n_components=ncomp, random_state=SEED).fit(Ytr)
    m3 = ExtraTreesRegressor(n_estimators=trees(N_EST), max_features=MF1, min_samples_leaf=1,
                             random_state=SEED + 200 + seed_off, n_jobs=NJOB).fit(
        Xtr, pca.transform(Ytr))
    point['M3'] = pca.inverse_transform(m3.predict(Xte))
    if keep_trees:
        # 逆变换 = 主成分分数 @ components_ + mean_，对每棵树独立做
        T3 = np.stack([e.predict(Xte) for e in m3.estimators_], axis=2)
        tz['M3'] = np.einsum('pkn,kd->pdn', T3, pca.components_) + pca.mean_[None, :, None]

    # M4/M5：逐维用折内 |Pearson| 前 60 列，ET 与 RF 各一套
    top = top_corr_cols(Xc_tr, Ytr, TOP_K)
    n_est_dim = trees(N_EST_DIM)
    if keep_trees:
        T4 = np.zeros((n_te, len(TARGET_COLS), n_est_dim))
        T5 = np.zeros_like(T4)
    p4 = np.zeros((n_te, len(TARGET_COLS)))
    p5 = np.zeros_like(p4)
    for j in range(len(TARGET_COLS)):
        sel = top[:, j]
        e4 = ExtraTreesRegressor(n_estimators=n_est_dim, max_features=MF1, min_samples_leaf=1,
                                 random_state=SEED + 300 + seed_off,
                                 n_jobs=NJOB).fit(Xc_tr[:, sel], Ytr[:, j])
        r5 = RandomForestRegressor(n_estimators=n_est_dim, max_features='sqrt',
                                   min_samples_leaf=1, random_state=SEED + 400 + seed_off,
                                   n_jobs=NJOB).fit(Xc_tr[:, sel], Ytr[:, j])
        p4[:, j] = e4.predict(Xc_te[:, sel])
        p5[:, j] = r5.predict(Xc_te[:, sel])
        if keep_trees:
            T4[:, j, :] = np.column_stack([e.predict(Xc_te[:, sel]) for e in e4.estimators_])
            T5[:, j, :] = np.column_stack([e.predict(Xc_te[:, sel]) for e in r5.estimators_])
    point['M4'] = p4
    point['M5'] = p5
    if keep_trees:
        tz['M4'] = T4
        tz['M5'] = T5
    return point, tz


# 逐树 z 空间预测 → 0-6 等级经验分布 (n, 50, 7)
# 全部成员走同一个入口，直方图行和恒为 1（漏洞 01 的结构性修复）
def member_hist(trees_z, mu, sg):
    g = score_to_grade(np.clip(mu[None, :, None] + sg[None, :, None] * trees_z, 0.0, 6.0))
    hist = np.zeros((trees_z.shape[0], trees_z.shape[1], 7))
    for k in range(7):
        hist[:, :, k] = np.mean(g == k, axis=2)
    return hist

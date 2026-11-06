# -*- coding: utf-8 -*-
# 03 特征构建：化合物块（哨兵填充 + 缺失指示）、双曲几何块、风格 one-hot 块
# 所有统计量只在训练折上拟合，测试折只做变换，避免泄漏
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import importlib
H = importlib.import_module('04_hyperbolic')
from common import chem_feature_cols

# 双曲块超参数，与开发文档一致：kNN 取 10，嵌入维数取 4
K_GRAPH = 10
DIM_HYP = 4
# 众数占比超过该值的列几乎不含可分信息，从化合物块中剔除
MODE_SHARE_MAX = 0.5


# 剔除 4 个汇总列与众数占比过半的近常数列，返回保留的特征列名
# rows 给定时只用这些行统计众数占比（折内拟合，修 D2：原先用全量 250 行属泄漏）
def prune_cols(chem, sum_cols=('esters_sum', 'aroma_hops_sum', 'acids_sum', 'sulfur_sum'), rows=None):
    cols = chem_feature_cols(chem)
    sub = chem if rows is None else chem.iloc[rows]
    keep = []
    for c in cols:
        v = sub[c]
        if c in sum_cols:
            continue
        share = v.value_counts(dropna=True)
        if len(share) > 0 and share.iloc[0] / len(v) >= MODE_SHARE_MAX:
            continue
        keep.append(c)
    return keep, [c for c in cols if c not in keep]


# 折内秩变换：分位数表只由训练折决定，缺失值记 0.5（秩空间的中性位），再按训练折标准差标准化
def rank_transform(Xtr, Xte):
    n = Xtr.shape[0]
    Rtr = np.full(Xtr.shape, 0.5)
    Rte = np.full(Xte.shape, 0.5)
    for j in range(Xtr.shape[1]):
        col = Xtr[:, j]
        ok = ~np.isnan(col)
        if ok.sum() < 2:
            continue
        s = np.sort(col[ok])
        Rtr[ok, j] = np.searchsorted(s, col[ok], side='left') / max(len(s) - 1, 1)
        ok2 = ~np.isnan(Xte[:, j])
        Rte[ok2, j] = np.searchsorted(s, Xte[ok2, j], side='left') / max(len(s) - 1, 1)
    mu = Rtr.mean(axis=0)
    sd = Rtr.std(axis=0, ddof=1)
    sd = np.where(sd > 1e-9, sd, 1.0)
    return (Rtr - mu) / sd, (Rte - mu) / sd


# 欧氏近邻：训练池内部互相排除自身，测试折只查训练池，返回 (邻下标, 距离)
# 距离用平方展开算，(n,d) 规模下比 cdist 省一次全量分配
def knn_neighbors(Ztr, Zte, k, chunk=256):
    a2 = np.sum(Ztr * Ztr, axis=1)
    Dte = np.zeros((Zte.shape[0], k))
    Ite = np.zeros((Zte.shape[0], k), dtype=int)
    for s in range(0, Zte.shape[0], chunk):
        e = min(s + chunk, Zte.shape[0])
        b = Zte[s:e]
        d2 = a2[None, :] - 2.0 * b @ Ztr.T + np.sum(b * b, axis=1)[:, None]
        np.maximum(d2, 0.0, out=d2)
        Ite[s:e] = np.argsort(d2, axis=1)[:, :k]
        Dte[s:e] = np.sqrt(np.take_along_axis(d2, Ite[s:e], axis=1))
    Dtr = np.zeros((Ztr.shape[0], k))
    Itr = np.zeros((Ztr.shape[0], k), dtype=int)
    for i in range(Ztr.shape[0]):
        d2 = a2 - 2.0 * Ztr[i] @ Ztr.T + a2[i]
        d2[i] = np.inf
        idx = np.argsort(d2)[:k]
        Itr[i] = idx
        Dtr[i] = np.sqrt(np.maximum(d2[idx], 0.0))
    return (Itr, Dtr), (Ite, Dte)


# 近邻权重：1/(d+eps)，越大越近权重越高；k=0 时返回空
def knn_weights(D):
    return 1.0 / (D + 1e-6)


# 用训练折的最小值构造哨兵值，比最小值再低 10 个量程，树模型据此把缺失与低值分开
def sentinel_fill(Xtr, Xte):
    mn = np.nanmin(Xtr, axis=0)
    mx = np.nanmax(Xtr, axis=0)
    sent = mn - 10.0 * (mx - mn + 1.0)
    Ztr = np.where(np.isnan(Xtr), sent[None, :], Xtr)
    Zte = np.where(np.isnan(Xte), sent[None, :], Xte)
    return Ztr, Zte, sent


# 缺失指示列：对全部含缺失的列建指示列，与是否被裁剪无关
def miss_indicators(Xtr, Xte):
    has = np.isnan(Xtr).any(axis=0)
    return np.isnan(Xtr)[:, has].astype(float), np.isnan(Xte)[:, has].astype(float), np.where(has)[0]


# 风格 one-hot：类别表由训练折决定，测试折出现新风格时整行为 0
def style_onehot(style_tr, style_te):
    cats = sorted(pd.unique(style_tr))
    idx = {s: i for i, s in enumerate(cats)}
    Mtr = np.zeros((len(style_tr), len(cats)))
    Mte = np.zeros((len(style_te), len(cats)))
    for i, s in enumerate(style_tr):
        Mtr[i, idx[s]] = 1.0
    for i, s in enumerate(style_te):
        if s in idx:
            Mte[i, idx[s]] = 1.0
    return Mtr, Mte, cats


# 双曲几何块：训练折内部建图并应力嵌入，测试折用切空间岭回归映射进球
# c 取极小值时双曲距离退化为欧氏距离的常数倍，用于 E5 消融
def hyperbolic_block(Xtr, Xte, style_tr, style_te, c=1.0, dim=None, k=None):
    dim = DIM_HYP if dim is None else dim
    k = K_GRAPH if k is None else k
    D = H.knn_graph_distance(Xtr, k=k)
    Ztr_full, _ = H.fit_poincare(D, dim=dim, c=c, iters=500, lr=0.02)
    ridge = H.fit_tangent_ridge(Xtr, Ztr_full, c=c, alpha=1.0)
    Zte = H.transform_tangent_ridge(ridge, Xte, c=c)
    feats, _ = H.hyperbolic_features(Ztr_full, Zte, style_tr, style_te, c=c)
    names = list(feats.keys())
    M = np.column_stack([feats[n] for n in names])
    return M[:len(Xtr)], M[len(Xtr):], names, Ztr_full, Zte


# 组装完整特征矩阵，blocks 控制启用哪些块，feat_cols 为空时用裁剪后的化合物列
def build(chem, style, train_idx, test_idx, blocks=('chem', 'miss', 'hyp', 'style'),
          c=1.0, dim=None, k=None, feat_cols=None):
    if feat_cols is None:
        # 未指定列时按训练折众数占比裁剪，避免用测试折统计量（D2）
        feat_cols, _ = prune_cols(chem, rows=train_idx)
    allc = chem_feature_cols(chem)
    X_all = chem[allc].to_numpy(dtype=float)
    X_raw = chem[feat_cols].to_numpy(dtype=float)
    Xtr_raw, Xte_raw = X_raw[train_idx], X_raw[test_idx]
    Xtr_all, Xte_all = X_all[train_idx], X_all[test_idx]
    del allc
    Ftr, Fte, names = [], [], []
    Ztr, Zte, _ = sentinel_fill(Xtr_raw, Xte_raw)
    if 'chem' in blocks:
        Ftr.append(Ztr)
        Fte.append(Zte)
        names += list(feat_cols)
    if 'miss' in blocks:
        Mtr, Mte, which = miss_indicators(Xtr_all, Xte_all)
        Ftr.append(Mtr)
        Fte.append(Mte)
        names += ['miss__' + str(j) for j in which]
    if 'hyp' in blocks:
        Htr, Hte, hnames, _, _ = hyperbolic_block(Ztr, Zte, style[train_idx], style[test_idx],
                                                  c=c, dim=dim, k=k)
        Ftr.append(Htr)
        Fte.append(Hte)
        names += hnames
    if 'style' in blocks:
        Str, Ste, cats = style_onehot(style[train_idx], style[test_idx])
        Ftr.append(Str)
        Fte.append(Ste)
        names += ['style__' + s for s in cats]
    return np.column_stack(Ftr), np.column_stack(Fte), names

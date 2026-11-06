# -*- coding: utf-8 -*-
# 审查用分析二：诚实口径的融合增益（权重交叉拟合，避免自评乐观）
import importlib
import itertools
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.optimize import nnls
from sklearn.model_selection import StratifiedKFold

HERE = os.path.dirname(os.path.abspath(__file__))
Q1 = os.path.join(os.path.dirname(HERE), 'q1')
sys.path.insert(0, Q1)
from common import CORE_COLS, SEED, TARGET_COLS, load_chem, load_sensory, macro_r2

sens = load_sensory()
style = sens['tasting_category_fine'].to_numpy()
Y = sens[TARGET_COLS].to_numpy(float)
core_idx = [TARGET_COLS.index(c) for c in CORE_COLS]
folds = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED).split(np.zeros(len(style)), style))
P1 = np.load(os.path.join(HERE, 'probe_pred.npz'))
BASE = {k: P1[k] for k in P1.files}


def evalp(tag, P, out):
    r2 = macro_r2(Y, P)
    out[tag] = dict(macro_r2=float(r2.mean()), core_r2=float(r2[core_idx].mean()),
                    mae_z=float(np.abs(P - Y).mean()), n_bad=int((r2 < 0.10).sum()))
    print(f'{tag:34s} R2={out[tag]["macro_r2"]:.4f} core={out[tag]["core_r2"]:.4f} '
          f'MAEz={out[tag]["mae_z"]:.4f} dim<0.10={out[tag]["n_bad"]}', flush=True)


out = {}
for k in BASE:
    evalp('single_' + k, BASE[k], out)

# 等权平均：没有任何需要拟合的参数，因此不存在权重自评乐观
for keys in (['A0_multi_raw_mf0.5', 'A4_pca30_norm', 'A6_perdim_mf0.5_top60', 'A8_perdim_rf0.5_top60'],
             ['A0_multi_raw_mf0.5', 'A1_multi_norm_mf0.5', 'A3_multi_norm_mf0.8', 'A4_pca30_norm',
              'A6_perdim_mf0.5_top60', 'A7_perdim_mf1.0_top60', 'A8_perdim_rf0.5_top60'],
             ['A1_multi_norm_mf0.5', 'A4_pca30_norm', 'A6_perdim_mf0.5_top60', 'A8_perdim_rf0.5_top60']):
    P = np.mean([BASE[k] for k in keys], axis=0)
    evalp(f'mean_{len(keys)}', P, out)

# 逐维非负最小二乘，权重在训练折内交叉拟合，评的是没见过权重的样本
def nnls_cv(keys, tag):
    M = np.stack([BASE[k] for k in keys], axis=2)
    P = np.zeros_like(Y)
    for tr, te in folds:
        for j in range(Y.shape[1]):
            X = np.column_stack([M[tr, j, :], np.ones(len(tr))])
            w, _ = nnls(X, Y[tr, j])
            P[te, j] = np.column_stack([M[te, j, :], np.ones(len(te))]) @ w
    evalp(tag, P, out)
    return P


ALL = ['A0_multi_raw_mf0.5', 'A1_multi_norm_mf0.5', 'A3_multi_norm_mf0.8', 'A4_pca30_norm',
       'A6_perdim_mf0.5_top60', 'A7_perdim_mf1.0_top60', 'A8_perdim_rf0.5_top60']
nnls_cv(ALL, f'nnls_{len(ALL)}_all_et')
nnls_cv(['A0_multi_raw_mf0.5', 'A4_pca30_norm', 'A6_perdim_mf0.5_top60'], 'nnls_3_et')
nnls_cv(['A0_multi_raw_mf0.5', 'A6_perdim_mf0.5_top60', 'A8_perdim_rf0.5_top60'], 'nnls_multi_perdim_rf')

# 残差相关性：成员之间的多样性，解释融合为什么有效或无效
res = {k: Y - BASE[k] for k in ALL}
cs = []
for a, b in itertools.combinations(ALL, 2):
    cs.append(float(np.corrcoef(res[a].ravel(), res[b].ravel())[0, 1]))
print('\n残差两两相关：均值 %.4f  最小 %.4f  最大 %.4f' % (np.mean(cs), np.min(cs), np.max(cs)), flush=True)
json.dump(out, open(os.path.join(HERE, 'analyze_fusion.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('完成')

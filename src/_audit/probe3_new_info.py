# -*- coding: utf-8 -*-
# 审查用探针 3：两种"新信息"来源能不能抬动尾部维度
# 一、邻域平均（KNN）作为不同归纳偏置的成员；二、感官阈值交叉指标（超阈值计数）作为新特征
import importlib
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
Q1 = os.path.join(os.path.dirname(HERE), 'q1')
sys.path.insert(0, Q1)
FE = importlib.import_module('03_features')
from common import (ATT, CORE_COLS, OUT, SEED, TARGET_COLS, load_chem, load_sensory, load_x8,
                    macro_r2, norm_name)

chem = load_chem()
sens = load_sensory()
style = sens['tasting_category_fine'].to_numpy()
Y = sens[TARGET_COLS].to_numpy(float)
keep, _ = FE.prune_cols(chem)
core_idx = [TARGET_COLS.index(c) for c in CORE_COLS]
folds = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED).split(Y, style))
CACHE = [(tr, te) + tuple(FE.build(chem, style, tr, te, blocks=('chem', 'miss'), feat_cols=keep)[:2])
         for tr, te in folds]
P1 = np.load(os.path.join(HERE, 'probe_pred.npz'))
RES, PRED = {}, {}


def score(tag, P):
    r2 = macro_r2(Y, P)
    RES[tag] = dict(macro_r2=float(r2.mean()), core_r2=float(r2[core_idx].mean()),
                    mae_z=float(np.abs(P - Y).mean()), n_bad=int((r2 < 0.10).sum()),
                    r2_dim=r2.tolist())
    PRED[tag] = P.copy()
    print(f'{tag:30s} R2={RES[tag]["macro_r2"]:.4f} core={RES[tag]["core_r2"]:.4f} '
          f'MAEz={RES[tag]["mae_z"]:.4f} dim<0.10={RES[tag]["n_bad"]}', flush=True)


# 邻域平均：秩变换后标准化再算距离，避免大尺度列主导距离
XL = chem[keep].apply(lambda s: s.fillna(s.median())).to_numpy(float)
XR = np.apply_along_axis(lambda v: rankdata(v) / len(v), 0, XL)
for k in (5, 10, 25):
    P = np.zeros_like(Y)
    for tr, te, A, B in CACHE:
        sc = StandardScaler().fit(XR[tr])
        m = KNeighborsRegressor(n_neighbors=k, weights='distance').fit(sc.transform(XR[tr]), Y[tr])
        P[te] = m.predict(sc.transform(XR[te]))
    score(f'K_knn{k}', P)

# 超阈值计数特征：用 8.xlsx 的阈值均值，把「浓度是否超过感官阈值」变成显式指示与计数
th = load_x8('Sensory thresholds').dropna(subset=['threshold_mean'])
th = th[th['field_name_R'].isin(keep)]
print('可用于阈值特征的因素：%d 个' % len(th), flush=True)
cm = pd.read_csv(os.path.join(ATT, '1.csv'))
hi, cnt_all, cnt_by_origin = [], [], []
origin = th['origin_paper'].fillna('NA').to_numpy()
for _, r in th.iterrows():
    col = r['field_name_R']
    v = cm[col].to_numpy(float)
    v = np.where(np.isnan(v), np.nanmedian(v), v)
    # 列若已是 log10 浓度则先还原
    c = np.where(v.min() < 0, 10.0 ** v, v)
    t = float(r['threshold_mean'])
    hi.append((c > t).astype(float))
    cnt_all.append((c > t).astype(float))
Hf = np.column_stack(hi)
group = {}
for g in sorted(set(origin)):
    group[g] = np.array([g == o for o in origin])
agg = np.column_stack([Hf.sum(1)] + [Hf[:, m].sum(1) for g, m in group.items() if m.sum() >= 3])
print('阈值指示特征 %d 列，聚合 %d 列，超阈值计数均值 %.2f'
      % (Hf.shape[1], agg.shape[1], agg[:, 0].mean()), flush=True)

# 把阈值特征接到化合物块后面，看宏 R2 与尾部维度是否改善
for tag, extra in (('T1_thr_ind', Hf), ('T2_thr_agg', agg), ('T3_thr_all', np.hstack([Hf, agg]))):
    P = np.zeros_like(Y)
    for tr, te, A, B in CACHE:
        A2 = np.hstack([A, extra[tr]])
        B2 = np.hstack([B, extra[te]])
        m = ExtraTreesRegressor(n_estimators=300, max_features=0.5, min_samples_leaf=1,
                                n_jobs=1, random_state=SEED).fit(A2, Y[tr])
        P[te] = m.predict(B2)
    score(tag, P)

# 等权融合：现有四个 ET 变体 + KNN + 阈值特征版本
mem = ['A0_multi_raw_mf0.5', 'A4_pca30_norm', 'A6_perdim_mf0.5_top60', 'A8_perdim_rf0.5_top60']
for add in (['K_knn10'], ['T3_thr_all'], ['K_knn10', 'T3_thr_all']):
    keys = mem + add
    P = np.mean([P1[k] if k in P1.files else PRED[k] for k in keys], axis=0)
    score('M_mean+' + '+'.join(add), P)

json.dump(RES, open(os.path.join(HERE, 'probe3.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
np.savez_compressed(os.path.join(HERE, 'probe3_pred.npz'), **PRED)
print('完成', flush=True)

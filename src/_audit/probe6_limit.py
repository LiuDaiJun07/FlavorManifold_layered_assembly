# -*- coding: utf-8 -*-
# 审查用探针 6：判定"是数据到头了，还是模型不够好"
# 两件事：(1) 学习曲线——加大样本量还有没有得涨；(2) 残差里还有没有可利用的结构
import importlib
import json
import os
import sys

import numpy as np
from scipy.stats import spearmanr
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import StratifiedShuffleSplit

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'q1'))
FE = importlib.import_module('03_features')
from common import CORE_COLS, SEED, TARGET_COLS, load_chem, load_sensory, load_sensory7, macro_r2

chem = load_chem()
sens = load_sensory()
s7 = load_sensory7()
style = sens['tasting_category_fine'].to_numpy()
Y = sens[TARGET_COLS].to_numpy(float)
keep, _ = FE.prune_cols(chem)
core_idx = [TARGET_COLS.index(c) for c in CORE_COLS]
OUT = {}

# 第一件：学习曲线。固定一个 60 瓶的测试集，训练集从 40 涨到 190，看 R2 还涨不涨。
sss = StratifiedShuffleSplit(n_splits=1, test_size=60, random_state=SEED)
tr_pool, te = next(sss.split(Y, style))
sizes = [40, 70, 100, 130, 160, 190]
rows = []
for n in sizes:
    if n > len(tr_pool):
        continue
    acc = []
    for rep in range(3):
        rng = np.random.RandomState(SEED + rep)
        sub = np.sort(rng.choice(tr_pool, size=n, replace=False))
        A, B, _ = FE.build(chem, style, sub, te, blocks=('chem', 'miss'), feat_cols=keep)
        m = ExtraTreesRegressor(n_estimators=300, max_features=0.5, min_samples_leaf=1,
                                n_jobs=1, random_state=SEED).fit(A, Y[sub])
        P = m.predict(B)
        acc.append(macro_r2(Y[te], P).mean())
    rows.append((n, float(np.mean(acc)), float(np.std(acc))))
    print(f'训练 {n:3d} 瓶 → 测试 R2 = {np.mean(acc):.4f} (sd {np.std(acc):.4f})', flush=True)
OUT['learning_curve'] = rows

# 第二件：残差里还有没有结构。用 10 折 OOF 残差与 7.csv 的 186 个描述词做相关。
# 7.csv 是感官数据，只用于"诊断残差是否可解释"，绝不进任何模型的特征。
d = np.load(os.path.join(HERE, 'probe4_pred.npz'))
P = d['10f_M1']
R = Y - P
s7cols = [c for c in s7.columns if c.startswith(('aroma_', 'palate_', 'taste_'))]
S7 = s7[s7cols].to_numpy(float)
best = []
for j, col in enumerate(TARGET_COLS):
    cors = []
    for k in range(S7.shape[1]):
        v = S7[:, k]
        ok = np.isfinite(v) & np.isfinite(R[:, j])
        if ok.sum() < 60:
            continue
        cors.append(abs(spearmanr(v[ok], R[ok, j]).statistic))
    cors = np.array([c for c in cors if np.isfinite(c)])
    best.append(float(cors.max()) if len(cors) else np.nan)
best = np.array(best)
OUT['residual_vs_7'] = dict(mean_max_abs_rho=float(np.nanmean(best)),
                            n_gt_030=int((best > 0.30).sum()),
                            n_gt_020=int((best > 0.20).sum()),
                            per_dim=dict(zip(TARGET_COLS, best.tolist())))
print(f'残差与 7.csv 的最大 |rho|：均值 {np.nanmean(best):.3f}，'
      f'超 0.30 的维度 {int((best > 0.30).sum())}/50，超 0.20 的 {int((best > 0.20).sum())}/50', flush=True)

# 第三件：不同家族的残差相关，判断"共同漏掉的部分"占多大比例
sets = {}
for name, fn in (('probe_pred.npz', ('A0_multi_raw_mf0.5', 'A5_perdim_mf0.5_all', 'A8_perdim_rf0.5_top60')),
                 ('probe2_pred.npz', ('B1_gbr_perdim',)),
                 ('probe3_pred.npz', ('K_knn10',))):
    p = os.path.join(HERE, fn[0])
    if not os.path.exists(p):
        continue
    z = np.load(p)
    for k in fn[1]:
        if k in z:
            sets[k] = z[k]
sets['10f_M1'] = P
names = list(sets)
C = np.eye(len(names))
for i in range(len(names)):
    for j in range(i + 1, len(names)):
        r1 = (Y - sets[names[i]]).ravel()
        r2 = (Y - sets[names[j]]).ravel()
        C[i, j] = C[j, i] = float(np.corrcoef(r1, r2)[0, 1])
print('残差相关矩阵：', flush=True)
for i, n in enumerate(names):
    print(f'  {n:22s} ' + ' '.join(f'{C[i, j]:.3f}' for j in range(len(names))), flush=True)
OUT['residual_corr'] = {'names': names, 'C': C.tolist()}

json.dump(OUT, open(os.path.join(HERE, 'probe6.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('完成', flush=True)

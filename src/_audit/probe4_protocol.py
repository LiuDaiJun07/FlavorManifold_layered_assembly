# -*- coding: utf-8 -*-
# 审查用探针 4：换协议能不能抬动（10 折，每折训练 225 行，更接近交付模型的 250 行）
import importlib
import json
import os
import sys
import time

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold

HERE = os.path.dirname(os.path.abspath(__file__))
Q1 = os.path.join(os.path.dirname(HERE), 'q1')
sys.path.insert(0, Q1)
FE = importlib.import_module('03_features')
from common import CORE_COLS, SEED, TARGET_COLS, load_chem, load_sensory, macro_r2

chem = load_chem()
sens = load_sensory()
style = sens['tasting_category_fine'].to_numpy()
Y = sens[TARGET_COLS].to_numpy(float)
keep, _ = FE.prune_cols(chem)
core_idx = [TARGET_COLS.index(c) for c in CORE_COLS]
RES, PRED = {}, {}


def score(tag, P):
    r2 = macro_r2(Y, P)
    RES[tag] = dict(macro_r2=float(r2.mean()), core_r2=float(r2[core_idx].mean()),
                    mae_z=float(np.abs(P - Y).mean()), n_bad=int((r2 < 0.10).sum()),
                    r2_dim=r2.tolist())
    PRED[tag] = P.copy()
    print(f'{tag:26s} R2={RES[tag]["macro_r2"]:.4f} core={RES[tag]["core_r2"]:.4f} '
          f'MAEz={RES[tag]["mae_z"]:.4f} dim<0.10={RES[tag]["n_bad"]}', flush=True)


for K in (10,):
    folds = list(StratifiedKFold(n_splits=K, shuffle=True, random_state=SEED).split(Y, style))
    t0 = time.time()
    CACHE = [(tr, te) + tuple(FE.build(chem, style, tr, te, blocks=('chem', 'miss'), feat_cols=keep)[:2])
             for tr, te in folds]
    print(f'{K} 折特征就绪 {time.time() - t0:.0f}s', flush=True)
    P = {k: np.zeros_like(Y) for k in ('M1', 'M2', 'M3', 'M4', 'M5')}
    for tr, te, A, B in CACHE:
        mu, sd = Y[tr].mean(0), Y[tr].std(0, ddof=1)
        sd = np.where(sd < 1e-12, 1.0, sd)
        P['M1'][te] = ExtraTreesRegressor(n_estimators=300, max_features=0.5, min_samples_leaf=1,
                                          n_jobs=1, random_state=SEED).fit(A, Y[tr]).predict(B)
        m = ExtraTreesRegressor(n_estimators=300, max_features=0.8, min_samples_leaf=1,
                                n_jobs=1, random_state=SEED).fit(A, (Y[tr] - mu) / sd)
        P['M2'][te] = m.predict(B) * sd + mu
        Zn = (Y[tr] - mu) / sd
        pca = PCA(n_components=30, random_state=SEED).fit(Zn)
        m = ExtraTreesRegressor(n_estimators=300, max_features=0.5, min_samples_leaf=1,
                                n_jobs=1, random_state=SEED).fit(A, pca.transform(Zn))
        P['M3'][te] = pca.inverse_transform(m.predict(B)) * sd + mu
        from scipy.stats import spearmanr
        for tag, est in (('M4', 'et'), ('M5', 'rf')):
            out = np.zeros((len(te), Y.shape[1]))
            for j in range(Y.shape[1]):
                col = np.nan_to_num([abs(spearmanr(A[:, i], Y[tr, j]).statistic) for i in range(A.shape[1])])
                idx = np.argsort(-col)[:60]
                if est == 'et':
                    mm = ExtraTreesRegressor(n_estimators=300, max_features=0.5, min_samples_leaf=1,
                                             n_jobs=1, random_state=SEED)
                else:
                    mm = RandomForestRegressor(n_estimators=300, max_features=0.5, min_samples_leaf=1,
                                               n_jobs=1, random_state=SEED)
                out[:, j] = mm.fit(A[:, idx], Y[tr, j]).predict(B[:, idx])
            P[tag][te] = out
    print(f'{K} 折拟合完成 {time.time() - t0:.0f}s', flush=True)
    for k in P:
        score(f'{K}f_{k}', P[k])
    score(f'{K}f_mean4', np.mean([P[k] for k in ('M1', 'M3', 'M4', 'M5')], axis=0))
    score(f'{K}f_mean3', np.mean([P[k] for k in ('M1', 'M3', 'M4')], axis=0))
    score(f'{K}f_mean5', np.mean([P[k] for k in ('M1', 'M2', 'M3', 'M4', 'M5')], axis=0))

json.dump(RES, open(os.path.join(HERE, 'probe4.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
np.savez_compressed(os.path.join(HERE, 'probe4_pred.npz'), **PRED)
print('完成', flush=True)

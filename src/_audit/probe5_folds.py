# -*- coding: utf-8 -*-
# 审查用探针 5：只测"折数"这一件事——10 折基线 vs 5 折基线（同一个 M1 配置）
import importlib
import os
import sys

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import StratifiedKFold

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'q1'))
FE = importlib.import_module('03_features')
from common import CORE_COLS, SEED, TARGET_COLS, load_chem, load_sensory, macro_r2

chem = load_chem()
sens = load_sensory()
style = sens['tasting_category_fine'].to_numpy()
Y = sens[TARGET_COLS].to_numpy(float)
keep, _ = FE.prune_cols(chem)
core_idx = [TARGET_COLS.index(c) for c in CORE_COLS]

for K in (5, 10, 25):
    P = np.zeros_like(Y)
    for tr, te in StratifiedKFold(n_splits=K, shuffle=True, random_state=SEED).split(Y, style):
        A, B, _ = FE.build(chem, style, tr, te, blocks=('chem', 'miss'), feat_cols=keep)
        m = ExtraTreesRegressor(n_estimators=300, max_features=0.5, min_samples_leaf=1,
                                n_jobs=1, random_state=SEED).fit(A, Y[tr])
        P[te] = m.predict(B)
    r2 = macro_r2(Y, P)
    print(f'{K:2d} 折基线  R2={r2.mean():.4f} core={r2[core_idx].mean():.4f} '
          f'MAEz={np.abs(P - Y).mean():.4f} dim<0.10={int((r2 < 0.10).sum())}', flush=True)
print('完成', flush=True)

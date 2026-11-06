# -*- coding: utf-8 -*-
# 临时测试：验证双曲模块的数值正确性与收敛性
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import importlib
h = importlib.import_module('04_hyperbolic')
from common import load_chem, load_sensory, TARGET_COLS

c = load_chem()
feat = [x for x in c.columns if x not in ('beer', 'beer_id', 'tasting_category_fine')]
X = c[feat].to_numpy(dtype=float)
X = np.where(np.isnan(X), np.nanmin(X, axis=0) - 10.0, X)

t0 = time.time()
D = h.knn_graph_distance(X, k=10)
print('knn_graph_distance 用时 %.2fs' % (time.time() - t0))
print('图距离: min=%.4f max=%.4f mean=%.4f, 有限比例=%.4f'
      % (D[D > 0].min(), D.max(), D[D > 0].mean(), np.isfinite(D).mean()))

# 单点自检：同一点距离应为 0，球内点在界内
Z = h.exp_map0(np.random.RandomState(0).randn(5, 4) * 0.3, 1.0)
print('球内半径 max=%.4f (<1)' % np.linalg.norm(Z, axis=1).max())
print('自身距离 diag=%.2e' % np.abs(np.diag(h.poincare_dist(Z, Z, 1.0))).max())

for dim in [2, 4]:
    t0 = time.time()
    Zd, loss = h.fit_poincare(D, dim=dim, c=1.0, iters=400, lr=0.02)
    dt = time.time() - t0
    Dn = D / max(D[D > 0].mean(), 1e-9)
    dh = h.poincare_dist(Zd, Zd, 1.0)
    iu = np.triu_indices(len(Zd), 1)
    corr = np.corrcoef(dh[iu], Dn[iu])[0, 1]
    resid = np.sqrt(np.mean((dh[iu] - Dn[iu]) ** 2))
    print('dim=%d 用时 %.1fs loss=%.5f  距离保序相关=%.4f  RMSE=%.4f' % (dim, dt, loss, corr, resid))

    s = load_sensory()
    styles = s['tasting_category_fine'].to_numpy()
    tr = np.arange(200)
    te = np.arange(200, 250)
    Ztr, Zte = Zd[:200], Zd[200:]
    feats, sl = h.hyperbolic_features(Ztr, Zte, styles[:200], styles[200:], 1.0)
    print('  双曲特征数=%d 名字示例=%s' % (len(feats), list(feats)[:3]))
    bad = [k for k, v in feats.items() if not np.all(np.isfinite(v))]
    print('  非有限特征:', bad if bad else '无')
    print('  hyp_radius 范围 [%.4f, %.4f]' % (feats['hyp_radius'].min(), feats['hyp_radius'].max()))

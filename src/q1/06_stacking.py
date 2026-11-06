# -*- coding: utf-8 -*-
# 06 折外 Stacking：一层四个基学习器，二层逐维岭回归得到 E3/E4
# 修复：中位数填充只用训练折（D3）、二层不再截断到训练目标值域（D4）、
#       不再重算 ET_E1 取 hist，直接复用 05 的折外分布（D5）
import importlib
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
FE = importlib.import_module('03_features')
from common import (OUT, SEED, TARGET_COLS, ensure_dirs, load_chem, load_sensory,
                    load_sensory_grade, macro_r2, r2_groups, split_folds, trees)

ensure_dirs()
L = []


# 追加一行到报告并打印
def P(s=''):
    L.append(str(s))
    print(s)


N_EST = 300
MF = 0.5
MSL = 1
PCA_DIM = 30
NJOB = int(os.environ.get('NJOB', '1'))

chem = load_chem()
sens = load_sensory()
Y = sens[TARGET_COLS].to_numpy(dtype=float)
G = load_sensory_grade()
style = sens['tasting_category_fine'].astype(str).to_numpy()
N = len(Y)
gfolds = (G.mean(axis=1) >= 3.0).astype(int)
folds = split_folds(np.char.add(style, np.where(gfolds > 0, '_h', '_l')))

# ET_E0 / ET_E1 的折外预测直接取 05 的产物，避免重复训练与口径漂移
z05 = np.load(os.path.join(OUT, '05_oof.npz'), allow_pickle=False)
ET_E0 = z05['E0_pred']
ET_E1 = z05['E1_pred']

P('# 06 折外 Stacking（四基 + 二层逐维岭回归）')
P()
P(f'- 基学习器：ET_E0 / ET_E1（来自 05 折外）/ RF_E1 / RD_E1（PCA{PCA_DIM}+岭回归）')
P(f'- 树数 {trees(N_EST)}，NJOB={NJOB}，折数 {len(folds)}')
P(f'- 中位数填充只用训练折（修 D3）；二层预测不再截断到训练目标值域（修 D4）')
P()

RF_E1 = np.zeros_like(Y)
RD_E1 = np.zeros_like(Y)
for f, (tr, te) in enumerate(folds):
    keep, _ = FE.prune_cols(chem, rows=tr)
    Xtr, Xte, names = FE.build(chem, style, tr, te, blocks=('chem', 'miss', 'style'),
                               feat_cols=keep)
    m = RandomForestRegressor(n_estimators=trees(N_EST), max_features='sqrt',
                              min_samples_leaf=MSL, random_state=SEED + f, n_jobs=NJOB)
    m.fit(Xtr, Y[tr])
    RF_E1[te] = m.predict(Xte)
    # RD_E1：化合物原值做折内中位数填充（修 D3）+ 标准化 + 主成分降维，风格 one-hot 直接拼入
    # 注意必须用原始 chem 的 NaN 结构；特征块里的缺失已经被哨兵值顶掉，拿它填中位数等于没填
    Xr = chem[keep].to_numpy(float)
    med = chem.iloc[tr][keep].median().to_numpy(float)
    med = np.where(np.isfinite(med), med, 0.0)
    Xc_tr = np.where(np.isnan(Xr[tr]), med[None, :], Xr[tr])
    Xc_te = np.where(np.isnan(Xr[te]), med[None, :], Xr[te])
    # 折内 0.5%/99.5% 分位截断：个别化合物存在远超训练折量级的离群值，
    # 不截断会让 PCA 的载荷被单条样本的极端值主导，测试折预测直接炸到 ±100
    lo = np.quantile(Xc_tr, 0.005, axis=0)
    hi = np.quantile(Xc_tr, 0.995, axis=0)
    Xc_tr = np.clip(Xc_tr, lo, hi)
    Xc_te = np.clip(Xc_te, lo, hi)
    sc = StandardScaler().fit(Xc_tr)
    Ptr = sc.transform(Xc_tr)
    Pte = sc.transform(Xc_te)
    pdim = min(PCA_DIM, Ptr.shape[1], Ptr.shape[0] - 1)
    pca = PCA(n_components=pdim, random_state=SEED).fit(Ptr)
    Rtr = np.column_stack([pca.transform(Ptr), Xtr[:, len(keep):]])
    Rte = np.column_stack([pca.transform(Pte), Xte[:, len(keep):]])
    rd = Ridge(alpha=10.0).fit(Rtr, Y[tr])
    RD_E1[te] = rd.predict(Rte)
    print(f'  fold {f + 1}/{len(folds)}  rf+pca 完成  n_feat={Xtr.shape[1]}', flush=True)

base = {'ET_E0': ET_E0, 'ET_E1': ET_E1, 'RF_E1': RF_E1, 'RD_E1': RD_E1}
L2 = {'E3': ['ET_E0', 'ET_E1', 'RF_E1', 'RD_E1'], 'E4': ['ET_E0', 'ET_E1', 'RF_E1']}
stk = {}
for name, mem in L2.items():
    out = np.zeros_like(Y)
    for f, (tr, te) in enumerate(folds):
        Mtr = np.column_stack([base[k][tr] for k in mem])
        Mte = np.column_stack([base[k][te] for k in mem])
        for j in range(Y.shape[1]):
            # 逐维岭回归，alpha=30；不做值域截断，只把最终答案裁剪到 0-6
            m = Ridge(alpha=30.0).fit(Mtr, Y[tr, j])
            out[te, j] = m.predict(Mte)
    stk[name] = out

rows = []
for name in ['ET_E0', 'ET_E1', 'RF_E1', 'RD_E1', 'E3', 'E4']:
    v = base.get(name, stk.get(name))
    mac, weak, strong, rd = r2_groups(Y, v)
    rows.append({'cfg': name, 'macro_r2': mac, 'weak_r2': weak, 'strong_r2': strong,
                 **{f'r2_{c}': float(rd[j]) for j, c in enumerate(TARGET_COLS)}})
    P(f'- {name}: 宏 R2 = {mac:.4f}，12 弱维 = {weak:.4f}，5 强维 = {strong:.4f}')
P()

R = pd.DataFrame(rows)
R.to_csv(os.path.join(OUT, '06_stacking.csv'), index=False, encoding='utf-8-sig')
np.savez_compressed(os.path.join(OUT, '06_stack.npz'), y=Y, g=G,
                    **{f'oof_{k}': v for k, v in base.items()},
                    **{f'oof_{k}': v for k, v in stk.items()},
                    E1_hist=z05['E1_hist'])
with open(os.path.join(OUT, '06_stacking.md'), 'w', encoding='utf-8') as f:
    f.write('\n'.join(L))
print()
print('06 完成，输出写入', OUT)

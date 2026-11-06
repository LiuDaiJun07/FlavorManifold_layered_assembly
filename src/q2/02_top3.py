# -*- coding: utf-8 -*-
# Q2 步骤二：每个感官维度的 Top-3 关键因素化合物
# 三条独立证据链：
#   一、单变量：化合物与感官维度的 Spearman 相关，200 次 bootstrap 给入选频率
#   二、多变量：逐维极端森林的折外置换重要性，5 折轮换
#   三、网络：把重要性沿共表达图向同模块邻居扩散，抑制孤立噪声
# 三者秩归一后加权合成综合得分取 Top-3，再做跨方法一致性与化学合理性检验
import importlib
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import StratifiedKFold

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'q1'))
FE = importlib.import_module('03_features')
from common import (ATT, OUT, OUT_Q2, SEED, TARGET_COLS, chem_meta, ensure_dirs, load_chem,
                    load_graph_corr, load_sensory, load_x8)

ensure_dirs()
L = []


def P(s=''):
    L.append(str(s))
    print(s)


chem = load_chem()
sens = load_sensory()
style = sens['tasting_category_fine'].to_numpy()
Y = sens[TARGET_COLS].to_numpy(float)
keep, dropped = FE.prune_cols(chem)
Xraw = chem[keep].to_numpy(float)
X, _, _ = FE.sentinel_fill(Xraw, Xraw)
MI, _, mi_idx = FE.miss_indicators(Xraw, Xraw)
A1 = np.hstack([X, MI])
FEAT = list(keep) + [f'miss__{chem.columns[i]}' for i in mi_idx]
p = A1.shape[1]
N_BOOT = 200
N_EST = 200
t0 = time.time()
P('# Q2 步骤二：每个感官维度的 Top-3 关键因素化合物')
P()
P(f'候选因素 {p} 个 = 化合物与理化列 {len(keep)} 个 + 缺失指示 {len(mi_idx)} 个；'
  f'目标为 4.csv 的 {len(TARGET_COLS)} 个感官维度。')
P()

# 一、单变量 Spearman 相关与 bootstrap 入选频率
rho = np.zeros((p, len(TARGET_COLS)))
boot_hit = np.zeros((p, len(TARGET_COLS)))
rng = np.random.default_rng(SEED)
for b in range(N_BOOT):
    idx = rng.integers(0, len(style), len(style))
    xb = A1[idx]
    rb = np.zeros((p, len(TARGET_COLS)))
    for j in range(len(TARGET_COLS)):
        yb = Y[idx, j]
        ok = np.isfinite(yb)
        sd = xb[ok].std(0)
        for i in range(p):
            if sd[i] > 0:
                rb[i, j] = spearmanr(xb[ok, i], yb[ok]).statistic
    rb = np.nan_to_num(rb)
    if b == 0:
        rho = rb.copy()
    top = np.argsort(-np.abs(rb), axis=0)[:3]
    for j in range(len(TARGET_COLS)):
        boot_hit[top[:, j], j] += 1
boot_hit /= N_BOOT
P(f'单变量 {N_BOOT} 次 bootstrap 完成，用时 {time.time() - t0:.0f}s。')
P()

# 二、逐维极端森林的折外置换重要性
# 打乱某列后折外 MAE 的上升量即该列对该维度的边际信息量，方向交给单变量相关的符号
folds = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED).split(A1, style))
perm = np.zeros((p, len(TARGET_COLS)))
for fi, (tr, te) in enumerate(folds):
    for j in range(len(TARGET_COLS)):
        m = ExtraTreesRegressor(n_estimators=N_EST, max_features=0.5, min_samples_leaf=1,
                                n_jobs=1, random_state=SEED).fit(A1[tr], Y[tr, j])
        base = np.abs(m.predict(A1[te]) - Y[te, j]).mean()
        r = permutation_importance(m, A1[te], Y[te, j], n_repeats=2, random_state=SEED,
                                   scoring='neg_mean_absolute_error', n_jobs=1)
        perm[:, j] += (base - (-r.importances.mean(axis=1))) / len(folds)
    P(f'  折 {fi + 1}/5 的置换重要性完成，累计 {time.time() - t0:.0f}s')
perm = np.maximum(perm, 0.0)
P()

# 三、沿共表达图扩散
names_g, Rg = load_graph_corr()
Adf = pd.DataFrame(np.abs(Rg), index=names_g, columns=names_g)
W = np.zeros((p, p))
Gi = [c for c in keep if c in Adf.columns]
pos = {c: i for i, c in enumerate(keep)}
for ca in Gi:
    for cb in Gi:
        v = Adf.at[ca, cb]
        if v > 0:
            W[pos[ca], pos[cb]] = v
deg = W.sum(1, keepdims=True)
Wn = np.divide(W, np.where(deg == 0, 1, deg))
ALPHA = 0.5
perm_g = (1 - ALPHA) * perm + ALPHA * (Wn @ perm)
P(f'图扩散：{len(Gi)} 个化合物在 2.csv 共表达图上至少有一个邻居，混合系数 alpha = {ALPHA}。')
P()


def rank01(M):
    out = np.zeros_like(M)
    for j in range(M.shape[1]):
        o = np.argsort(np.argsort(M[:, j]))
        out[:, j] = o / max(len(o) - 1, 1)
    return out


S = 0.40 * rank01(np.abs(rho)) + 0.45 * rank01(perm_g) + 0.15 * boot_hit
P('综合得分 = 0.40 * 单变量相关秩 + 0.45 * 图扩散置换重要性秩 + 0.15 * bootstrap 入选频率。')
P()

mm = chem_meta()
comm = {}
gmeta = os.path.join(OUT_Q2, '01_graph_meta.json')
if os.path.exists(gmeta):
    comm = json.load(open(gmeta, encoding='utf-8'))['community']

rows = []
for j, dim in enumerate(TARGET_COLS):
    o = np.argsort(-S[:, j])[:3]
    for k, i in enumerate(o):
        c = FEAT[i]
        rows.append(dict(dim=dim, rank=k + 1, compound=c,
                         chemical_name=str(mm.get(c, {}).get('chemical_name', 'NA')),
                         origin=str(mm.get(c, {}).get('origin_paper', 'NA')),
                         community=int(comm.get(c, -1)), score=float(S[i, j]),
                         rho=float(rho[i, j]), perm=float(perm[i, j]),
                         perm_graph=float(perm_g[i, j]), boot_freq=float(boot_hit[i, j])))
TOP = pd.DataFrame(rows)
P('## 1 各维度的 Top-3 关键因素')
P()
P('| 感官维度 | 第 1（rho） | 第 2（rho） | 第 3（rho） |')
P('| --- | --- | --- | --- |')
for dim in TARGET_COLS:
    s3 = TOP[TOP['dim'] == dim].sort_values('rank')
    P(f'| {dim} | ' + ' | '.join(f'{r.compound} ({r.rho:+.2f})' for _, r in s3.iterrows()) + ' |')
P()

P('## 2 稳定性')
P()
P('| 感官维度 | Top-3 的 bootstrap 入选频率均值 | 单变量与多变量 Top-3 重合数 |')
P('| --- | --- | --- |')
jac = []
for j, dim in enumerate(TARGET_COLS):
    s3 = TOP[TOP['dim'] == dim]
    uni = set(np.array(FEAT)[np.argsort(-np.abs(rho[:, j]))[:3]])
    mul = set(np.array(FEAT)[np.argsort(-perm_g[:, j])[:3]])
    inter = len(uni & mul)
    jac.append(inter / len(uni | mul))
    P(f'| {dim} | {s3["boot_freq"].mean():.2f} | {inter} |')
jac = np.array(jac)
P()
P(f'两组独立方法 Top-3 的 Jaccard 系数均值 {jac.mean():.3f}，完全一致 {int((jac == 1).sum())} 个维度，'
  f'完全不一致 {int((jac == 0).sum())} 个维度。')
P()

# 化学合理性：气味活度比 OAV 与阈值
P('## 3 化学合理性')
P()
thr = load_x8('Sensory thresholds').dropna(subset=['threshold_mean'])
tmap = {str(r['field_name_R']): float(r['threshold_mean']) for _, r in thr.iterrows()}
oav_rows = []
for c in sorted(set(TOP['compound'])):
    t = tmap.get(c)
    if t and t > 0:
        oav = float(np.median(np.power(10.0, chem[c].to_numpy(float))) / t)
        oav_rows.append(dict(compound=c, threshold=t, median_conc=float(np.median(np.power(10.0, chem[c].to_numpy(float)))), oav=oav))
OAV = pd.DataFrame(oav_rows).sort_values('oav', ascending=False) if oav_rows else pd.DataFrame()
P(f'{len(OAV)} 个 Top-3 化合物在 8.xlsx 中有实测阈值，可算气味活度比 OAV = 中位浓度 / 阈值；'
  f'OAV > 1 表示该化合物在典型浓度下已能被感知，是化学上说得通的关键因素。')
if len(OAV):
    P()
    P('| 化合物 | 阈值 | 中位浓度 | OAV |')
    P('| --- | --- | --- | --- |')
    for _, r in OAV.head(15).iterrows():
        P(f'| {r.compound} | {r.threshold:.4g} | {r.median_conc:.4g} | {r.oav:.3g} |')
    P()
    P(f'其中 OAV > 1 的占 {int((OAV["oav"] > 1).sum())}/{len(OAV)}。')
P()

# 与 6.csv 全样本相关阵的方向核对，6.csv 不参与建模，只作事后一致性检查
S6 = pd.read_csv(os.path.join(ATT, '6.csv'), index_col=0)
P('用 6.csv（全样本化合物-感官 Spearman，未参与建模）核对方向：')
P()
hit = tot = 0
P('| 感官维度 | 化合物 | 本方法 rho | 6.csv rho | 方向一致 |')
P('| --- | --- | --- | --- | --- |')
for _, r in TOP.iterrows():
    if r.compound not in S6.index or r.dim not in S6.columns:
        continue
    v = float(S6.at[r.compound, r.dim])
    ok = np.sign(v) == np.sign(r.rho)
    hit += int(ok)
    tot += 1
    if tot <= 24:
        P(f'| {r.dim} | {r.compound} | {r.rho:+.3f} | {v:+.3f} | {"是" if ok else "否"} |')
P()
P(f'可比对的 {tot} 个组合中方向一致 {hit} 个，占 {hit / max(tot, 1):.1%}。')
P()
TOP.to_csv(os.path.join(OUT_Q2, '02_top3.csv'), index=False, encoding='utf-8-sig')
OAV.to_csv(os.path.join(OUT_Q2, '02_oav.csv'), index=False, encoding='utf-8-sig')
np.savez_compressed(os.path.join(OUT_Q2, '02_scores.npz'), S=S, rho=rho, perm=perm,
                    perm_graph=perm_g, boot=boot_hit, feat=np.array(FEAT))
with open(os.path.join(OUT_Q2, '02_top3.md'), 'w', encoding='utf-8') as f:
    f.write('\n'.join(L))
print()
print('Top-3 结果已写入', OUT_Q2)

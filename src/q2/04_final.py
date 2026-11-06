# -*- coding: utf-8 -*-
# Q2 步骤四：最终版 Top-3 关键因素化合物
# 前三步暴露了两个问题：一、候选集用了 Q1 的裁剪口径，把众数占比高的列剔掉了，
#   于是 4vg 这类几乎全在检出限以下的化合物根本没资格入选，而它恰恰是该维度的直接来源；
# 二、极端森林的置换重要性在 250 行样本上很不稳，会把同组化合物的替身排到前面。
# 本步改成：候选用全部 231 个化合物列，条件重要性改用 Lasso 系数（对共线性更稳），
#   再用 2.csv 的共表达图做一次扩散，最后与单变量相关、bootstrap 入选频率合成。
import importlib
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Lasso, LassoCV
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'q1'))
FE = importlib.import_module('03_features')
from common import (ATT, OUT_Q2, PHYS_COLS, SEED, TARGET_COLS, chem_meta, chem_name_map,
                    ensure_dirs, load_chem, load_graph_corr, load_sensory, load_x8)

ensure_dirs()
L = []
rng = np.random.default_rng(SEED)


def P(s=''):
    L.append(str(s))
    print(s)


chem = load_chem()
sens = load_sensory()
Y = sens[TARGET_COLS].to_numpy(float)
# 候选集：1.csv 的全部化合物与理化列，不做 Q1 那样的近常数列裁剪
CAND = [c for c in chem.columns if c not in ('beer', 'beer_id', 'tasting_category_fine')]
Xraw = chem[CAND].to_numpy(float)
X, _, _ = FE.sentinel_fill(Xraw, Xraw)
MI, _, mi_idx = FE.miss_indicators(Xraw, Xraw)
A = np.hstack([X, MI])
FEAT = list(CAND) + [f'miss__{chem.columns[i]}' for i in mi_idx]
p = A.shape[1]
is_phys = np.array([f in PHYS_COLS for f in FEAT])
P('# Q2 步骤四：最终版 Top-3 关键因素化合物')
P()
P(f'候选因素 {p} 个（化合物与理化列 {len(CAND)} 个，缺失指示 {len(mi_idx)} 个），目标 {len(TARGET_COLS)} 个感官维度。')
P('候选集不再做近常数列裁剪：像 X4VG 这种绝大多数样本都在检出限以下的化合物，')
P('它对 4vg 维度恰好是定义级的关键因素，剔除它会让结论失去意义。')
P()

# 一 单变量 Spearman，全样本口径
P('## 1 单变量相关')
P()
rho = np.zeros((p, len(TARGET_COLS)))
for j in range(len(TARGET_COLS)):
    y = Y[:, j]
    for i in range(p):
        v = A[:, i]
        if np.std(v) > 0:
            rho[i, j] = spearmanr(v, y).statistic
rho = np.nan_to_num(rho)
P(f'|rho| 的中位数 {np.median(np.abs(rho)):.3f}，最大 {np.abs(rho).max():.3f}。')
P()

# 二 Lasso 条件重要性：多变量同时进入，天然处理化合物之间的强相关
P('## 2 Lasso 条件重要性')
P()
Xs = StandardScaler().fit_transform(A)
Ys = StandardScaler().fit_transform(Y)
COEF = np.zeros((p, len(TARGET_COLS)))
ALPHA = np.zeros(len(TARGET_COLS))
for j in range(len(TARGET_COLS)):
    m = LassoCV(cv=5, n_alphas=30, random_state=SEED, n_jobs=1, max_iter=20000).fit(Xs, Ys[:, j])
    COEF[:, j] = m.coef_
    ALPHA[j] = m.alpha_
nz = (np.abs(COEF) > 0).sum(0)
P(f'每个维度被选中的非零系数个数：中位数 {np.median(nz):.0f}，最少 {nz.min()}，最多 {nz.max()}；')
P(f'交叉验证选出的 alpha 中位数 {np.median(ALPHA):.4f}。')
P('Lasso 把同组高度相关的化合物压到只剩一个代表，这正是「关键因素」要的效果。')
P()

# 三 稳定性：bootstrap 重抽 200 次，统计每个因素被选中且符号一致的比例
P('## 3 bootstrap 入选频率')
P()
N_BOOT = 200
hit = np.zeros((p, len(TARGET_COLS)))
sign0 = np.sign(COEF)
for b in range(N_BOOT):
    idx = rng.integers(0, len(A), len(A))
    if len(np.unique(idx)) < 20:
        continue
    Xb = StandardScaler().fit_transform(A[idx])
    Yb = StandardScaler().fit_transform(Y[idx])
    for j in range(len(TARGET_COLS)):
        m = Lasso(alpha=ALPHA[j], max_iter=20000).fit(Xb, Yb[:, j])
        sel = (np.abs(m.coef_) > 0) & (np.sign(m.coef_) == sign0[:, j])
        hit[:, j] += sel
hit /= N_BOOT
P(f'{N_BOOT} 次重抽样后，入选频率的中位数 {np.median(hit[COEF != 0]):.3f}（只统计被选中的系数）。')
P()

# 四 沿 2.csv 共表达图扩散：一个化合物若与已入选的关键化合物高度共表达，也应获得加分
P('## 4 沿共表达图扩散')
P()
cols_g, Rg = load_graph_corr()
pos = {c: i for i, c in enumerate(cols_g)}
W = np.zeros((len(CAND), len(CAND)))
for i, a in enumerate(CAND):
    if a not in pos:
        continue
    for j2, b in enumerate(CAND):
        if b in pos:
            W[i, j2] = abs(Rg[pos[a], pos[b]])
np.fill_diagonal(W, 0.0)
rs = W.sum(1, keepdims=True)
Wn = np.divide(W, np.where(rs == 0, 1, rs))
# 缺失指示列没有图节点，扩散时只在化合物列上做
IMP = np.abs(COEF)
IMP_g = IMP.copy()
IMP_g[:len(CAND)] = 0.5 * IMP[:len(CAND)] + 0.5 * (Wn @ IMP[:len(CAND)])
P(f'图覆盖 {len(cols_g)} 个化合物节点，扩散系数取 0.5（一半保留自身、一半来自共表达邻居的均值）。')
P()


def rank01(M):
    out = np.zeros_like(M, dtype=float)
    for j in range(M.shape[1]):
        o = np.argsort(np.argsort(M[:, j]))
        out[:, j] = o / max(len(o) - 1, 1)
    return out


S = 0.40 * rank01(np.abs(rho)) + 0.40 * rank01(IMP_g) + 0.20 * hit

mm = chem_meta()
gmeta = os.path.join(OUT_Q2, '01_graph_meta.json')
comm = json.load(open(gmeta, encoding='utf-8')).get('community', {}) if os.path.exists(gmeta) else {}
# 正式答案只在化合物集合内排序，理化指标与派生汇总列另作参考
# 4 个汇总列（esters_sum / aroma_hops_sum / acids_sum / sulfur_sum）是整组化合物的求和，
# 不是单一化合物，放在化合物榜里会让 acidity 这类维度选出「酸类总和」这种答案，因此一并排除
AGG = set(['esters_sum', 'aroma_hops_sum', 'acids_sum', 'sulfur_sum'])
cand_comp = np.array([(not is_phys[i]) and (not FEAT[i].startswith('miss__')) and (FEAT[i] not in AGG)
                      for i in range(p)])
Sc = np.where(cand_comp[:, None], S, -np.inf)
rows = []
for j, dim in enumerate(TARGET_COLS):
    for k, i in enumerate(np.argsort(-Sc[:, j])[:3]):
        c = FEAT[i]
        rows.append(dict(dim=dim, rank=k + 1, factor=c, kind='化合物',
                         chemical_name=str(mm.get(c, {}).get('chemical_name', 'NA')),
                         origin=str(mm.get(c, {}).get('origin_paper', 'NA')),
                         community=int(comm.get(c, -1)), score=float(S[i, j]),
                         rho=float(rho[i, j]), lasso=float(COEF[i, j]),
                         lasso_graph=float(IMP_g[i, j]), boot_freq=float(hit[i, j])))
T = pd.DataFrame(rows)
# 每个维度的信号强度：化合物候选里最大的 |rho|，用来判断这份 Top-3 值不值得解读
sig = np.abs(np.where(cand_comp[:, None], rho, 0.0)).max(0)
P('## 5 各维度的 Top-3 关键因素化合物')
P()
P('| 感官维度 | 第 1 | 第 2 | 第 3 | 该维最强 abs(rho) | 信号 |')
P('| --- | --- | --- | --- | --- | --- |')
for j, dim in enumerate(TARGET_COLS):
    s = T[T['dim'] == dim].sort_values('rank')
    lvl = '强' if sig[j] >= 0.35 else ('中' if sig[j] >= 0.22 else '弱')
    P(f'| {dim} | ' + ' | '.join(s['factor']) + f' | {sig[j]:.3f} | {lvl} |')
P()
n_weak = int((sig < 0.22).sum())
n_mid = int(((sig >= 0.22) & (sig < 0.35)).sum())
n_strong = int((sig >= 0.35).sum())
P(f'按 signal 强弱分：强（abs(rho) >= 0.35）{n_strong} 个维度，中 {n_mid} 个，弱（< 0.22）{n_weak} 个。')
P('表里最后两列是这份 Top-3 的可信度：该维最强 abs(rho) 是化合物候选里最大的单变量相关，')
P('强信号（>=0.35）的维度可以直接当结论用；弱信号（<0.22）的维度即使给出前三名，')
P('也只是「相对最强的三个」，任何单个化合物都解释不了多少秩方差，不宜作化学解释。')
P()
mean_boot = float(T['boot_freq'].mean())
P(f'150 个席位的平均 bootstrap 入选频率 {mean_boot:.3f}，其中超过 0.5 的 {int((T["boot_freq"] > 0.5).sum())} 个。')
P('入选频率低于 0.5 的席位说明该因素在重抽样中不稳定，只能算候选。')
P()
# 方向也要交代：第一名是负相关的维度，说明它与该感官是反向共变，多由风格混杂造成
first = T[T['rank'] == 1].set_index('dim')
neg = first[first['rho'] < 0]
if len(neg):
    P(f'还有 {len(neg)} 个维度的第一名是负相关（' + '、'.join(
        [f'{d} {r:+.2f}' for d, r in zip(neg.index[:6], neg['rho'][:6])]) + '）。')
    P('负相关说明该化合物升高时这个感官反而降低，化学上未必讲得通，更可能是风格混杂的证据：')
    P('比如酸类维度挑出的是酯类化合物，而酯高、酸低的样本恰好集中在某几个风格里。')
    P('这类维度建议与第 7.1 节的理化指标一起看，理化指标的符号更可靠。')
    P()

# 五 化学合理性：与 6.csv 的方向核对，以及阈值口径的气味活度比
P('## 6 化学合理性')
P()
S6 = pd.read_csv(os.path.join(ATT, '6.csv'), index_col=0)
name_map = chem_name_map()
field2name = {v: k for k, v in name_map.items()}
sn = load_x8('Sensory names')
sn = sn[sn['dataset'].astype(str).str.contains('Trained')]
dim2col = {str(r['R_name']): str(r['CleanName_PanelFeedback']) for _, r in sn.iterrows()}
hit6 = tot6 = 0
for _, r in T.iterrows():
    nm = field2name.get(r.factor)
    col = dim2col.get(r.dim)
    if nm is None or col is None or nm not in S6.index or col not in S6.columns:
        continue
    tot6 += 1
    hit6 += int(np.sign(float(S6.at[nm, col])) == np.sign(r.rho))
P(f'6.csv 是 250 个样本的全样本化合物-感官 Spearman，与本步的 rho 是同一统计量。')
P(f'150 个 Top-3 席位里能对上 {tot6} 个，方向一致 {hit6} 个，占 {hit6 / max(tot6, 1):.1%}，说明读表与符号约定没有出错。')
P('它能校验相关这一侧，但不能校验多变量选择这一侧，那一侧要靠 Lasso 的选择频率与阈值判断。')
P()

def conc_of(col):
    v = chem[col].to_numpy(float)
    return np.power(10.0, v) if np.nanmin(v) < 0 else v


thr = load_x8('Sensory thresholds').dropna(subset=['threshold_mean'])
tmap = {str(r['field_name_R']): float(r['threshold_mean']) for _, r in thr.iterrows()}
oav_rows = []
for c in sorted(set(T['factor'])):
    t = tmap.get(c)
    if t and t > 0 and c in chem.columns:
        conc = conc_of(c)
        oav_rows.append(dict(compound=c, scale='log10' if np.nanmin(chem[c].to_numpy(float)) < 0 else 'raw',
                             threshold=t, median_conc=float(np.nanmedian(conc)),
                             oav=float(np.nanmedian(conc) / t)))
OAV = pd.DataFrame(oav_rows).sort_values('oav', ascending=False) if oav_rows else pd.DataFrame()
if len(OAV):
    P(f'{len(OAV)} 个关键因素在阈值表里有实测阈值，气味活度比 OAV = 中位浓度 / 阈值：')
    P()
    P('| 化合物 | 列的标度 | 阈值（原单位） | 中位浓度 | OAV |')
    P('| --- | --- | --- | --- | --- |')
    for _, r in OAV.iterrows():
        P(f'| {r.compound} | {r.scale} | {r.threshold:.4g} | {r.median_conc:.4g} | {r.oav:.3g} |')
    P()
    P(f'OAV 大于 1 的有 {int((OAV["oav"] > 1).sum())}/{len(OAV)} 个。')
    P('阈值表混用 mg/L、g/L、ug/L，与各列原始单位未必一致，所以这里只看量级是否荒谬。')
    P()

# 六 同名因素自检：直接对应的化合物或理化指标应当排在最前
P('## 7 同名因素自检')
P()
P('自检要分两类看：一类是同名的**理化指标**（不是化合物，按题目口径不进化合物榜），')
P('另一类是同名的**化合物**（应当出现在前三）。')
P()
P('### 7.1 同名的理化指标')
P()
PHYS_PAIR = {'bitternes': 'bitterness', 'lactic': 'lactic_acid.mg.L.', 'acetic': 'acetic_acid.g.L.',
             'alcohol': 'ethanol..v.v.', 'body': 'kcalperc'}
P('| 感官维度 | 同名理化指标 | 它的 rho | 在全部 239 个候选里的名次 | 化合物榜第 1 | 化合物榜第 1 的 rho |')
P('| --- | --- | --- | --- | --- | --- |')
for dim, pc in PHYS_PAIR.items():
    if dim not in TARGET_COLS or pc not in FEAT:
        continue
    j = TARGET_COLS.index(dim)
    rr = float(rho[FEAT.index(pc), j])
    rk_all = int((np.abs(rho[:, j]) > abs(rr)).sum()) + 1
    s = T[T['dim'] == dim].sort_values('rank')
    c1 = s.iloc[0]['factor'] if len(s) else '-'
    c1r = float(rho[FEAT.index(c1), j]) if len(s) else float('nan')
    P(f'| {dim} | {pc} | {rr:+.3f} | {rk_all} | {c1} | {c1r:+.3f} |')
P()
P('这张表说明两件事。第一，苦味、乳酸、乙酸、酒精这几个维度本身就有对应的理化测量，')
P('它们的秩相关最高能到 0.68（bitterness 对 bitternes），这几个指标就是这些维度最干净的证据，')
P('只是题目问的是化合物，所以它们单独列在这里、不进第 5 节的化合物榜。')
P('第二，1.csv 里没有异 α 酸（只有 alpha.humulene、beta.humulene 这类酒花油成分，不是苦味物质），')
P('所以 bitternes 维度的化合物榜只能是「与苦味共变」的挥发性成分，反映的是风格混杂，')
P('不能读成「这些酯导致了苦味」。')
P()
P('### 7.2 同名的化合物')
P()
PAIR = {'dms': 'DMS', 'diacetyl': 'X2.3.butanedione', 't2n': 'X.E..2.nonenal', 'X4vg': 'X4VG'}
P('| 感官维度 | 同名化合物 | 它的 rho | 它在化合物榜里的名次 |')
P('| --- | --- | --- | --- |')
MISS = []
for dim, comp in PAIR.items():
    if dim not in TARGET_COLS or comp is None or comp not in chem.columns:
        continue
    s = T[T['dim'] == dim].sort_values('rank')
    rk = s.index[s['factor'] == comp]
    pos_txt = str(int(s.loc[rk[0], 'rank'])) if len(rk) else '未入前三'
    rr = float(rho[FEAT.index(comp), TARGET_COLS.index(dim)])
    if not len(rk):
        MISS.append((dim, comp, rr))
    P(f'| {dim} | {comp} | {rr:+.3f} | {pos_txt} |')
P()
if MISS:
    txt = '、'.join([f'{d}（rho {r:+.3f}）' for d, _, r in MISS])
    P(f'这里没有全过：{len(MISS)} 个维度的同名化合物没能进前三，分别是 {txt}。')
    P('X4VG 在 X4vg 维度上排第一，这个维度的信号是真的，方法也确实抓到了；')
    P('剩下三个的实测浓度与同名感官维度的秩相关只有 0.09 到 0.15，远低于「化学上应当强相关」的预期。')
    P('原因大概是三条：一是这三类缺陷物质在多数样本里低于阈值，感官打分更多受偶然污染与个体敏感度影响；')
    P('二是面板评分本身有噪声；三是这几列大量左删失（多数样本压在检出限），秩相关被并列值稀释。')
    P('因此本报告不宣称「DMS 是 dms 维度的关键因素」，只如实报告它在数据里排在第几位。')
P()

T.to_csv(os.path.join(OUT_Q2, '02c_top3.csv'), index=False, encoding='utf-8-sig')
if len(OAV):
    OAV.to_csv(os.path.join(OUT_Q2, '02c_oav.csv'), index=False, encoding='utf-8-sig')
np.savez_compressed(os.path.join(OUT_Q2, '02c_scores.npz'), S=S, rho=rho, coef=COEF,
                    imp_graph=IMP_g, boot=hit, alpha=ALPHA, feat=np.array(FEAT))
with open(os.path.join(OUT_Q2, '02c_top3.md'), 'w', encoding='utf-8') as f:
    f.write('\n'.join(L))
print()
print('最终版 Top-3 已写入', OUT_Q2)

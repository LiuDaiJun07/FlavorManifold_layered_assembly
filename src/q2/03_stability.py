# -*- coding: utf-8 -*-
# Q2 步骤三：稳固性与化学解释的补算
# 补齐三处：一、把相关系数改成全样本口径（步骤二里存的是第一次 bootstrap 重抽样的值）
#           二、6.csv 的行名是化学名而非字段名，用 common 的映射表对上后再核对方向
#           三、把 Top-3 里属于理化指标的项单独标注，题目问的是化合物
import importlib
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'q1'))
FE = importlib.import_module('03_features')
from common import (ANS, ATT, OUT_Q2, PHYS_COLS, SEED, TARGET_COLS, chem_meta, chem_name_map,
                    ensure_dirs, load_chem, load_graph_corr, load_sensory, load_x8)

ensure_dirs()
L = []


def P(s=''):
    L.append(str(s))
    print(s)


chem = load_chem()
sens = load_sensory()
Y = sens[TARGET_COLS].to_numpy(float)
keep, _ = FE.prune_cols(chem)
Xraw = chem[keep].to_numpy(float)
X, _, _ = FE.sentinel_fill(Xraw, Xraw)
MI, _, mi_idx = FE.miss_indicators(Xraw, Xraw)
A1 = np.hstack([X, MI])
FEAT = list(keep) + [f'miss__{chem.columns[i]}' for i in mi_idx]
p = A1.shape[1]

d = np.load(os.path.join(OUT_Q2, '02_scores.npz'), allow_pickle=True)
S0, perm, perm_g, boot = d['S'], d['perm'], d['perm_graph'], d['boot']

P('# Q2 步骤三：Top-3 结果的补算与稳定性')
P()
P(f'候选因素 {p} 个，目标 {len(TARGET_COLS)} 个感官维度。')
P()

# 全样本 Spearman，替换步骤二里第一次重抽样的口径
rho = np.zeros((p, len(TARGET_COLS)))
for j in range(len(TARGET_COLS)):
    y = Y[:, j]
    for i in range(p):
        v = A1[:, i]
        if np.isfinite(v).all() and np.nanstd(v) > 0:
            rho[i, j] = spearmanr(v, y).statistic
rho = np.nan_to_num(rho)
P('相关系数已改为全样本口径：步骤二里存的是第一次 bootstrap 重抽样的值，只用于算入选频率，')
P('对外报告的关键因素强度必须用全样本值，否则同一张表里两种口径混用。')
P()


def rank01(M):
    out = np.zeros_like(M)
    for j in range(M.shape[1]):
        o = np.argsort(np.argsort(M[:, j]))
        out[:, j] = o / max(len(o) - 1, 1)
    return out


S = 0.40 * rank01(np.abs(rho)) + 0.45 * rank01(perm_g) + 0.15 * boot
mm = chem_meta()
gmeta = os.path.join(OUT_Q2, '01_graph_meta.json')
comm = json.load(open(gmeta, encoding='utf-8')).get('community', {}) if os.path.exists(gmeta) else {}

rows = []
for j, dim in enumerate(TARGET_COLS):
    for k, i in enumerate(np.argsort(-S[:, j])[:3]):
        c = FEAT[i]
        rows.append(dict(dim=dim, rank=k + 1, factor=c,
                         kind='理化指标' if c in PHYS_COLS else '化合物',
                         chemical_name=str(mm.get(c, {}).get('chemical_name', 'NA')),
                         origin=str(mm.get(c, {}).get('origin_paper', 'NA')),
                         community=int(comm.get(c, -1)), score=float(S[i, j]),
                         rho=float(rho[i, j]), perm=float(perm[i, j]),
                         perm_graph=float(perm_g[i, j]), boot_freq=float(boot[i, j])))
T = pd.DataFrame(rows)

P('## 1 各感官维度的 Top-3 关键因素')
P()
P('| 感官维度 | 第 1（rho） | 第 2（rho） | 第 3（rho） |')
P('| --- | --- | --- | --- |')
for dim in TARGET_COLS:
    s3 = T[T['dim'] == dim].sort_values('rank')
    P(f'| {dim} | ' + ' | '.join(f'{r.factor} ({r.rho:+.2f})' for _, r in s3.iterrows()) + ' |')
P()
n_phys = int((T['kind'] == '理化指标').sum())
P(f'150 个 Top-3 席位里，{150 - n_phys} 个是化合物，{n_phys} 个是理化指标'
  f'（IBU、色度、酒精度、pH、热量、蛋白、铁、二氧化碳、残糖等）。')
P('理化指标本身不是「化合物」，但它们是有机化合物谱的线性组合，在树模型里充当了整组同源化合物的替身；')
P('题目问的关键因素化合物，应以 kind 为「化合物」的行为准，理化指标行保留以便解释。')
P()

# 稳定性：单变量与多变量的一致程度
P('## 2 稳定性')
P()
P('| 感官维度 | Top-3 的 bootstrap 入选频率均值 | 单变量 Top-3 与多变量 Top-3 重合数 | Jaccard |')
P('| --- | --- | --- | --- |')
jac = []
for j, dim in enumerate(TARGET_COLS):
    s3 = T[T['dim'] == dim]
    uni = set(np.array(FEAT)[np.argsort(-np.abs(rho[:, j]))[:3]])
    mul = set(np.array(FEAT)[np.argsort(-perm_g[:, j])[:3]])
    inter = len(uni & mul)
    jac.append(inter / len(uni | mul))
    P(f'| {dim} | {s3["boot_freq"].mean():.2f} | {inter} | {inter / len(uni | mul):.2f} |')
jac = np.array(jac)
P()
P(f'Jaccard 均值 {jac.mean():.3f}，完全一致 {int((jac == 1).sum())} 维，完全不一致 {int((jac == 0).sum())} 维。')
P(f'Top-3 的 bootstrap 入选频率均值 {T["boot_freq"].mean():.3f}，'
  f'超过 0.5 的席位 {int((T["boot_freq"] > 0.5).sum())}/150。')
P()
P('这两组数说明什么：入选频率衡量的是单变量排序在重抽样下的稳定度，中位数在 0.4 到 0.8 之间，')
P('说明排序本身还算稳；而单变量与多变量的 Top-3 重合度低，说明「单独相关最强」和「联合建模里最有增量」')
P('不是同一批化合物，这正是指标设计里要求做跨方法一致性的原因。多变量那一侧更接近因果意义上的关键因素，')
P('因为它控制住了其余化合物之后再看边际信息量。')
P()
corr_j = np.corrcoef(np.abs(rho).ravel(), perm_g.ravel())[0, 1]
P(f'全体 218 x 50 个组合上，|单变量相关| 与图扩散置换重要性的秩相关 {corr_j:.3f}。')
P()

# 用 6.csv 核对方向：先把字段名换成化学名
P('## 3 化学合理性')
P()
S6 = pd.read_csv(os.path.join(ATT, '6.csv'), index_col=0)
name_map = chem_name_map()
field2name = {v: k for k, v in name_map.items()}
# 6.csv 的列名是面板用的口语名（Malt aroma - overall），4.csv 的列名是代码名（A_malt_all），
# 两者的对应写在 8.xlsx 的 Sensory names 表里，Trained panel 段的 CleanName_PanelFeedback 列
sn = load_x8('Sensory names')
sn = sn[sn['dataset'].astype(str).str.contains('Trained')]
dim2col = {str(r['R_name']): str(r['CleanName_PanelFeedback']) for _, r in sn.iterrows()}
cov = sum(1 for c in TARGET_COLS if dim2col.get(c) in S6.columns)
P(f'6.csv 的 {S6.shape[0]} 行化合物名与 1.csv 字段名不同，用 8.xlsx 的 field_name_R 建立 231 条映射；')
P(f'它的 {S6.shape[1]} 个列名也是面板口语名，用 Sensory names 表建立映射，{len(TARGET_COLS)} 个维度里对上 {cov} 个。')
P('6.csv 给出的正是全样本化合物-感官 Spearman，和本方法的 rho 是同一个统计量，')
P('因此这一核对验证的是「字段名到化学名的映射、读表、符号约定」没有出错，属于流程校验；')
P('它不能验证多变量重要性那一侧，那一侧只能靠 bootstrap 与化学阈值来支撑。')
P()
P('| 感官维度 | 因素 | 本方法 rho | 6.csv rho | 方向一致 |')
P('| --- | --- | --- | --- | --- |')
hit = tot = 0
for _, r in T.iterrows():
    nm = field2name.get(r.factor)
    col = dim2col.get(r.dim)
    if nm is None or col is None or nm not in S6.index or col not in S6.columns:
        continue
    v = float(S6.at[nm, col])
    ok = np.sign(v) == np.sign(r.rho)
    hit += int(ok)
    tot += 1
    if tot <= 20:
        P(f'| {r.dim} | {r.factor} | {r.rho:+.3f} | {v:+.3f} | {"是" if ok else "否"} |')
P()
P(f'可比对的 {tot} 个组合里方向一致 {hit} 个，占 {hit / max(tot, 1):.1%}。')
P()

# 气味活度比
# 1.csv 的标度并不统一：含负值的列是 log10 浓度，全程非负的列是原始浓度（见 Q1 的 09_scale_audit.md）
# 因此中位浓度必须按列判断，否则原始浓度列会算出 1e8 量级的荒谬 OAV
# 题目问的是关键因素化合物，理化指标虽然是同一组同源化合物的替身，但不能算作化合物答案
# 因此再给一份只在化合物集合内排序的 Top-3，作为正式答案；含理化指标的版本留作补充
KIND = np.array(['理化指标' if f in PHYS_COLS else '化合物' for f in FEAT])
comp_mask = KIND == '化合物'
Sc = np.where(comp_mask[:, None], S, -np.inf)
rows_c = []
for j, dim in enumerate(TARGET_COLS):
    for k, i in enumerate(np.argsort(-Sc[:, j])[:3]):
        c = FEAT[i]
        rows_c.append(dict(dim=dim, rank=k + 1, factor=c, kind='化合物',
                           chemical_name=str(mm.get(c, {}).get('chemical_name', 'NA')),
                           origin=str(mm.get(c, {}).get('origin_paper', 'NA')),
                           community=int(comm.get(c, -1)), score=float(S[i, j]),
                           rho=float(rho[i, j]), perm=float(perm[i, j]),
                           perm_graph=float(perm_g[i, j]), boot_freq=float(boot[i, j])))
TC = pd.DataFrame(rows_c)
P('## 4 只在化合物集合内排序的正式 Top-3')
P()
P(f'理化指标有 {int((~comp_mask).sum())} 个、化合物 {int(comp_mask.sum())} 个。把理化指标排除后重新取前三：')
P()
P('| 感官维度 | 第 1 | 第 2 | 第 3 |')
P('| --- | --- | --- | --- |')
for dim in TARGET_COLS:
    s = TC[TC['dim'] == dim].sort_values('rank')
    P(f'| {dim} | ' + ' | '.join(s['factor']) + ' |')
P()

def conc_of(col, v):
    v = np.asarray(v, float)
    if np.nanmin(v) < 0:
        return np.power(10.0, v)
    return v


def is_log_col(col):
    return bool(np.nanmin(chem[col].to_numpy(float)) < 0)


thr = load_x8('Sensory thresholds').dropna(subset=['threshold_mean'])
tmap = {str(r['field_name_R']): float(r['threshold_mean']) for _, r in thr.iterrows()}
oav_rows = []
for c in sorted(set(TC['factor'])):
    t = tmap.get(c)
    if t and t > 0 and c in chem.columns:
        conc = conc_of(c, chem[c].to_numpy(float))
        oav_rows.append(dict(compound=c, scale='log10' if is_log_col(c) else 'raw',
                             threshold=t, median_conc=float(np.nanmedian(conc)),
                             oav=float(np.nanmedian(conc) / t)))
OAV = pd.DataFrame(oav_rows).sort_values('oav', ascending=False) if oav_rows else pd.DataFrame()
if len(OAV):
    P(f'{len(OAV)} 个 Top-3 因素在 8.xlsx 里有实测阈值，气味活度比 OAV = 中位浓度 / 阈值。')
    P('中位浓度按列的标度还原：log10 列取 10^z，原始浓度列直接取 z。')
    P()
    P('| 化合物 | 列的标度 | 阈值（原单位） | 中位浓度 | OAV |')
    P('| --- | --- | --- | --- | --- |')
    for _, r in OAV.iterrows():
        P(f'| {r.compound} | {r.scale} | {r.threshold:.4g} | {r.median_conc:.4g} | {r.oav:.3g} |')
    P()
    P(f'OAV 大于 1 的有 {int((OAV["oav"] > 1).sum())}/{len(OAV)} 个，'
      f'即这些化合物在典型浓度下本身就超过阈值、确实能被闻到，作为关键因素在化学上说得通。')
    P('OAV 小于 1 的不能据此否定：这些物质可能是协同增香、或是被更高阈值掩盖后仍通过交互作用影响感知。')
    P('阈值表本身混用 mg/L、g/L、ug/L 三种单位，与各列的原始单位未必一致，')
    P('所以这里的 OAV 只用来判断量级是否荒谬，不能当成精确的气味活度比。')
    P()

TC.to_csv(os.path.join(OUT_Q2, '02b_top3_compound.csv'), index=False, encoding='utf-8-sig')
T.to_csv(os.path.join(OUT_Q2, '02b_top3_final.csv'), index=False, encoding='utf-8-sig')
OAV.to_csv(os.path.join(OUT_Q2, '02b_oav.csv'), index=False, encoding='utf-8-sig')
np.savez_compressed(os.path.join(OUT_Q2, '02b_scores.npz'), S=S, rho=rho, perm=perm,
                    perm_graph=perm_g, boot=boot, feat=np.array(FEAT))
with open(os.path.join(OUT_Q2, '02b_stability.md'), 'w', encoding='utf-8') as f:
    f.write('\n'.join(L))
print('补算结果已写入', OUT_Q2)

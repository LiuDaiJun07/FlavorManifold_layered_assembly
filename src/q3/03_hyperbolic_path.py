# -*- coding: utf-8 -*-
# Q3 步骤三：风格层级与双曲路径（补丁 §8.4 的升级部分）
# ① 22 个风格在 0-6 分制感官画像空间的质心 → 相关距离 D=1-r
# ② 双曲嵌入 fit_poincare(D, dim=2, c=1.0, iters=800, lr=0.02, seed=SEED) 与 MDS 对照，两应力都报
# ③ 门禁：欧氏/双曲应力比 ≥1.3 才用测地线软分配，否则降级为直线插值
# ④ 沿 Pils/Lager→Tripel 的测地线取 5 个 t，软分配权重 w_s(t)∝exp(-d_H²/2h²)，端点强制 one-hot
# ⑤ 每个 t 过缺陷守护；若劣于源则 w←(1-κ)w+κe_DST，κ 从 0.25 起翻倍并记录
# ⑥ 硬校验：t=1 必须与现行迁移输出一致（max|Δ|<1e-8）
import json
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'q1'))
import common as C
import importlib
H = importlib.import_module('04_hyperbolic')
from common import (ANS, OUT_Q3, SEED, TARGET_COLS, ensure_dirs, load_scale, load_sensory,
                    z_to_score)

ensure_dirs()
L = []


# 追加一行到报告并打印
def P(s=''):
    L.append(str(s))
    print(s)


T_S = [0.0, 0.25, 0.5, 0.75, 1.0]
SRC, DST = 'Pils/Lager', 'Tripel'
CURV = 1.0
DIM = 2
ITERS = 800
LR = 0.02
GATE = 1.3

# 22 风格按工艺/色系分家族，只用于图上着色
FAMILY = {
    'Pils/Lager': '下层发酵', 'Blond': '淡色艾尔', 'Hoppy': '淡色艾尔', 'Saison': '淡色艾尔',
    'Wheat': '小麦与酸', 'Lambic': '小麦与酸', 'Kriek': '小麦与酸', 'Faro': '小麦与酸',
    'Fruitbeer': '小麦与酸', 'Brett/cofermented': '小麦与酸',
    'Tripel': '烈性', 'Dubbel': '烈性', 'Strong ale': '烈性', 'Scotch': '烈性',
    'Brut': '烈性', 'Christmas': '烈性',
    'Amber': '深色', 'Brown': '深色', 'Stout/Porter': '深色', 'West Flanders ale': '深色',
    'Flanders old brown': '深色', 'Low/No alcohol': '深色',
}
FAM_COLOR = {'下层发酵': '#4c72b0', '淡色艾尔': '#55a868', '小麦与酸': '#c44e52',
             '烈性': '#8172b2', '深色': '#8c564b'}

sens = load_sensory()
style = sens['tasting_category_fine'].astype(str).to_numpy()
MU, SG = load_scale()
RS = z_to_score(sens[TARGET_COLS].to_numpy(float), MU, SG)
cats = sorted(pd.unique(style).tolist())
n_by = {c: int((style == c).sum()) for c in cats}
cent = np.vstack([RS[style == c].mean(0) for c in cats])

# 相关距离：先按维 z 化，再 1-r；负值截 0（相关系数>1 的数值噪声）
Zc_s = (cent - cent.mean(0)) / np.maximum(cent.std(0), 1e-9)
Rho = np.corrcoef(Zc_s)
D = np.clip(1.0 - Rho, 0.0, None)
np.fill_diagonal(D, 0.0)
P('# Q3 风格层级与双曲路径')
P()
P(f'- 风格数 {len(cats)}，质心建立在 0-6 分制感官画像空间（4.csv 的 z 已用主口径反标准化）')
P()

Z2, loss_hyp = H.fit_poincare(D, dim=DIM, c=CURV, iters=ITERS, lr=LR, seed=SEED)
from sklearn.manifold import MDS

mds = MDS(n_components=DIM, dissimilarity='precomputed', random_state=SEED, n_init=8,
          normalized_stress=True, max_iter=600)
Xm = mds.fit_transform(D)
stress_eu = float(mds.stress_)
stress_hy = float(loss_hyp)
ratio = stress_eu / max(stress_hy, 1e-12)
use_hyp = ratio >= GATE
P('## 1 嵌入应力')
P()
P('| 方法 | 归一化应力 | 说明 |')
P('| --- | --- | --- |')
P(f'| 欧氏 MDS（2 维） | {stress_eu:.4f} | 线性距离保持的下界参照 |')
P(f'| 双曲 Poincaré（2 维，c=1.0） | {stress_hy:.4f} | {ITERS} 步 Adam，lr={LR} |')
P()
P(f'- 应力比 欧氏/双曲 = {ratio:.3f}，门禁 ≥{GATE}：' +
  ('**通过，用双曲测地线软分配**' if use_hyp else '**未通过，降级为直线插值**'))
P()

# 软分配尺度 h：22 个质心到最近其它质心的双曲距离中位数
Dmm = H.poincare_dist(Z2, Z2, CURV)
np.fill_diagonal(Dmm, np.inf)
h = float(np.median(Dmm.min(axis=1)))
P(f'- 软分配带宽 h = {h:.4f}（质心间最近距离的中位数）')
P()

i_src, i_dst = cats.index(SRC), cats.index(DST)
z_s = Z2[i_src]
z_d = Z2[i_dst]
t_u = H.log_map0(z_s[None, :], CURV)[0]
t_v = H.log_map0(z_d[None, :], CURV)[0]

# 化学侧：源样本的潜编码残差 + 逐 t 的风格均值加权
d = np.load(os.path.join(OUT_Q3, '01_generative.npz'), allow_pickle=False)
keep = list(d['keep'])
mu_x, sd_x, lam, MU_S_ARR = d['mu_x'], d['sd_x'], d['lam'], d['mu_s']
# npz 里 mu_s 是 (22, 50) 矩阵、行序与 cats 一致，再建一个按风格名取值的字典
MUS = {c: MU_S_ARR[i] for i, c in enumerate(cats)}
chem = C.load_chem()
Xmed = chem[keep].median().to_numpy(float)
X = chem[keep].fillna(pd.Series(Xmed, index=keep)).to_numpy(float)
XLO, XHI = X.min(0), X.max(0)


# 反标准化并截断到支持域
def back(Zz):
    return np.clip(Zz * sd_x[None, :] + mu_x[None, :], XLO[None, :], XHI[None, :])


# 源样本在标准化化学空间的坐标：与 02_transfer 完全同一条链路，再反解潜因子残差
Zsrc = (X - mu_x[None, :]) / sd_x[None, :]
src_idx = np.where(style == SRC)[0]
Zs = np.linalg.lstsq(lam, (Zsrc[src_idx] - MUS[SRC][None, :]).T, rcond=None)[0].T
res = Zs @ lam.T
Xsrc = back(Zsrc[src_idx])

rows = []
guarded = {}
for t in T_S:
    if use_hyp:
        zt = H.exp_map0(((1 - t) * t_u + t * t_v)[None, :], CURV)
    else:
        zt = ((1 - t) * z_s + t * z_d)[None, :]
    dd = H.poincare_dist(zt, Z2, CURV)[0]
    w = np.exp(-dd ** 2 / (2 * h ** 2))
    w = w / w.sum()
    if t == 0.0:
        w = np.zeros(len(cats))
        w[i_src] = 1.0
    if t == 1.0:
        w = np.zeros(len(cats))
        w[i_dst] = 1.0

    drew = t
    kappa = 0.0
    Xt = back(w @ MU_S_ARR + res)
    Xg = Xt.copy()
    nfix = {}
    for c in ['DMS', 'X2.3.butanedione', 'X.E..2.nonenal', 'humulene.epoxide.II',
              'X14.hydroxy.alpha.humulene', 'X.E..dihydro.apofarnesal']:
        if c in keep:
            i = keep.index(c)
            bad = Xg[:, i] > Xsrc[:, i]
            nfix[c] = int(bad.sum())
            Xg[bad, i] = Xsrc[bad, i]
    # 若守护后仍有缺陷劣于源风格均值，收紧权重向目标覆盖（κ 翻倍）
    def worse_cnt(Xa):
        n = 0
        for c in ['DMS', 'X2.3.butanedione', 'X.E..2.nonenal', 'humulene.epoxide.II',
                  'X14.hydroxy.alpha.humulene', 'X.E..dihydro.apofarnesal']:
            if c in keep:
                i = keep.index(c)
                n += int((Xa[:, i] > Xsrc[:, i] + 1e-9).sum())
        return n
    while 0.0 < t < 1.0 and worse_cnt(Xg) > 0 and kappa < 1.0:
        kappa = 0.25 if kappa == 0 else kappa * 2
        w2 = (1 - kappa) * w + kappa * np.eye(len(cats))[i_dst]
        Xg = back(w2 @ MU_S_ARR + res)
        for c in ['DMS', 'X2.3.butanedione', 'X.E..2.nonenal', 'humulene.epoxide.II',
                  'X14.hydroxy.alpha.humulene', 'X.E..dihydro.apofarnesal']:
            if c in keep:
                i = keep.index(c)
                bad = Xg[:, i] > Xsrc[:, i]
                Xg[bad, i] = Xsrc[bad, i]
        w = w2
    guarded[t] = dict(X=Xg, kappa=kappa, nfix=nfix, w=w)
    for c, wc in zip(cats, w):
        rows.append({'t': t, 'style': c, 'weight': float(wc), 'kappa': float(kappa),
                     'n_guard_fix': int(sum(nfix.values()))})
P('## 2 路径权重')
P()
P('| t | κ | 权重前五 |')
P('| --- | --- | --- |')
for t in T_S:
    w = guarded[t]['w']
    top = ', '.join(f'{cats[i]} {w[i]:.3f}' for i in np.argsort(w)[::-1][:5])
    P(f"| {t:.2f} | {guarded[t]['kappa']:.2f} | {top} |")
P()

# 硬校验：t=1 必须与 02_transfer 的交付输出（已过缺陷守护）逐单元一致
Xref = back(MUS[DST][None, :] + res)
tr = pd.read_csv(os.path.join(OUT_Q3, '02_transfer.csv'))
ref_mat = tr[keep].to_numpy(float)
eps = float(np.max(np.abs(guarded[1.0]['X'] - np.clip(ref_mat, XLO, XHI))))
P('## 3 端点校验')
P()
P(f'- t=1 与 02_transfer 交付输出（已过缺陷守护）最大绝对差 = {eps:.2e}，'
  f'{"通过（<1e-8）" if eps < 1e-8 else "**未通过，需排查**"}')
P()

P('## 4 缺陷守护')
P()
P('| t | 守护压回的样本数合计 | κ |')
P('| --- | --- | --- |')
for t in T_S:
    P(f"| {t:.2f} | {int(sum(guarded[t]['nfix'].values()))} | {guarded[t]['kappa']:.2f} |")
P()

WT = pd.DataFrame(rows)
WT.to_csv(os.path.join(OUT_Q3, '03_path_weights.csv'), index=False, encoding='utf-8-sig')

# 变化最大的 14 维（沿 t 的均值振幅）
arr = np.stack([guarded[t]['X'].mean(0) for t in T_S])
amp = np.abs(arr - arr[0]).max(0)
top14 = np.argsort(amp)[::-1][:14]
fig, ax = plt.subplots(1, 2, figsize=(13.5, 5.6))
ax0 = ax[0]
ax0.add_patch(plt.Circle((0, 0), 1.0, fill=False, lw=0.8, color='k'))
# 标签沿「由中心向外的方向」偏移，避免 22 个风格名相互压盖
_ctr = Z2.mean(0)
for c, zc in zip(cats, Z2):
    ax0.scatter(zc[0], zc[1], s=30 + 4 * n_by[c], color=FAM_COLOR[FAMILY[c]],
                edgecolor='w', zorder=3)
    v = np.asarray(zc) - _ctr
    v = v / (np.linalg.norm(v) + 1e-9)
    ax0.annotate(c, (zc[0], zc[1]), fontsize=5.5, color='#333333', zorder=5,
                 xytext=(6 * v[0], 6 * v[1]), textcoords='offset points',
                 ha='left' if v[0] >= 0 else 'right')
# 测地线：原点切空间的直线映射回球内
tt = np.linspace(0, 1, 60)
geo = H.exp_map0(np.outer(1 - tt, t_u) + np.outer(tt, t_v), CURV)
ax0.plot(geo[:, 0], geo[:, 1], 'k-', lw=2.0, label=f'{SRC} → {DST}')
for t in T_S:
    zt = H.exp_map0((((1 - t) * t_u + t * t_v))[None, :], CURV)[0]
    ax0.scatter(zt[0], zt[1], marker='X', s=60, color='#d94801', zorder=4)
    ax0.annotate(f't={t:g}', (zt[0], zt[1]), fontsize=6.5, color='#d94801', zorder=6,
                 xytext=(6, -4 - 7 * T_S.index(t)), textcoords='offset points')
# 家族配色图例，说明颜色含义
for fam, col in FAM_COLOR.items():
    ax0.scatter([], [], s=28, color=col, label=fam)
ax0.set_xlim(-1.05, 1.05)
ax0.set_ylim(-1.05, 1.05)
ax0.set_aspect('equal')
ax0.set_title(f'Poincaré 盘（应力 双曲 {stress_hy:.3f} / 欧氏 {stress_eu:.3f}）')
ax0.legend(fontsize=7, loc='lower left')

xpos = np.arange(len(top14))
wid = 0.15
# 右图画相对 t=0 的变化量：绝对值会被两条量级大的列压扁，看不出趋势
delta = arr - arr[0][None, :]
for k, t in enumerate(T_S):
    ax[1].bar(xpos + (k - 2) * wid, delta[k, top14], wid, label=f't={t:g}')
ax[1].axhline(0.0, lw=0.8, color='k')
ax[1].set_xticks(xpos)
ax[1].set_xticklabels([keep[i] for i in top14], rotation=75, fontsize=6)
ax[1].set_ylabel('相对 t=0 的变化量（各列自身标度）')
ax[1].set_title('沿路径变化最大的 14 个化合物')
ax[1].legend(fontsize=7)
fig.tight_layout()
fig.savefig(os.path.join(OUT_Q3, '图_风格层级.png'), dpi=170)
fig.savefig(os.path.join(ANS, '图_风格层级.png'), dpi=170)
plt.close(fig)

with open(os.path.join(OUT_Q3, '03_hyperbolic_path.json'), 'w', encoding='utf-8') as f:
    json.dump({'stress_euclid': stress_eu, 'stress_hyperbolic': stress_hy, 'ratio': ratio,
               'use_hyperbolic': bool(use_hyp), 'h': h, 't': T_S,
               'kappa': {str(t): guarded[t]['kappa'] for t in T_S},
               't1_max_abs_diff': eps,
               'dims_changed': [keep[i] for i in top14]}, f, ensure_ascii=False, indent=2)
with open(os.path.join(OUT_Q3, '03_hyperbolic_path.md'), 'w', encoding='utf-8') as f:
    f.write('\n'.join(L))
print()
print('03 完成：权重表与图写入', OUT_Q3)

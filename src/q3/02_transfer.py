# -*- coding: utf-8 -*-
# Q3 步骤二：跨风格迁移（Pils/Lager 到 Tripel）与缺陷化合物守护
# 迁移机制：把真实啤酒在潜因子空间里的编码保留，只换风格均值，再解码回化合物谱
# 缺陷守护：DMS、双乙酰、4VG、反式-2-壬烯醛、酒花陈味标志物一旦上升就做单调截断
import importlib
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'q1'))
FE = importlib.import_module('03_features')
FM = importlib.import_module('final_model')
from common import (ANS, OUT_Q3, SEED, TARGET_COLS, chem_meta, ensure_dirs, load_chem, load_scale,
                    load_sensory, z_to_score)

ensure_dirs()
L = []


def P(s=''):
    L.append(str(s))
    print(s)


rng = np.random.default_rng(SEED)
chem = load_chem()
sens = load_sensory()
style = sens['tasting_category_fine'].to_numpy()
keep, _ = FE.prune_cols(chem)
X = chem[keep].to_numpy(float)
for j in range(X.shape[1]):
    c = X[:, j]
    if np.isnan(c).any():
        X[np.isnan(c), j] = np.nanmedian(c)
mu_x, sd_x = X.mean(0), X.std(0)
sd_x = np.where(sd_x == 0, 1.0, sd_x)
Z = (X - mu_x) / sd_x

d = np.load(os.path.join(OUT_Q3, '01_generative.npz'), allow_pickle=True)
cats = list(d['cats'])
LAM, SIG2, MU_STK = d['lam'], d['sig2'], d['mu_s']
MU_S = {c: MU_STK[i] for i, c in enumerate(cats)}
Q_DIM = LAM.shape[1]
back = lambda Zs: Zs * sd_x[None, :] + mu_x[None, :]

P('# Q3 步骤二：跨风格迁移与缺陷化合物守护')
P()
P(f'潜因子空间维数 q = {Q_DIM}，风格 {len(cats)} 类。')
P()

# 缺陷化合物：题目点名的五类不良风味
DEFECT = {
    'DMS（二甲基硫，煮玉米/蔬菜味）': ['DMS'],
    '双乙酰（黄油/奶油味）': ['X2.3.butanedione'],
    '4VG（4-乙烯基愈创木酚，丁香/药味）': ['X4VG'],
    '反式-2-壬烯醛（纸板/陈旧味）': ['X.E..2.nonenal'],
    '酒花陈味标志物（蛇麻烯环氧化物等）': ['humulene.epoxide.II', 'X14.hydroxy.alpha.humulene',
                                            'X.E..dihydro.apofarnesal'],
}
P('缺陷化合物集合（1.csv 字段名）：')
for k, v in DEFECT.items():
    P(f'- {k}: {", ".join(v)}')
P()
DEF_COLS = sorted({c for v in DEFECT.values() for c in v if c in chem.columns})
# 有些缺陷物质在 Q3 的裁剪口径里被剔除了，它们不在生成模型的支持集内，迁移不会触碰
DEF_IN_KEEP = [c for c in DEF_COLS if c in keep]
DEF_OUT = [c for c in DEF_COLS if c not in keep]
DEF_IDX = [keep.index(c) for c in DEF_IN_KEEP]
P(f'其中 {len(DEF_IN_KEEP)} 个在生成模型的支持集内，{len(DEF_OUT)} 个不在支持集内'
  f'（{", ".join(DEF_OUT) if DEF_OUT else "无"}）。')
P('不在支持集内的缺陷物质在迁移过程中根本不会被改写，因此「不恶化」由构造保证；')
P('在支持集内的用单调截断守护，超不过迁移前的值。')
P()
SENS_DEF = ['dms', 'diacetyl', 'X4vg', 't2n', 'stale_hops']

# 用问题一最终模型给生成样本打感官画像
pred = FM.fit_final(chem, sens)
P('已用问题一的最终模型（E0 与 E1 两套森林平均）作为感官画像的读出器。')
P()

GEN_SEN = {}
GEN_MEAN = {}
for c in cats:
    g = d[f'gen_{cats.index(c)}']
    df = pd.DataFrame(g, columns=keep)
    s_s, s_e, s_m = pred(df, new_style=c)
    GEN_SEN[c] = dict(score=s_s, exp=s_e, med=s_m)
    GEN_MEAN[c] = g.mean(0)
P('## 1 生成样本的感官画像')
P()
P('| 风格 | 生成样本数 | 生成画像均值（0-6 空间） | 真实画像均值（0-6 空间） | 画像相关系数 |')
P('| --- | --- | --- | --- | --- |')
# 真实画像必须换算到 0-6 分制空间，否则拿 4.csv 的 z 值和模型的分数直接比是两种量纲
MU, SG = load_scale()
RS = z_to_score(sens[TARGET_COLS].to_numpy(float), MU, SG)
real_sens = {c: RS[style == c].mean(0) for c in cats}
rows = []
for c in cats:
    gm = GEN_SEN[c]['score'].mean(0)
    rm = real_sens[c]
    r = np.corrcoef(gm, rm)[0, 1]
    rows.append(dict(style=c, n_gen=len(d[f'gen_{cats.index(c)}']), gen_mean=float(gm.mean()),
                     real_mean=float(rm.mean()), corr=float(r)))
    P(f'| {c} | {len(d[f"gen_{cats.index(c)}"])} | {gm.mean():.3f} | {rm.mean():.3f} | {r:.3f} |')
R = pd.DataFrame(rows)
P()
P(f'各风格生成画像与真实画像的相关系数均值 {R["corr"].mean():.3f}，最低 {R["corr"].min():.3f}（{R.loc[R["corr"].idxmin(), "style"]}）。')
P()

# 把生成样本的感官画像单独落盘，供答案汇总使用
gp = []
for c in cats:
    dfp = pd.DataFrame(GEN_SEN[c]['score'], columns=TARGET_COLS)
    dfp.insert(0, 'sample', np.arange(len(dfp)))
    dfp.insert(0, 'style', c)
    gp.append(dfp)
GP = pd.concat(gp, ignore_index=True)
GP.to_csv(os.path.join(OUT_Q3, '03_gen_portrait.csv'), index=False, encoding='utf-8-sig')

# 跨风格迁移：保留个体潜编码，只换风格均值
SRC, DST = 'Pils/Lager', 'Tripel'
src_idx = np.where(style == SRC)[0]
P(f'## 2 从 {SRC} 迁移到 {DST}')
P()
P(f'源风格真实样本 {len(src_idx)} 条。对每条样本反解潜变量 z = argmin ||Lambda z - (x - mu_src)||，')
P(f'再用目标风格均值重建 x_new = mu_dst + Lambda z，个体在风格内的差异被完整保留。')
P()
Zs = np.linalg.lstsq(LAM, (Z[src_idx] - MU_S[SRC][None, :]).T, rcond=None)[0].T
Xtr_new = back(MU_S[DST][None, :] + Zs @ LAM.T)
Xsrc = back(Z[src_idx])

# 缺陷守护：任何缺陷化合物超过原始值就压回原始值，取下层优先的单调截断
Xguard = Xtr_new.copy()
n_fix = {}
for c, i in zip(DEF_IN_KEEP, DEF_IDX):
    bad = Xguard[:, i] > Xsrc[:, i]
    n_fix[c] = int(bad.sum())
    Xguard[bad, i] = Xsrc[bad, i]
P('缺陷守护：逐条样本比较迁移前后同一缺陷化合物的 log10 浓度，若升高则压回原值（只会降不会升）。')
P()
P('| 缺陷化合物 | 未守护时升高的样本数 | 守护后升高的样本数 | 处理方式 |')
P('| --- | --- | --- | --- |')
for c in DEF_IN_KEEP:
    P(f'| {c} | {n_fix[c]} | 0 | 单调截断 |')
for c in DEF_OUT:
    P(f'| {c} | 0 | 0 | 不在生成模型支持集内，迁移不改写 |')
P()

# 迁移效果的化学侧核对
P('| 缺陷化合物 | 源风格均值（迁移前） | 迁移后均值 | 目标风格真实均值 | 是否劣于源 | 是否劣于目标 |')
P('| --- | --- | --- | --- | --- | --- |')
def_rows = []
for c in DEF_COLS:
    if c in keep:
        i = keep.index(c)
        a, b = float(Xsrc[:, i].mean()), float(Xguard[:, i].mean())
    else:
        a = b = float(np.nanmean(chem.loc[src_idx, c].to_numpy(float)))
    t = float(np.nanmean(chem.loc[style == DST, c].to_numpy(float)))
    def_rows.append(dict(compound=c, src=a, after=b, target=t,
                         worse_src=bool(b > a + 1e-9), worse_dst=bool(b > t + 1e-9)))
    P(f'| {c} | {a:.3f} | {b:.3f} | {t:.3f} | {"是" if b > a + 1e-9 else "否"} | '
      f'{"是" if b > t + 1e-9 else "否"} |')
P()
D = pd.DataFrame(def_rows)
P('注：表中数值是各列自身的标度（大多为 log10 浓度），数值越小表示该缺陷物质越少；')
P('不同化合物之间不可直接比较高低，只能比较同一行前后两列。')
P(f'{len(D)} 个缺陷化合物（覆盖题目列出的五类）里，{int((~D["worse_src"]).sum())}/{len(D)} 个在迁移后不高于迁移前，'
  f'{int((~D["worse_dst"]).sum())}/{len(D)} 个不高于目标风格的真实水平。')
P()

# 感官画像的迁移前后对照
df_src = pd.DataFrame(Xsrc, columns=keep)
df_new = pd.DataFrame(Xguard, columns=keep)
s_before, _, g_before = pred(df_src, new_style=SRC)
s_after, _, g_after = pred(df_new, new_style=DST)
real_dst_sens = RS[style == DST].mean(0)
P('## 3 感官画像的迁移效果')
P()
P('| 感官维度 | 迁移前（源风格） | 迁移后（目标风格） | 目标风格真实值 | 变化 |')
P('| --- | --- | --- | --- | --- |')
delta = s_after.mean(0) - s_before.mean(0)
order = np.argsort(-np.abs(delta))
for j in order[:12]:
    P(f'| {TARGET_COLS[j]} | {s_before[:, j].mean():.2f} | {s_after[:, j].mean():.2f} | '
      f'{real_dst_sens[j]:.2f} | {delta[j]:+.2f} |')
P()
P(f'迁移后 50 维画像与目标风格真实均值的相关系数 {np.corrcoef(s_after.mean(0), real_dst_sens)[0, 1]:.3f}，'
  f'迁移前为 {np.corrcoef(s_before.mean(0), real_dst_sens)[0, 1]:.3f}。')
P()
P('缺陷感官维度的迁移前后对照：')
P()
P('| 缺陷感官维度 | 迁移前 | 迁移后 | 目标风格真实值 |')
P('| --- | --- | --- | --- |')
for c in SENS_DEF:
    j = TARGET_COLS.index(c)
    P(f'| {c} | {s_before[:, j].mean():.2f} | {s_after[:, j].mean():.2f} | {real_dst_sens[j]:.2f} |')
P()

out = pd.DataFrame(Xguard, columns=keep)
out.insert(0, 'source_style', SRC)
out.insert(1, 'target_style', DST)
out.insert(2, 'source_beer', chem.loc[src_idx, 'beer'].to_numpy() if 'beer' in chem else src_idx)
out.to_csv(os.path.join(OUT_Q3, '02_transfer.csv'), index=False, encoding='utf-8-sig')
R.to_csv(os.path.join(OUT_Q3, '02_portrait_corr.csv'), index=False, encoding='utf-8-sig')
pd.DataFrame(s_before, columns=TARGET_COLS).to_csv(
    os.path.join(OUT_Q3, '02_transfer_sensory_before.csv'), index=False, encoding='utf-8-sig')
pd.DataFrame(s_after, columns=TARGET_COLS).to_csv(
    os.path.join(OUT_Q3, '02_transfer_sensory_after.csv'), index=False, encoding='utf-8-sig')

# 出图：缺陷化合物迁移前后箱线对照 + 感官画像迁移雷达式条形
fig, ax = plt.subplots(1, 2, figsize=(15, 6))
pos = np.arange(len(DEF_IN_KEEP))
ax[0].boxplot([Xsrc[:, keep.index(c)] for c in DEF_IN_KEEP], positions=pos - 0.22, widths=0.34,
              patch_artist=True, boxprops=dict(facecolor='#9ecae1'), medianprops=dict(color='k'))
ax[0].boxplot([Xguard[:, keep.index(c)] for c in DEF_IN_KEEP], positions=pos + 0.22, widths=0.34,
              patch_artist=True, boxprops=dict(facecolor='#f4a582'), medianprops=dict(color='k'))
ax[0].plot(pos, [float(np.nanmean(chem.loc[style == DST, c].to_numpy(float))) for c in DEF_IN_KEEP],
           'k*', ms=12, label=f'{DST} 真实均值')
ax[0].set_xticks(pos)
ax[0].set_xticklabels([c.replace('.', ' ') for c in DEF_IN_KEEP], rotation=25, ha='right', fontsize=8)
ax[0].set_ylabel('浓度（各列原标度）')
ax[0].set_title(f'缺陷化合物：{SRC}（蓝）迁移到 {DST}（橙）与守护后')
ax[0].legend(fontsize=8)
ax[0].grid(alpha=0.25, axis='y')

sel = list(np.argsort(-np.abs(delta))[:14])
w = 0.27
ax[1].barh(np.arange(len(sel)) - w, s_before.mean(0)[sel], height=w, label='迁移前')
ax[1].barh(np.arange(len(sel)), s_after.mean(0)[sel], height=w, label='迁移后')
ax[1].barh(np.arange(len(sel)) + w, real_dst_sens[sel], height=w, label='目标真实')
ax[1].set_yticks(np.arange(len(sel)))
ax[1].set_yticklabels([TARGET_COLS[j] for j in sel], fontsize=8)
ax[1].set_xlabel('0-6 分制得分')
ax[1].set_title('迁移前后与目标风格的感官画像（变化最大的 14 维）')
ax[1].legend(fontsize=8)
ax[1].grid(alpha=0.25, axis='x')
plt.tight_layout()
plt.savefig(os.path.join(OUT_Q3, 'fig02_transfer.png'), dpi=140)
plt.close('all')

with open(os.path.join(OUT_Q3, '02_transfer.md'), 'w', encoding='utf-8') as f:
    f.write('\n'.join(L))
print()
print('迁移结果已写入', OUT_Q3)

# -*- coding: utf-8 -*-
# 02 反标准化：把 4.csv 的标准化感官值反解回 0-6 分制
# 4.csv 每列已中心化但标准差中位数仅 0.5536，标准化的分母不是本列样本标准差
# 故 (mu, sigma) 在 4.csv 内部不可辨识，必须借助口径约定
# 主口径（补丁 §3，修 D14）：sigma_j = 该维标准差，mu_j = -sigma_j * z_min_j
# 另两口径只作并列披露：sigma=1（unit）与 min-max（把值域线性拉满 0-6）
# R2 在 z 空间逐维计算，与口径无关；MAE 随口径线性变化，报 MAE 必须声明口径
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (ATT, CORE_COLS, OUT, TARGET_COLS, calib_params, ensure_dirs,
                    load_sensory, z_to_score)

ensure_dirs()
L = []


# 追加一行到报告并打印
def P(s=''):
    L.append(str(s))
    print(s)


# 吸附残差：还原分到最近整数的平方距离均值，衡量 0-6 格点对观测值的解释力
def snap_resid(s):
    return float(np.mean((s - np.round(s)) ** 2))


# 网格搜索 (mu, sigma)：在完整覆盖约束下最小化吸附残差，仅作口径诊断用
def grid_search(z, mu_lo=0.2, mu_hi=5.8, sg_lo=0.3, sg_hi=3.5, step=0.05):
    mus = np.arange(mu_lo, mu_hi + 1e-9, step)
    sgs = np.arange(sg_lo, sg_hi + 1e-9, step)
    best = (None, None, np.inf)
    for mu in mus:
        s_all = mu + sgs[:, None] * z[None, :]
        ok = (s_all.min(axis=1) >= -1e-9) & (s_all.max(axis=1) <= 6 + 1e-9)
        ok &= (s_all.min(axis=1) <= 0.5) & (s_all.max(axis=1) >= 5.5)
        if not ok.any():
            continue
        idx = np.where(ok)[0]
        res = np.mean((s_all[idx] - np.round(s_all[idx])) ** 2, axis=1)
        j = int(np.argmin(res))
        if res[j] < best[2]:
            best = (float(mu), float(sgs[idx[j]]), float(res[j]))
    return best


sens = load_sensory()
Z = sens[TARGET_COLS].to_numpy(dtype=float)

# 三口径参数一次算出，主口径取 sd
MU_U, SG_U = calib_params(Z, 'unit')
MU_S, SG_S = calib_params(Z, 'sd')
MU_M, SG_M = calib_params(Z, 'minmax')
CAL = {'unit': (MU_U, SG_U), 'sd': (MU_S, SG_S), 'minmax': (MU_M, SG_M)}
S = {k: z_to_score(Z, v[0], v[1]) for k, v in CAL.items()}

P('# 02 反标准化：0-6 分制标度还原（三口径并列）')
P()
P('## 1 口径与依据')
P()
P('4.csv 的 50 列均值中位数 0.019、标准差中位数 0.5536，说明每列被中心化过，')
P('但标准化的分母不是本列样本标准差，故 (mu, sigma) 无法只用 4.csv 识别。')
P()
P('- 主口径 **sd**：sigma_j 取该维标准差，mu_j = -sigma_j * z_min_j，即最小观测值贴 0 分。')
P('  该口径不再假设「1 个标准差 = 1 分」，而是承认标度与列内离散度成比例，')
P('  是补丁 §3 指定的主口径（原 v1 用 sigma=1 并把论证写成「落在网格内」，该论证不成立，已修 D10）。')
P('- 并列口径 **unit**：sigma=1，mu=-z_min，等价于 z 直接线性搬到 0-6。')
P('- 并列口径 **minmax**：sigma=6/(z_max-z_min)，把训练集值域线性拉满 0-6。')
P()
P('三口径只影响连续分的绝对尺度，不影响 z 空间 R2；MAE 随口径同比例变化。')
P()

rows = []
for j, c in enumerate(TARGET_COLS):
    z = Z[:, j]
    zmin, zmax = float(z.min()), float(z.max())
    su, ss, sm = S['unit'][:, j], S['sd'][:, j], S['minmax'][:, j]
    gmu, gsg, gres = grid_search(z)
    rows.append({
        'dim': c, 'z_min': zmin, 'z_max': zmax, 'z_mean': float(z.mean()),
        'z_sd': float(z.std(ddof=1)),
        'mu': float(MU_S[j]), 'sigma': float(SG_S[j]),
        'mu_unit': float(MU_U[j]), 'sigma_unit': float(SG_U[j]),
        'mu_mm': float(MU_M[j]), 'sigma_mm': float(SG_M[j]),
        'raw_mean': float(ss.mean()), 'raw_std': float(ss.std(ddof=1)),
        'raw_min': float(ss.min()), 'raw_max': float(ss.max()),
        'snap_resid': snap_resid(ss),
        'unit_std': float(su.std(ddof=1)), 'unit_resid': snap_resid(su),
        'mm_std': float(sm.std(ddof=1)), 'mm_resid': snap_resid(sm),
        'grid_mu': gmu, 'grid_sigma': gsg, 'grid_resid': gres,
        'spearman_vs_z': float(spearmanr(z, ss).correlation),
    })

M = pd.DataFrame(rows)
P('## 2 逐维还原参数（主口径 sd）')
P()
P('| 维度 | z_min | z_max | z_sd | mu | sigma | 还原均值 | 还原标准差 | 吸附残差 | minmax sigma |')
P('| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |')
for _, r in M.iterrows():
    P(f"| {r['dim']} | {r['z_min']:.3f} | {r['z_max']:.3f} | {r['z_sd']:.3f} | {r['mu']:.3f} | "
      f"{r['sigma']:.3f} | {r['raw_mean']:.3f} | {r['raw_std']:.3f} | {r['snap_resid']:.4f} | "
      f"{r['sigma_mm']:.3f} |")
P()

P('## 3 三口径对照')
P()
P('| 口径 | sigma 定义 | 全表还原标准差均值 | 吸附残差均值 | 说明 |')
P('| --- | --- | --- | --- | --- |')
P(f"| sd（主） | 该维标准差 | {M['raw_std'].mean():.4f} | {M['snap_resid'].mean():.4f} | 交付口径 |")
P(f"| unit | 1 | {M['unit_std'].mean():.4f} | {M['unit_resid'].mean():.4f} | 敏感性对照 |")
P(f"| minmax | 6/(z_max-z_min) | {M['mm_std'].mean():.4f} | {M['mm_resid'].mean():.4f} | 敏感性对照 |")
P()
P(f"全部维度 Spearman(还原分, 标准化值)=1：{bool(np.allclose(M['spearman_vs_z'], 1.0))}")
P(f"主口径还原分取值范围 [{S['sd'].min():.3f}, {S['sd'].max():.3f}]（裁剪到 [0,6]）")
P()

G = np.clip(np.round(S['sd']), 0, 6).astype(int)
P('## 4 0-6 分级结果分布（主口径 sd）')
P()
P('| 维度 | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 均值 |')
P('| --- | --- | --- | --- | --- | --- | --- | --- | --- |')
for j, c in enumerate(TARGET_COLS):
    cnt = np.bincount(G[:, j], minlength=7)
    P(f"| {c} | " + ' | '.join(str(int(x)) for x in cnt) + f" | {G[:, j].mean():.2f} |")
P()
core_idx = [TARGET_COLS.index(c) for c in CORE_COLS]
P(f'全表分级均值 = {G.mean():.3f}，0 级占比 {np.mean(G==0):.3f}，6 级占比 {np.mean(G==6):.4f}')
P(f'核心 30 维（A_ + F_）分级均值 = {G[:, core_idx].mean():.3f}')
P()

# 与 7.csv 的独立感官数据做一致性抽查，说明还原方向正确
d7 = pd.read_csv(os.path.join(ATT, '7.csv'))
pair = [('A_malt_all', 'aroma_malt'), ('F_malt_all', 'taste_malt'),
        ('A_hops_all', 'aroma_hoppy'), ('bitternes', 'taste_bitter'),
        ('sweetness', 'taste_sweet'), ('dms', 'taste_corn'),
        ('diacetyl', 'taste_buttery'), ('lactic', 'taste_sour'),
        ('stale_hops', 'taste_stale'), ('t2n', 'aroma_cardboard')]
P('## 5 与 7.csv 独立描述词评分的相关校验')
P()
P('7.csv 是同一批啤酒的论坛描述词评分，独立于 4.csv，可用于验证还原结果的方向性。')
P()
P('| 4.csv 维度 | 7.csv 描述词 | Spearman |')
P('| --- | --- | --- |')
for a, b in pair:
    if b in d7.columns:
        r = spearmanr(sens[a], d7[b], nan_policy='omit').correlation
        P(f'| {a} | {b} | {r:.3f} |')
P()

M.to_csv(os.path.join(OUT, '02_destandardize.csv'), index=False, encoding='utf-8-sig')
gdf = sens[['beer', 'beer_id', 'tasting_category_fine']].copy()
gdf[TARGET_COLS] = G
gdf.to_csv(os.path.join(OUT, '02_target_grade.csv'), index=False, encoding='utf-8-sig')
sdf = sens[['beer', 'beer_id', 'tasting_category_fine']].copy()
sdf[TARGET_COLS] = np.round(S['sd'], 6)
sdf.to_csv(os.path.join(OUT, '02_target_score.csv'), index=False, encoding='utf-8-sig')

with open(os.path.join(OUT, '02_destandardize.json'), 'w', encoding='utf-8') as f:
    json.dump({'mode': 'sd 主口径，mu=-sigma*z_min；并列 unit / minmax',
               'calib': {k: {'mu': v[0].tolist(), 'sigma': v[1].tolist()} for k, v in CAL.items()},
               'dims': rows,
               'grade_mean': float(G.mean()),
               'core_grade_mean': float(G[:, core_idx].mean()),
               'mae_scale': {k: float(M[m].mean()) for k, m in
                             (('sd', 'raw_std'), ('unit', 'unit_std'), ('minmax', 'mm_std'))}},
              f, ensure_ascii=False, indent=2)

with open(os.path.join(OUT, '02_destandardize.md'), 'w', encoding='utf-8') as f:
    f.write('\n'.join(L))
print()
print('反标准化结果已写入', OUT)

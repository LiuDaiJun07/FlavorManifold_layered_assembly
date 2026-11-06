# -*- coding: utf-8 -*-
# 审查用分析：标度口径对 MAE 的影响、双通道信度给出的 R2 上限、真实可达性
import importlib
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

HERE = os.path.dirname(os.path.abspath(__file__))
Q1 = os.path.join(os.path.dirname(HERE), 'q1')
sys.path.insert(0, Q1)
from common import CORE_COLS, OUT, TARGET_COLS, load_sensory, macro_r2

sens = load_sensory()
Y = sens[TARGET_COLS].to_numpy(float)
core_idx = [TARGET_COLS.index(c) for c in CORE_COLS]
n, m = Y.shape
S1 = pd.read_csv(os.path.join(OUT, '02_destandardize.csv')).set_index('dim').loc[TARGET_COLS]
sd = Y.std(0, ddof=1)
sd_mm = S1['mm_sigma'].to_numpy(float)
sd_one = np.ones(m)
print('列标准差 sd_j: 中位 %.4f 均值 %.4f' % (np.median(sd), sd.mean()))
print('min-max 口径 sigma_j: 均值 %.4f 中位 %.4f' % (sd_mm.mean(), np.median(sd_mm)))

P1 = np.load(os.path.join(HERE, 'probe_pred.npz'))
mem = ['A1_multi_norm_mf0.5', 'A4_pca30_norm', 'A6_perdim_mf0.5_top60',
       'A7_perdim_mf1.0_top60', 'A8_perdim_rf0.5_top60']
R2m = np.array([macro_r2(Y, P1[k]) for k in mem])
W = np.clip(R2m, 1e-3, None)
W = W / W.sum(0, keepdims=True)
fuse = np.sum([P1[k] * W[i][None, :] for i, k in enumerate(mem)], axis=0)
P2 = np.load(os.path.join(HERE, 'probe2_pred.npz')) if os.path.exists(
    os.path.join(HERE, 'probe2_pred.npz')) else None
sets = {'A0_baseline': P1['A0_multi_raw_mf0.5'],
        'A3_norm_mf0.8': P1['A3_multi_norm_mf0.8'],
        'A4_pca30': P1['A4_pca30_norm'],
        'A6_perdim_top60': P1['A6_perdim_mf0.5_top60'],
        'A8_rf_top60': P1['A8_perdim_rf0.5_top60'],
        'F3_all5_r2w': fuse}
if P2 is not None:
    for k in P2.files:
        sets[k] = P2[k]
# 序数分类的输出是等级单位，这里按 sigma=1 口径换回 z 空间再算 R2
if 'B2_ordinal_et' in sets:
    zmin = S1['z_min'].to_numpy(float)
    sets['B2_ordinal_rescaled'] = sets['B2_ordinal_et'] - zmin[None, :]

rows = []
for k, P in sets.items():
    if P.shape != Y.shape:
        continue
    err = np.abs(P - Y)
    r2 = macro_r2(Y, P)
    rows.append(dict(tag=k, macro_r2=float(r2.mean()), core_r2=float(r2[core_idx].mean()),
                     mae_sig1=float(err.mean()),
                     mae_sigSd=float((err * sd[None, :]).mean()),
                     mae_sigMM=float((err * sd_mm[None, :]).mean()),
                     n_dim_below_0p1=int((r2 < 0.10).sum())))
T = pd.DataFrame(rows).sort_values('macro_r2', ascending=False)
print(T.to_string(index=False, float_format=lambda v: f'{v:.4f}'))
T.to_csv(os.path.join(HERE, 'analyze_scale.csv'), index=False, encoding='utf-8-sig')

# 双通道信度：A_ 与 F_ 同名维度的相关，作为单通道测量可靠度的估计，进而给出 R2 的上限
best = T.iloc[0]['tag']
P = sets[best]
r2 = macro_r2(Y, P)
rel = []
for c in CORE_COLS:
    if not c.startswith('A_'):
        continue
    f = 'F_' + c[2:]
    if f not in TARGET_COLS:
        continue
    a, b = TARGET_COLS.index(c), TARGET_COLS.index(f)
    r = float(pearsonr(Y[:, a], Y[:, b])[0])
    rel.append(dict(concept=c[2:], r_AF=r, r2_A=float(r2[a]), r2_F=float(r2[b]),
                    ratio_A=float(r2[a] / r) if r > 0 else np.nan,
                    ratio_F=float(r2[b] / r) if r > 0 else np.nan))
R = pd.DataFrame(rel)
print('\n===== 双通道信度与已实现的 R2 占比（预测集 %s）=====' % best)
print(R.to_string(index=False, float_format=lambda v: f'{v:.3f}'))
print('平均 r_AF = %.4f，A 侧 R2/信度 = %.3f，F 侧 = %.3f'
      % (R['r_AF'].mean(), R['ratio_A'].mean(), R['ratio_F'].mean()))
R.to_csv(os.path.join(HERE, 'analyze_reliability.csv'), index=False, encoding='utf-8-sig')

# 若要 MAE(sigma=1) 达到 0.30，在给定残差分布下需要的 R2 水平
for r2t in (0.30, 0.40, 0.50):
    resid = np.sqrt(1 - r2t) * sd.mean()
    print(f'R2={r2t:.2f} 时残差标准差≈{resid:.3f}，高斯近似下 MAE(sigma=1) 下界≈{0.7979 * resid:.3f}')
json.dump(dict(table=T.to_dict('records'), reliability=R.to_dict('records')),
          open(os.path.join(HERE, 'analyze.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('完成')

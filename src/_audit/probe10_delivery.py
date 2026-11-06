# -*- coding: utf-8 -*-
# 交付核对：新答案是否等于「五成员直方图等权混合 + 归一化中位数」；与旧答案逐格对比
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'q1'))
from common import ANS, TARGET_COLS, load_sensory_grade, median_grade  # noqa: E402

ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
OUT = os.path.join(ROOT, 'main', 'Output', 'Q1')
OLD = os.path.join(ROOT, 'temp', '_old_Q1', '答案')

Z = np.load(os.path.join(OUT, '06b_members.npz'), allow_pickle=False)
H = {k: Z[f'{k}_hist'] for k in ('M1', 'M2', 'M3', 'M4', 'M5')}
for k, h in H.items():
    print(f'{k}_hist 行和 min={h.sum(axis=2).min():.6f} max={h.sum(axis=2).max():.6f}')
mix = np.mean([H[k] for k in ('M1', 'M2', 'M3', 'M4', 'M5')], axis=0)
print(f'hist_mix 行和 min={mix.sum(axis=2).min():.6f} max={mix.sum(axis=2).max():.6f}')

G = load_sensory_grade()
m_med = median_grade(mix)
exp = np.tensordot(mix, np.arange(7, dtype=float), axes=([2], [0]))
m_exp = np.clip(np.round(exp), 0, 6).astype(int)
print(f'重算：中位数命中率={np.mean(m_med == G):.4f} 期望命中率={np.mean(m_exp == G):.4f}')

new = pd.read_csv(os.path.join(ANS, '问题1_感官评分_分级0到6.csv'))
old = pd.read_csv(os.path.join(OLD, '问题1_感官评分_分级0到6.csv'))
A = new[TARGET_COLS].to_numpy(dtype=int)
B = old[TARGET_COLS].to_numpy(dtype=int)
print(f'新答案 vs 重算中位数：完全一致 = {np.array_equal(A, m_med.astype(int))}')
print(f'新答案命中率 = {np.mean(A == G):.4f}；旧答案命中率 = {np.mean(B == G):.4f}')
print(f'新旧逐格不同的格点数 = {int((A != B).sum())} / {A.size}')
print(f'新答案等级分布 = {np.bincount(A.ravel(), minlength=7).tolist()}')
print(f'旧答案等级分布 = {np.bincount(B.ravel(), minlength=7).tolist()}')
print(f'真实等级分布   = {np.bincount(G.ravel(), minlength=7).tolist()}')
print(f'新答案均值 = {A.mean():.4f}；旧答案均值 = {B.mean():.4f}；真实均值 = {G.mean():.4f}')
print(f'新答案平均绝对偏差 = {np.abs(A - G).mean():.4f}；旧答案 = {np.abs(B - G).mean():.4f}')

cs = pd.read_csv(os.path.join(ANS, '问题1_感官评分_连续分.csv'))
C = cs[TARGET_COLS].to_numpy(dtype=float)
print(f'连续分范围 = [{C.min():.2f}, {C.max():.2f}]，行数 = {len(C)}')
print(f'连续分与分级的一致性（round 后等于分级）= {np.array_equal(np.clip(np.round(C), 0, 6).astype(int), A)}')

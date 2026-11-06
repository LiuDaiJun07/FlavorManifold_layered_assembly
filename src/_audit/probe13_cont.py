# -*- coding: utf-8 -*-
# 连续分列的口径诊断：量化下限、与真实连续分的距离、几种连续输出的 MAE 对照
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'q1'))
from common import OUT, TARGET_COLS, load_sensory_grade  # noqa: E402

G = load_sensory_grade()
E = np.load(os.path.join(OUT, '06e_final.npz'))
B = np.load(os.path.join(OUT, '06b_members.npz'))
Y = E['y']
F = E['final']
fold = E['fold_id']
hist = E['hist']

# 真实连续分（同一主口径仿射映射），逐折用该折 mu/sg
C_true = np.zeros_like(Y)
for f in np.unique(fold):
    te = np.where(fold == f)[0]
    mu = B[f'mu_{f}']
    sg = B[f'sg_{f}']
    C_true[te] = np.clip(mu[None, :] + sg[None, :] * Y[te], 0, 6)

pred = np.clip(C_true + (F - Y) * 0, 0, 6)  # 占位，避免误用
# 交付的连续分
import pandas as pd  # noqa: E402

C_del = pd.read_csv(os.path.join(os.path.dirname(HERE), '..', '..', '答案', '问题1_感官评分_连续分.csv'),
                    encoding='utf-8-sig')[TARGET_COLS].to_numpy(float)

# 期望口径（hist 均值）
W = np.arange(7, dtype=float)
C_exp = np.tensordot(hist, W, axes=([2], [0]))

floor = float(np.abs(C_true - G).mean())
print('== 连续分列的口径 ==')
print('真实连续分 vs 整数等级 G 的量化下限 = %.4f（任何模型都绕不过）' % floor)
print('交付连续分 vs 真实连续分  MAE = %.4f' % float(np.abs(C_del - C_true).mean()))
print('交付连续分 vs 整数 G      MAE = %.4f' % float(np.abs(C_del - G).mean()))
print('交付连续分 相关系数(r)    = %.4f' % float(np.corrcoef(C_del.ravel(), C_true.ravel())[0, 1]))
print()
print('== 三种连续输出的 MAE（都对整数 G）==')
print('当前交付（sd 映射五成员基座）  MAE = %.4f' % float(np.abs(C_del - G).mean()))
print('直方图期望（Σk·P(k)）         MAE = %.4f' % float(np.abs(C_exp - G).mean()))
print('中位数等级（现分级列）        MAE = %.4f' % float(np.abs(np.rint(C_exp * 0 + E['grade']) - G).mean()))
print()
print('交付连续分分布：min %.3f / 中位 %.3f / 均值 %.3f / max %.3f' % (
    C_del.min(), np.median(C_del), C_del.mean(), C_del.max()))
print('真实连续分分布：min %.3f / 中位 %.3f / 均值 %.3f / max %.3f' % (
    C_true.min(), np.median(C_true), C_true.mean(), C_true.max()))

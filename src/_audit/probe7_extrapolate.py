# -*- coding: utf-8 -*-
# 审查用探针 7：把"数据到头了没有"算成一个数——学习曲线外推 + 跨家族残差相关
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
import json

RES = json.load(open(os.path.join(HERE, 'probe6.json'), encoding='utf-8'))
n = np.array([r[0] for r in RES['learning_curve']], float)
r2 = np.array([r[1] for r in RES['learning_curve']], float)

# 三种学习曲线形式各自外推渐近值，看它们是否一致
fits = {}
# 形式一：a - b/n
u = 1.0 / n
b1, a1 = np.polyfit(u, r2, 1)
fits['a-b/n'] = dict(a=float(a1), b=float(-b1), r2_at_250=float(a1 + b1 / 250))
# 形式二：a - b/n^0.5
u = n ** -0.5
b2, a2 = np.polyfit(u, r2, 1)
fits['a-b/n^0.5'] = dict(a=float(a2), b=float(-b2), r2_at_250=float(a2 + b2 / 250 ** 0.5))
# 形式三：只用后三个点做 a - b/n（保守）
u3 = 1.0 / n[-3:]
b3, a3 = np.polyfit(u3, r2[-3:], 1)
fits['a-b/n(last3)'] = dict(a=float(a3), b=float(-b3), r2_at_250=float(a3 + b3 / 250))

print('学习曲线外推（渐近线 = 特征集在这些样本量下的信息上限估计）：', flush=True)
for k, v in fits.items():
    print(f'  {k:14s} 渐近 {v["a"]:.4f}  n=250 预测 {v["r2_at_250"]:.4f}', flush=True)
# 达到 0.30 需要多少瓶（按 a-b/n 形式）
a, b = fits['a-b/n']['a'], fits['a-b/n']['b']
if a > 0.30:
    print(f'  按 a-b/n 外推：R2 要到 0.30 需要 {b / (a - 0.30):.0f} 瓶啤酒', flush=True)
else:
    print('  按 a-b/n 外推：渐近线本身低于 0.30，加样本也到不了', flush=True)
# 最后一段的局部斜率
slope = (r2[-1] - r2[-2]) / (n[-1] - n[-2])
print(f'  最后一段局部斜率：每多 60 瓶 +{slope * 60:.4f}', flush=True)

# 跨家族残差相关：共同漏掉的部分占多大
import importlib
import sys
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'q1'))
from common import TARGET_COLS, load_sensory
Y = load_sensory()[TARGET_COLS].to_numpy(float)
sets = {}
for f, keys in (('probe_pred.npz', ['A0_multi_raw_mf0.5', 'A5_perdim_mf0.5_all', 'A8_perdim_rf0.5_top60']),
                ('probe2_pred.npz', ['B1_gbr_perdim', 'B3_chain_et']),
                ('probe3_pred.npz', ['K_knn10', 'T3_thr_all']),
                ('probe4_pred.npz', ['10f_M1'])):
    p = os.path.join(HERE, f)
    z = np.load(p)
    have = list(z.files)
    for k in keys:
        if k in have:
            sets[k] = z[k]
names = list(sets)
print('\n残差相关矩阵（跨家族）：', flush=True)
print('  ' + ' ' * 24 + ' '.join(f'{i:>7d}' for i in range(len(names))), flush=True)
C = np.eye(len(names))
for i in range(len(names)):
    for j in range(i + 1, len(names)):
        r1 = (Y - sets[names[i]]).ravel()
        r2v = (Y - sets[names[j]]).ravel()
        C[i, j] = C[j, i] = float(np.corrcoef(r1, r2v)[0, 1])
for i, nm in enumerate(names):
    print(f'  {i} {nm:23s} ' + ' '.join(f'{C[i, j]:7.3f}' for j in range(len(names))), flush=True)
json.dump({'fits': fits, 'names': names, 'C': C.tolist()},
          open(os.path.join(HERE, 'probe7.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('完成', flush=True)

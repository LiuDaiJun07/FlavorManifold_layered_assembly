# -*- coding: utf-8 -*-
# 修复前(temp/_old_Q1/答案) 与 修复后(答案) 两版交付的逐格对比
# 口径说明：G 是 02 用主口径仿射映射从 4.csv 反推的 0-6 整数等级，两版都用它当参照
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'q1'))
from common import ANS, TARGET_COLS, load_sensory_grade  # noqa: E402

ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
OLD = os.path.join(ROOT, 'temp', '_old_Q1', '答案')
OUT = os.path.join(ROOT, 'main', 'Output', 'Q1')
G = load_sensory_grade()


# 读一份交付：返回 (分级, 连续分)
def load(d):
    g = pd.read_csv(os.path.join(d, '问题1_感官评分_分级0到6.csv'))[TARGET_COLS].to_numpy(int)
    c = pd.read_csv(os.path.join(d, '问题1_感官评分_连续分.csv'))[TARGET_COLS].to_numpy(float)
    return g, c


g_old, c_old = load(OLD)
g_new, c_new = load(ANS)


# 一版交付的全部指标
def stat(name, g, c):
    return {
        'name': name,
        'mae_grade': float(np.abs(g - G).mean()),
        'rmse_grade': float(np.sqrt(((g - G) ** 2).mean())),
        'mae_cont': float(np.abs(c - G).mean()),
        'hit': float((g == G).mean()),
        'hit1': float((np.abs(g - G) <= 1).mean()),
        'bias': float(g.mean() - G.mean()),
        'pred_mean': float(g.mean()),
        'cont_span': (float(c.min()), float(c.max())),
    }


rows = [stat('修复前', g_old, c_old), stat('修复后', g_new, c_new)]
T = pd.DataFrame(rows)
print('== 交付对照（参照 = 02 构造的 0-6 整数等级 G，均值 %.4f）==' % G.mean())
for _, r in T.iterrows():
    print(f"{r['name']}: 等级MAE={r['mae_grade']:.4f} 等级RMSE={r['rmse_grade']:.4f} "
          f"连续分MAE={r['mae_cont']:.4f} 命中率={r['hit']:.4f} 偏差≤1={r['hit1']:.4f} "
          f"系统偏差={r['bias']:+.4f} 预测均值={r['pred_mean']:.4f} "
          f"连续分跨度=[{r['cont_span'][0]:.2f},{r['cont_span'][1]:.2f}]")

print()
print('== 两版之间的差别 ==')
print('分级列不同的格点 =', int((g_new != g_old).sum()), '/', g_new.size,
      '=%.2f%%' % (100 * float((g_new != g_old).mean())))
print('连续分列最大绝对差 = %.6f' % float(np.abs(c_new - c_old).max()))
print('连续分列是否逐格相同 =', bool(np.allclose(c_new, c_old, atol=1e-9)))

# 逐格胜负：谁离真实更近
e_old = np.abs(g_old - G)
e_new = np.abs(g_new - G)
print()
print('== 逐格胜负（等级列）==')
print('修复后更近的格点 =', int((e_new < e_old).sum()))
print('修复前更近的格点 =', int((e_old < e_new).sum()))
print('两者一样 =', int((e_new == e_old).sum()))
print('平均绝对误差下降 = %.4f -> %.4f（降 %.1f%%）' % (
    e_old.mean(), e_new.mean(), 100 * (1 - e_new.mean() / e_old.mean())))

# 逐维：改善了多少维
per_old = e_old.mean(axis=0)
per_new = e_new.mean(axis=0)
better = np.where(per_new < per_old - 1e-9)[0]
worse = np.where(per_new > per_old + 1e-9)[0]
print()
print('== 逐维（50 维）==')
print('改善的维度 =', len(better), '变差的维度 =', len(worse), '不变的 =', 50 - len(better) - len(worse))
if len(worse):
    print('变差的维度：', ', '.join(f'{TARGET_COLS[j]}({per_old[j]:.3f}->{per_new[j]:.3f})' for j in worse))
order = np.argsort(per_old - per_new)[::-1][:10]
print('改善最大的 10 维：')
for j in order:
    print(f'  {TARGET_COLS[j]:<20s} MAE {per_old[j]:.3f} -> {per_new[j]:.3f}  '
          f'真实均值 {G[:, j].mean():.2f}')

# 连续分口径：z 空间 R2 与交付一致性
Z = np.load(os.path.join(OUT, '06e_final.npz'), allow_pickle=False)
Y = Z['y']
F = Z['final']
r2 = 1.0 - ((Y - F) ** 2).sum(axis=0) / ((Y - Y.mean(axis=0)) ** 2).sum(axis=0)
print()
print('== z 空间（两版完全相同，因为交付列都退化为五成员基座）==')
print('宏 R2 = %.4f，逐维范围 [%.4f, %.4f]' % (r2.mean(), r2.min(), r2.max()))
print('z 空间 MAE = %.4f（标准化单位）' % float(np.abs(Y - F).mean()))

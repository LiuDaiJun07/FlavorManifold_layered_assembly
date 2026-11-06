# -*- coding: utf-8 -*-
# 07 消融矩阵汇总（适配 v2 产物）
# 行：E0/E1/E2/E5（单模型）+ E3/E4（二层）+ 五成员基座 + LSF + ZITS + 最终
# 三条门禁与 G3 两列对照同时打印，避免「只挑好看的报」
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import (CORE_COLS, OUT, TARGET_COLS, ensure_dirs, load_scale, r2_groups)

ensure_dirs()
L = []


# 追加一行到报告并打印
def P(s=''):
    L.append(str(s))
    print(s)


M5 = json.load(open(os.path.join(OUT, '05_metrics.json'), encoding='utf-8'))
d5 = np.load(os.path.join(OUT, '05_oof.npz'), allow_pickle=False)
d6 = np.load(os.path.join(OUT, '06_stack.npz'), allow_pickle=False)
d7 = np.load(os.path.join(OUT, '06b_members.npz'), allow_pickle=False)
d8 = np.load(os.path.join(OUT, '06c_lsf.npz'), allow_pickle=False)
d9 = np.load(os.path.join(OUT, '06e_final.npz'), allow_pickle=False)
J9 = json.load(open(os.path.join(OUT, '06e_final.json'), encoding='utf-8'))
Y = d5['y']
core_idx = [TARGET_COLS.index(c) for c in CORE_COLS]

rows = []
for k in ['E0', 'E1', 'E2', 'E5']:
    # 单模型指标直接从折外预测重算，避免依赖 05_metrics.json 的字段结构
    v = d5[f'{k}_pred']
    mac, weak, strong, r2 = r2_groups(Y, v)
    rows.append({'config': k, 'n_features': len(d5['keep']), 'macro_r2': mac,
                 'weak_r2': weak, 'strong_r2': strong,
                 'core_r2': float(np.mean(r2[core_idx])), 'mae_sd': np.nan})
for k in ['E3', 'E4']:
    v = d6[f'oof_{k}']
    mac, weak, strong, _ = r2_groups(Y, v)
    rows.append({'config': k, 'n_features': len(d5['keep']), 'macro_r2': mac,
                 'weak_r2': weak, 'strong_r2': strong,
                 'core_r2': float(np.mean(r2_groups(Y, v)[3][core_idx])), 'mae_sd': np.nan})
cand = [('BASE5', d7['base']), ('LSF_sel', d8['lsf_sel']), ('LSF_def', d8['lsf_def']),
        ('ZITS', d9['zits']), ('FINAL', d9['final'])]
for k, v in cand:
    mac, weak, strong, r2 = r2_groups(Y, v)
    rows.append({'config': k, 'n_features': len(d5['keep']), 'macro_r2': mac, 'weak_r2': weak,
                 'strong_r2': strong, 'core_r2': float(np.mean(r2[core_idx])), 'mae_sd': np.nan})
T = pd.DataFrame(rows)
T.to_csv(os.path.join(OUT, '07_ablation.csv'), index=False, encoding='utf-8-sig')

P('# 07 消融矩阵')
P()
P('| 配置 | 特征数 | 宏 R2 | 12 弱维 | 5 强维 | 30 核心维 |')
P('| --- | --- | --- | --- | --- | --- |')
for _, r in T.iterrows():
    P(f"| {r['config']} | {int(r['n_features'])} | {r['macro_r2']:.4f} | {r['weak_r2']:.4f} | "
      f"{r['strong_r2']:.4f} | {r['core_r2']:.4f} |")
P()
P(f"- G3 两列宏 R2 差 = {J9['g3_gap']:.4f}；采用列 = {J9['authority']}")
P(f"- 门禁：宏 R2 {J9['macro']:.4f}（≥0.274？{J9['gates']['macro']}）、"
  f"12 弱维 {J9['weak']:.4f}（≥0.075？{J9['gates']['weak']}）、"
  f"5 强维下降是否 ≤0.005：{J9['gates']['strong']}")
P()

fig, ax = plt.subplots(figsize=(9, 4.6))
cols = ['#bdbdbd'] * 6 + ['#6baed6', '#4292c6', '#08519c', '#41ab5d', '#d94801']
ax.bar(np.arange(len(T)), T['macro_r2'], color=cols[:len(T)])
ax.set_xticks(np.arange(len(T)))
ax.set_xticklabels(T['config'], rotation=20)
ax.axhline(0.27, color='r', ls='--', lw=0.8, label='旧硬门槛 0.27')
ax.axhline(0.274, color='g', ls=':', lw=0.8, label='新门禁 0.274')
ax.set_ylabel('宏 R²（z 空间逐维平均）')
ax.set_title('消融矩阵')
ax.legend(fontsize=7)
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'fig07_ablation.png'), dpi=170)
plt.close(fig)
with open(os.path.join(OUT, '07_ablation.md'), 'w', encoding='utf-8') as f:
    f.write('\n'.join(L))
print('07 完成')

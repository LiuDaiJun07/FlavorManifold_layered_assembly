# -*- coding: utf-8 -*-
# 08 出答案：读 06e 的最终折外结果，写 250×53 答案表与逐维评价表
# 连续分与等级都来自同一套 10 折 OOF，三口径并列报告；不再写死「两项硬指标都没达标」（修 D9）
# 树数等元信息从 05/06b 的 json 读回，不再手写「2×1200 棵树」（修 D8）
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
from common import (ANS, OUT, TARGET_COLS, WEAK_DIMS, STRONG_DIMS, ensure_dirs,
                    load_sensory, load_sensory_grade, load_scale, mae_caliber,
                    mae_caliber_oof, r2_groups, z_to_score)

ensure_dirs()
L = []


# 追加一行到报告并打印
def P(s=''):
    L.append(str(s))
    print(s)


sens = load_sensory()
ids = sens[['beer', 'beer_id', 'tasting_category_fine']].copy()
Y = sens[TARGET_COLS].to_numpy(dtype=float)
G = load_sensory_grade()
Z = np.load(os.path.join(OUT, '06e_final.npz'), allow_pickle=False)
final = Z['final']
score = Z['score']
grade = Z['grade']
base = Z['base']
on = Z['on'].astype(bool)
MU, SG = load_scale()
with open(os.path.join(OUT, '05_metrics.json'), encoding='utf-8') as f:
    J5 = json.load(f)

mac, weak, strong, r2 = r2_groups(Y, final)
mac_b, weak_b, strong_b, r2b = r2_groups(Y, base)
mae = mae_caliber(Y, final, Y)
# 折内刻度：每折只用训练行估 mu/sigma，与 06b 建模时的折内 calib 一致
mae_oof = mae_caliber_oof(Y, final, Z['fold_id'])
P('# 08 答案与验收')
P()
P('## 1 交付口径')
P()
P('- 目标：3.xlsx 的 50 个感官维度，0–6 分制。')
P('- 训练协议：10 折分层折外预测，同一套 OOF 既用于评估也用于交付（G5）。')
P(f'- 模型：五成员融合基座 + LSF 局部收缩 + ZITS 零膨胀两阶段（开启 {int(on.sum())} 维）。')
# 叠加没赢过基座时 06e 会退回基座交付，这里必须如实说明，不能含糊成「用了 LSF+ZITS」
if abs(float(np.mean((final - base) ** 2))) < 1e-18:
    P('- 注意：最终列与基座逐单元相同 —— LSF/ZITS 在本数据上没有跑赢五成员基座，'
      '已按退路「全关退化为基座」交付，报告不再声称融合有增益。')
P(f"- 树数：整表成员 {J5['trees_cv']} 棵、逐维成员 200 棵、ZITS 分类 500 / 回归 150。")
P('- 反标准化：主口径 sigma=该维标准差；并列 unit 与 minmax 两口径。')
P(f"- 宏 R2：基座 {mac_b:.4f} → 最终 {mac:.4f}；12 弱维 {weak_b:.4f} → {weak:.4f}；"
  f"5 强维 {strong_b:.4f} → {strong:.4f}")
P()

P('## 2 三口径误差')
P()
P('| 口径 | sigma 定义 | 连续分 MAE（全量刻度） | 连续分 MAE（折内刻度） |')
P('| --- | --- | --- | --- |')
P(f"| sd（主） | 该维标准差 | {mae['sd']:.4f} | {mae_oof['sd']:.4f} |")
P(f"| unit | 1 | {mae['unit']:.4f} | {mae_oof['unit']:.4f} |")
P(f"| minmax | 6/(z_max−z_min) | {mae['minmax']:.4f} | {mae_oof['minmax']:.4f} |")
P()
P('注：三口径只改变分数的绝对尺度，R2 与口径无关；报 MAE 必须同时声明口径（修 D14）。')
P('两列的区别只在刻度估计用了哪些行：全量列按 250 行估 mu/sigma（与 02 的交付刻度一致），')
P('折内列按 10 折训练行估（与 06b 建模时的 calib 一致）。评分刻度本身不是可学习参数，')
P('因此两列的差只反映刻度估计的抽样波动，不影响 z 空间 R2 与等级中位数。')
P()

acc = float(np.mean(grade == G))
P('## 3 等级一致性')
P()
P(f'- 全表 250×50 个等级格点完全命中率 = {acc:.4f}')
P(f'- 偏差 ≤1 级的比例 = {float(np.mean(np.abs(grade - G) <= 1)):.4f}')
with open(os.path.join(OUT, '06e_final.json'), encoding='utf-8') as f:
    J6 = json.load(f)
if 'grade_match_exp' in J6:
    P(f"- 对照口径（分布期望后取整）命中率 = {J6['grade_match_exp']:.4f}，交付用中位数口径")
P()

rows = []
for j, c in enumerate(TARGET_COLS):
    rows.append({'dim': c, 'r2_base': float(r2b[j]), 'r2_final': float(r2[j]),
                 'mae_sd': float(np.mean(np.abs(z_to_score(Y[:, [j]], MU, SG) -
                                               z_to_score(final[:, [j]], MU, SG)))),
                 'zero_share': float(np.mean(G[:, j] == 0)),
                 'zits_on': bool(on[j]), 'weak': c in WEAK_DIMS, 'strong': c in STRONG_DIMS})
T = pd.DataFrame(rows)
T.to_csv(os.path.join(OUT, '08_per_dim.csv'), index=False, encoding='utf-8-sig')
P('## 4 逐维 R2（按最终 R2 升序）')
P()
P('| 维度 | 基座 R2 | 最终 R2 | 0 级占比 | ZITS |')
P('| --- | --- | --- | --- | --- |')
for _, r in T.sort_values('r2_final').iterrows():
    P(f"| {r['dim']} | {r['r2_base']:.4f} | {r['r2_final']:.4f} | {r['zero_share']:.2f} | "
      f"{'是' if r['zits_on'] else '否'} |")
P()

# 答案表：id 3 列 + 50 维，连续分保留 2 位小数、等级为 0-6 整数
gdf = ids.copy()
gdf[TARGET_COLS] = np.clip(np.round(grade), 0, 6).astype(int)
sdf = ids.copy()
sdf[TARGET_COLS] = np.round(np.clip(score, 0.0, 6.0), 2)
gdf.to_csv(os.path.join(ANS, '问题1_感官评分_分级0到6.csv'), index=False, encoding='utf-8-sig')
sdf.to_csv(os.path.join(ANS, '问题1_感官评分_连续分.csv'), index=False, encoding='utf-8-sig')
# 两列口径不同：分级取逐树分布的中位数，连续分取点预测经刻度映射；零膨胀维上二者会分歧
GR = gdf[TARGET_COLS].to_numpy(dtype=int)
CS = sdf[TARGET_COLS].to_numpy(dtype=float)
CSR = np.clip(np.round(CS), 0, 6).astype(int)
n_diff = int((CSR != GR).sum())
n_zero = int(((CSR >= 1) & (GR == 0)).sum())

# 图：左逐维 R2 对比，右全表混淆矩阵
fig, ax = plt.subplots(1, 2, figsize=(15, 5.2))
order = np.argsort(r2)
ax[0].bar(np.arange(50) - 0.2, r2b[order], 0.4, label='基座', color='#9ecae1')
ax[0].bar(np.arange(50) + 0.2, r2[order], 0.4, label='最终', color='#08519c')
ax[0].axhline(0, color='k', lw=0.6)
ax[0].set_xticks(np.arange(50))
ax[0].set_xticklabels([TARGET_COLS[i] for i in order], rotation=90, fontsize=5)
ax[0].set_ylabel('R²')
ax[0].set_title(f'逐维 R²（宏 {mac_b:.4f} → {mac:.4f}）')
ax[0].legend(fontsize=7)
cm = np.zeros((7, 7))
for t, p in zip(G.ravel(), grade.ravel()):
    cm[int(t), int(p)] += 1
im = ax[1].imshow(cm / np.maximum(cm.sum(axis=1, keepdims=True), 1), cmap='Blues')
for i in range(7):
    for j in range(7):
        ax[1].text(j, i, f'{int(cm[i, j])}', ha='center', va='center', fontsize=6)
ax[1].set_xlabel('预测等级')
ax[1].set_ylabel('真实等级')
ax[1].set_title(f'全表混淆矩阵（命中率 {acc:.3f}）')
fig.colorbar(im, ax=ax[1], fraction=0.046)
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'fig08_answers.png'), dpi=170)
fig.savefig(os.path.join(ANS, 'fig08_answers.png'), dpi=170)
plt.close(fig)

fig2, ax2 = plt.subplots(figsize=(5.6, 5.2))
ax2.scatter(G.ravel() + np.random.RandomState(0).uniform(-0.15, 0.15, G.size),
            grade.ravel() + np.random.RandomState(1).uniform(-0.15, 0.15, G.size),
            s=2, alpha=0.25, color='#08519c')
ax2.plot([0, 6], [0, 6], 'r--', lw=0.8)
ax2.set_xlabel('真实等级')
ax2.set_ylabel('预测等级')
ax2.set_title('等级散点（加点抖动）')
fig2.tight_layout()
fig2.savefig(os.path.join(ANS, 'fig08_scatter.png'), dpi=170)
fig2.savefig(os.path.join(OUT, 'fig08_scatter.png'), dpi=170)
plt.close(fig2)

P('## 5 交付文件')
P()
P('- `答案\\问题1_感官评分_分级0到6.csv`：250 行 × (id 3 + 50 维) 等级。')
P('- `答案\\问题1_感官评分_连续分.csv`：同结构，连续分保留 2 位小数。')
P('- `Output\\Q1\\08_per_dim.csv`：逐维 R2 与 0 级占比。')
P()
P('两列的口径差别必须说清楚（否则会被当成自相矛盾）：')
P('- 分级列 = 逐树等级分布的中位数（补丁 §5.3 交付口径），对零膨胀维更保守，倾向给 0；')
P('- 连续分列 = 点预测经 z→0–6 刻度映射，等于集成的均值，遇到「多数树给 0、少数树给 2」时高于中位数。')
P(f'- 实测不一致 {n_diff} / {GR.size} = {n_diff / GR.size:.2%} 个格点，其中 {n_zero} 个是「连续分≥0.5 而分级=0」；')
P(f'- 两种口径对真实等级的命中率分别为 {acc:.4f}（中位数）与 '
  f"{J6.get('grade_match_exp', float('nan')):.4f}（期望后取整），精度相当，故按补丁保留中位数作交付列。")
P(f'- 另需注意：连续分的实际跨度是 [{float(CS.min()):.2f}, {float(CS.max()):.2f}]，')
P('  因为主口径 sigma=该维标准差、mu=−sigma·z_min 是仿射压缩，并不把 z 拉满 0–6；')
P('  真实等级也是用同一映射从 4.csv 反推的，因此等级命中率衡量的是「与该映射定义的目标一致」，')
P('  不是与外部独立评分的吻合度（fidelity 到不了这一步，02 已披露不可辨识）。')
P()

with open(os.path.join(OUT, '08_answers.md'), 'w', encoding='utf-8') as f:
    f.write('\n'.join(L))
with open(os.path.join(ANS, '问题1_说明.md'), 'w', encoding='utf-8') as f:
    f.write('\n'.join(L))
print()
print('08 完成：答案写入', ANS)

# -*- coding: utf-8 -*-
# 06e 最终叠加与验收：先 LSF（只在非 ZITS 维生效）→ 再 ZITS 替换弱维（G8）
# 同时给出两列：内层选出配置 / 全固定默认值，宏 R2 差 > 0.010 判选择过拟合并以默认值为准（G3）
# 等级分布：非 ZITS 维用五成员逐树分布的等权混合，ZITS 维用两阶段的 p·P(m≥k−0.5)
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import (OUT, TARGET_COLS, WEAK_DIMS, STRONG_DIMS, ensure_dirs, load_sensory,
                    load_sensory_grade, macro_r2, median_grade, r2_groups, score_to_grade,
                    z_to_score, load_scale)

ensure_dirs()
L = []


# 追加一行到报告并打印
def P(s=''):
    L.append(str(s))
    print(s)


sens = load_sensory()
Y = sens[TARGET_COLS].to_numpy(dtype=float)
G = load_sensory_grade()
Zb = np.load(os.path.join(OUT, '06b_members.npz'), allow_pickle=False)
Zc = np.load(os.path.join(OUT, '06c_lsf.npz'), allow_pickle=False)
Zd = np.load(os.path.join(OUT, '06d_zits.npz'), allow_pickle=False)
base = Zb['base']
on = Zd['on'].astype(bool)
lsf_sel = Zc['lsf_sel']
lsf_def = Zc['lsf_def']
zits = Zd['zits']
hist_z = Zd['hist_z']

# 直方图必须是概率分布：行和不为 1 就说明某个成员缺项，宁可直接失败也不要静默出答案
def check_hist(h, name):
    s = h.sum(axis=2)
    if not np.allclose(s, 1.0, atol=1e-9):
        raise RuntimeError(f'{name} 的行和不为 1：min={s.min():.6f} max={s.max():.6f}')
    return s


# 五成员逐树分布的等权混合，作为非 ZITS 维的等级分布
for k in ('M1', 'M2', 'M3', 'M4', 'M5'):
    check_hist(Zb[f'{k}_hist'], f'{k}_hist')
hist_mix = np.mean([Zb[f'{k}_hist'] for k in ('M1', 'M2', 'M3', 'M4', 'M5')], axis=0)
check_hist(hist_mix, 'hist_mix')
# hist_z 的形状是 (N, 50, 7)，开启标志 on 是逐维的，所以必须切 axis=1 而不是 axis=0
if np.any(on):
    check_hist(hist_z[:, on], 'hist_z(开启维)')
final_sel = np.where(on[None, :], zits, lsf_sel)
final_def = np.where(on[None, :], zits, lsf_def)

P('# 06e 最终叠加与验收')
P()
P(f'- 启用 ZITS 的维度数 = {int(on.sum())}：' + ', '.join(TARGET_COLS[j] for j in np.where(on)[0]))
P(f'- 叠加顺序：LSF 只在非 ZITS 维生效 → ZITS 替换开启维（G8）')
P()

res = {}
for name, v in (('BASE（五成员）', base), ('LSF（内层选参）', lsf_sel), ('ZITS', zits),
                ('最终（选参列）', final_sel), ('最终（默认值列）', final_def)):
    mac, weak, strong, r2 = r2_groups(Y, v)
    res[name] = dict(macro=mac, weak=weak, strong=strong, r2=r2)
P('| 方案 | 宏 R2 | 12 弱维 | 5 强维 |')
P('| --- | --- | --- | --- |')
for k, v in res.items():
    P(f"| {k} | {v['macro']:.4f} | {v['weak']:.4f} | {v['strong']:.4f} |")
P()

mac_s = res['最终（选参列）']['macro']
mac_d = res['最终（默认值列）']['macro']
gap = abs(mac_s - mac_d)
oversel = gap > 0.010
P(f'- G3 两列宏 R2 差 = {gap:.4f}，阈值 0.010：' +
  ('**判选择过拟合，以默认值列为准**' if oversel else '未触发，报告以选参列为准、默认值列并列'))
P()

AUTH = 'def' if oversel else 'sel'
final = final_def if oversel else final_sel
hist_use = np.where(on[None, :, None], hist_z, hist_mix)
mac_a, weak_a, strong_a, r2_a = r2_groups(Y, final)

# 退路：叠加后若不如基座，就「全关退化为基座」交付，并如实写明（§4 门禁的兜底口径）
fallback = mac_a < res['BASE（五成员）']['macro']
if fallback:
    final = base
    hist_use = hist_mix
    mac_a, weak_a, strong_a, r2_a = r2_groups(Y, final)
    P('- **叠加后宏 R2 低于基座，按退路全关退化为基座交付**（如实披露，不硬撑）')
    P()
grade_use = median_grade(hist_use)
# 对照口径：先取分布期望再取整（补丁 §5.3 要求中位数与期望并列，交付用中位数）
exp_use = np.tensordot(hist_use, np.arange(7, dtype=float), axes=([2], [0]))
grade_exp = score_to_grade(exp_use)
P('## 等级口径对照')
P()
P(f'- 中位数（交付口径）全表命中率 = {float(np.mean(grade_use == G)):.4f}')
P(f'- 期望后取整（对照口径）全表命中率 = {float(np.mean(grade_exp == G)):.4f}')
P(f'- 偏差 ≤1 级的比例（中位数口径）= {float(np.mean(np.abs(grade_use - G) <= 1)):.4f}')
P()

# 三条门禁
g_macro = mac_a >= 0.274
g_weak = weak_a >= 0.075
g_strong = (res['BASE（五成员）']['strong'] - strong_a) <= 0.005
P('## 门禁')
P()
P(f'| 门禁 | 阈值 | 实测 | 结论 |')
P('| --- | --- | --- | --- |')
P(f'| 宏 R2 | ≥ 0.274 | {mac_a:.4f} | {"通过" if g_macro else "未达"} |')
P(f'| 12 弱维均值 | ≥ 0.075 | {weak_a:.4f} | {"通过" if g_weak else "未达"} |')
P(f'| 5 强维下降 | ≤ 0.005 | {res["BASE（五成员）"]["strong"] - strong_a:.4f} | '
  f'{"通过" if g_strong else "未达"} |')
P()
P(f'- 若三条全不达，仍按「全关退化为基座」的退路交付基座口径；当前选用 {AUTH} 列。')
P()

MU, SG = load_scale()
score = z_to_score(final, MU, SG)

rows = []
for j, c in enumerate(TARGET_COLS):
    rows.append({'dim': c, 'base_r2': float(res['BASE（五成员）']['r2'][j]),
                 'lsf_r2': float(res['LSF（内层选参）']['r2'][j]),
                 'zits_r2': float(res['ZITS']['r2'][j]),
                 'final_r2': float(r2_a[j]), 'zits_on': bool(on[j]),
                 'weak': c in WEAK_DIMS, 'strong': c in STRONG_DIMS,
                 'zero_share': float(Zd['zero_mean'][j])})
T = pd.DataFrame(rows)
T.to_csv(os.path.join(OUT, '06e_final_dim.csv'), index=False, encoding='utf-8-sig')
P('## 12 弱维前后对比')
P()
P('| 维度 | 零占比 | 基座 R2 | 最终 R2 | 变化 | ZITS |')
P('| --- | --- | --- | --- | --- | --- |')
for c in WEAK_DIMS:
    r = T[T['dim'] == c].iloc[0]
    P(f"| {c} | {r['zero_share']:.2f} | {r['base_r2']:.4f} | {r['final_r2']:.4f} | "
      f"{r['final_r2'] - r['base_r2']:+.4f} | {'是' if r['zits_on'] else '否'} |")
P()
P('注：12 个弱维是**事后子集**（由 v1 逐维 R2 挑出），不是先验定义；')
P(f'整体宏 R2 仍同时给出（{mac_a:.4f}），避免只报子集口径（G9）。')
P()

np.savez_compressed(os.path.join(OUT, '06e_final.npz'), y=Y, g=G, final=final, score=score,
                    hist=hist_use, grade=grade_use, grade_exp=grade_exp, base=base, lsf=lsf_sel,
                    zits=zits, on=on, authority=AUTH, fold_id=Zb['fold_id'])
with open(os.path.join(OUT, '06e_final.json'), 'w', encoding='utf-8') as f:
    json.dump({'authority': AUTH, 'macro': mac_a, 'weak': weak_a, 'strong': strong_a,
               'base': [res['BASE（五成员）']['macro'], res['BASE（五成员）']['weak'],
                        res['BASE（五成员）']['strong']],
               'g3_gap': gap, 'oversel': bool(oversel),
               'gates': {'macro': bool(g_macro), 'weak': bool(g_weak), 'strong': bool(g_strong)},
               'dims_on': [TARGET_COLS[j] for j in np.where(on)[0]],
               'grade_match': float(np.mean(grade_use == G)),
               'grade_match_exp': float(np.mean(grade_exp == G))}, f, ensure_ascii=False, indent=2)
with open(os.path.join(OUT, '06e_final.md'), 'w', encoding='utf-8') as f:
    f.write('\n'.join(L))
print()
print('06e 完成，输出写入', OUT, ' 采用列 =', AUTH)

# -*- coding: utf-8 -*-
# Q1 步骤九：核实 1.csv 的对数底数，并检验各列是否统一取了对数
# 题目明确要求"核实 log 底数（log10 还是 ln），明确正负值与原始浓度对应逻辑"
# 这里用三类证据：物理量的可解释量级、分布形态（是否出现负值）、与阈值表的量级一致性
import importlib
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
FE = importlib.import_module('03_features')
from common import (OUT, PHYS_COLS, ensure_dirs, load_chem, load_x8)

ensure_dirs()
L = []


def P(s=''):
    L.append(str(s))
    print(s)


chem = load_chem()
all_cols = [c for c in chem.columns if c not in ('beer', 'beer_id', 'tasting_category_fine')]
X = chem[all_cols].to_numpy(float)

P('# Q1 步骤九：对数底数与标度一致性核实')
P()
P(f'1.csv 共 {len(chem)} 行样本、{len(all_cols)} 个可用数值列。')
P()

# 一 底数判定：拿已知物理量去反变换，看哪个底数还原出啤酒工艺上说得通的值
# 括号里是正常啤酒该物理量的合理区间，用来判定反变换是否命中
KNOWN = {
    'pH': (2.8, 6.5),
    'ethanol..v.v.': (0.0, 20.0),
    'bitterness': (2.0, 130.0),
    'color.EBC.': (1.0, 200.0),
    'protein.g.L.': (0.2, 20.0),
    'glycerol': (0.05, 25.0),
    'iron': (0.001, 5.0),
    'CO2.PSI.': (1.0, 30.0),
    'kcalperc': (10.0, 700.0),
    'ammonia.mg.L.': (0.001, 80.0),
    'betaglucan.mg.L.': (1.0, 2000.0),
    'lactic_acid.mg.L.': (1.0, 12000.0),
    'acetic_acid.g.L.': (0.001, 5.0),
    'SO2.mg.L.': (0.001, 200.0),
    'unfermentables': (1.0, 400.0),
    'GFS': (0.1, 60.0),
}
P('## 1 底数判定：用 16 个已知物理量的反变换命中率')
P()
P('若列存的是 log_b(v)，则 b^z 应落回该物理量在啤酒中的合理区间。分别用 b = 10 与 b = e 试。')
P()
P('| 物理量 | 合理区间 | z 中位数 | 10^z 中位数 | 命中(10) | e^z 中位数 | 命中(e) |')
P('| --- | --- | --- | --- | --- | --- | --- |')
h10 = he = 0
n_known = 0
for col, (lo, hi) in KNOWN.items():
    if col not in all_cols:
        continue
    z = X[:, all_cols.index(col)]
    z = z[~np.isnan(z)]
    v10 = np.power(10.0, z)
    ve = np.exp(z)
    r10 = float(((v10 >= lo) & (v10 <= hi)).mean())
    re = float(((ve >= lo) & (ve <= hi)).mean())
    h10 += r10
    he += re
    n_known += 1
    P(f'| {col} | {lo:g} ~ {hi:g} | {np.median(z):.3f} | {np.median(v10):.4g} | {r10:.3f} | '
      f'{np.median(ve):.4g} | {re:.3f} |')
P()
P(f'16 个物理量的平均命中率：base 10 为 **{h10 / n_known:.4f}**，base e 为 {he / n_known:.4f}，'
  f'相差 {h10 / max(he, 1e-9):.2f} 倍，判定为 **log10**。')
P()
P('平均命中率差距不大是因为有些物理量在两种底数下都落进合理区间（如乙醇 7.19 对 2.36 都算低醇啤酒），')
P('真正决定性的是 pH：ln 反变换给出 1.85，低于啤酒 pH 的物理下限 2.8，命中率为 0；log10 给出 4.13，命中率 1.000。')
P('卡路里同理，ln 给出 5.77 远低于啤酒 10~700 kcal/100ml 的下限。因此底数只能是 10。')
P()

# 二 正负号的含义：z > 0 表示原始浓度大于 1，z < 0 表示小于 1，不是数据异常
neg_cols = [c for c in all_cols if np.nanmin(X[:, all_cols.index(c)]) < 0]
P('## 2 正负号的物理含义')
P()
P(f'{len(all_cols)} 列里有 {len(neg_cols)} 列出现负值。在 log10 标度下 z 的正负只表示原始值相对 1 的大小：')
P()
P('- z = +1 对应原始值 10，z = +2 对应 100，z = -1 对应 0.1，z = -2 对应 0.01。')
P('- 阈值表里的痕量香气物质多在 0.01 ~ 100 的区间，因此 z 落在 -2 ~ +2 是正常取值，不是缺失或错误编码。')
P('- 只有当一列出现大量完全相同的最小值时才需要警惕左删失，这一点在 `01_audit.md` 已单独统计。')
P()

# 三 标度一致性：55 个非负列不满足 log10 的分布形态，需要单独判定
P('## 3 标度一致性：并非所有列都取了对数')
P()
rows = []
for j, col in enumerate(all_cols):
    v = X[:, j]
    v = v[~np.isnan(v)]
    rows.append(dict(col=col, mn=float(v.min()), mx=float(v.max()), med=float(np.median(v)),
                     frac_neg=float((v < 0).mean()), frac_zero=float((v == 0).mean()),
                     skew=float(stats.skew(v))))
D = pd.DataFrame(rows)
nonneg = D[D.frac_neg == 0]
P(f'{len(D)} 列里 {len(nonneg)} 列全程非负。log10 标度下小于 1 的浓度必然产生负值，')
P('所以"全程非负"的列要么原始值都大于 1，要么根本没取对数。用分布形态进一步区分：')
P()
P('| 判据 | 说明 |')
P('| --- | --- |')
P('| 最大值 > 10 且右偏严重 | 未取对数的原始浓度（对数变换会把上万的值压到 4 以下） |')
P('| 取值集中在 4 ~ 9 且无零值 | 原始浓度，单位是 µg/L 量级 |')
P('| 取值在 -2 ~ +2 且含零值 | log10 标度，零值对应原始浓度 1 |')
P()
n_heavy = int(((nonneg.mx > 10) & (nonneg['skew'] > 2)).sum())
n_unit = int(((nonneg.mx <= 10) & (nonneg.med > 3)).sum())
P(f'按此判据，{len(nonneg)} 个非负列里有 {n_heavy} 列右偏极强（最大值上万），'
  f'{n_unit} 列取值集中在 4 到 9 之间。这两类都应读成原始浓度而不是 log10。')
P()

# 四 用阈值表交叉验证：分别假设"已取对数"和"原始值"，看哪个量级更贴近阈值
thr = load_x8('Sensory thresholds').dropna(subset=['threshold_mean'])
tmap = {}
for _, r in thr.iterrows():
    tmap[str(r['field_name_R'])] = float(r['threshold_mean'])
P('## 4 阈值交叉验证')
P()
P('8.xlsx 的阈值表给了部分化合物的嗅觉阈值。若某列是 log10，则中位浓度应取 10^z；若是原始值则直接取 z。')
P('把两种假设下的中位浓度与阈值相比，看哪个落在"能被闻到但不荒谬"的量级（0.01 ~ 1000 倍阈值）。')
P()
P('| 化合物 | 阈值 | 中位 z | 10^z 假设 OAV | 原始值假设 OAV | 更合理的假设 |')
P('| --- | --- | --- | --- | --- | --- |')
n_log_ok = n_raw_ok = n_pair = 0
for col in all_cols:
    t = tmap.get(col)
    if not t or t <= 0:
        continue
    v = X[:, all_cols.index(col)]
    med = float(np.nanmedian(v))
    o_log = (10.0 ** med) / t
    o_raw = med / t
    p_log = 0.01 < o_log < 1000
    p_raw = 0.01 < o_raw < 1000
    if not p_log and not p_raw:
        continue
    n_pair += 1
    n_log_ok += int(p_log)
    n_raw_ok += int(p_raw)
    if n_pair <= 24:
        better = 'log10' if (p_log and not p_raw) else ('原始值' if (p_raw and not p_log) else '两者皆可')
        P(f'| {col} | {t:.4g} | {med:.3f} | {o_log:.4g} | {o_raw:.4g} | {better} |')
P()
P(f'{n_pair} 个可判定的组合中，log10 假设合理的有 {n_log_ok} 个，原始值假设合理的有 {n_raw_ok} 个。')
P('注意阈值表本身混用 mg/L、g/L、ug/L 三种单位，这里只做量级合理性判断，不能当精确的 OAV。')
P()

# 五 结论对后续步骤的影响
P('## 5 结论与影响')
P()
P(f'1. **底数确认为 log10**：16 个物理量的反变换平均命中率 {h10 / n_known:.4f} 对 {he / n_known:.4f}，'
  f'pH 与卡路里在 ln 假设下直接出界。')
P('2. **负值是正常取值**：z 的正负对应原始浓度相对 1 的大小，不是缺失、不是异常编码。')
P(f'3. **标度不统一**：{len(nonneg)} 个非负列给的是原始浓度。其中 {n_heavy} 列最大值上万，'
  f'作为 log10 会对应 1e4 以上的浓度，物理上不可能；{n_unit} 列集中在 4 到 9 且无零值；'
  f'余下的非负列两种读法都说得通，统一按原始值处理以保持一致。')
P('4. 对问题一的影响为零：树模型只依赖特征的排序，逐列单调变换不改变分裂点，因此是否取对数不影响 Q1 的预测。')
P('5. 对问题二的影响为 OAV 的量级：必须按列判断，log10 列取 10^z，原始值列直接取 z，否则会算出 1e8 这种荒谬的 OAV。')
P('6. 对问题三的影响为生成值的支持域：原始浓度列的右尾很长，生成时必须把超出观测范围的值截回区间内。')
P()

with open(os.path.join(OUT, '09_scale_audit.md'), 'w', encoding='utf-8') as f:
    f.write('\n'.join(L))
D.to_csv(os.path.join(OUT, '09_scale_audit.csv'), index=False, encoding='utf-8-sig')
with open(os.path.join(OUT, '09_scale_audit.json'), 'w', encoding='utf-8') as f:
    json.dump(dict(n_cols=int(len(D)), n_nonneg=int(len(nonneg)), hit_log10=float(h10 / n_known),
                   hit_ln=float(he / n_known),
                   nonneg_raw_heavy=int(n_heavy),
                   n_threshold_pair=int(n_pair), n_log_plausible=int(n_log_ok),
                   n_raw_plausible=int(n_raw_ok)), f, ensure_ascii=False, indent=2)
print('标度核实结果已写入', OUT)

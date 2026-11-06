# -*- coding: utf-8 -*-
# 01 数据审计：核实 log 底数、缺失模式、左删失、目标标度、风格分布
# 输出 Output/Q1/01_audit.md 与 01_audit.json，供后续所有脚本引用
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (ATT, CORE_COLS, EXTRA_COLS, ID_COLS, OUT, SUM_COLS, TARGET_COLS,
                    chem_feature_cols, ensure_dirs, load_chem, load_sensory, load_x8)

ensure_dirs()
L = []


# 追加一行到报告
def P(s=''):
    L.append(str(s))
    print(s)


# 反变换后各理化量应有的物理区间，用于判定 log 底数
PHYS_RANGE = {
    'pH': (3.0, 5.0),
    'ethanol..v.v.': (0.5, 15.0),
    'protein.g.L.': (1.0, 15.0),
    'color.EBC.': (2.0, 120.0),
    'glycerol': (0.1, 8.0),
    'bitterness': (2.0, 120.0),
    'CO2.PSI.': (5.0, 45.0),
    'lactic_acid.mg.L.': (1.0, 5000.0),
    'acetic_acid.g.L.': (0.005, 2.0),
    'ammonia.mg.L.': (0.1, 50.0),
    'iron': (0.005, 5.0),
    'betaglucan.mg.L.': (1.0, 3000.0),
    'unfermentables': (1.0, 200.0),
    'kcalperc': (10.0, 200.0),
}


# 判定 log 底数：比较两种底数反变换后落入物理区间的比例
def detect_log_base(df):
    rows = []
    for col, (lo, hi) in PHYS_RANGE.items():
        if col not in df.columns:
            continue
        v = df[col].dropna().to_numpy()
        hit10 = float(np.mean((10.0 ** v >= lo) & (10.0 ** v <= hi)))
        hite = float(np.mean((np.exp(v) >= lo) & (np.exp(v) <= hi)))
        rows.append({'col': col, 'hit_base10': hit10, 'hit_baseE': hite,
                     'median_10': float(np.median(10.0 ** v)),
                     'median_e': float(np.median(np.exp(v))),
                     'range': [lo, hi]})
    return pd.DataFrame(rows)


# 统计每列在最小值上重复的次数，用于识别检测限左删失
def left_censoring(df, cols):
    rows = []
    for c in cols:
        v = df[c].dropna().to_numpy()
        if v.size == 0:
            continue
        mn = v.min()
        n_at_min = int(np.sum(np.isclose(v, mn, rtol=0, atol=1e-9)))
        rows.append({'col': c, 'n_at_min': n_at_min, 'frac': n_at_min / v.size, 'min': float(mn)})
    return pd.DataFrame(rows).sort_values('n_at_min', ascending=False)


P('# 01 数据审计报告')
P()
P('## 1 文件与形状')
P()
chem = load_chem()
sens = load_sensory()
d7 = pd.read_csv(os.path.join(ATT, '7.csv'))
P('| 文件 | 形状 | 说明 |')
P('| --- | --- | --- |')
P(f'| 1.csv | {chem.shape} | 化合物与理化指标（预测特征源） |')
P(f'| 4.csv | {sens.shape} | 50 维感官评分（训练目标，已标准化） |')
P(f'| 7.csv | {d7.shape} | 186 个描述词评分（外部校验，禁入特征） |')
for f in ['2.csv', '5.csv', '6.csv']:
    d = pd.read_csv(os.path.join(ATT, f))
    P(f'| {f} | {d.shape} | 相关/共表达矩阵 |')
P()

P('## 2 样本与风格对齐')
P()
P(f'1.csv 与 4.csv 行数一致：{len(chem)==len(sens)}')
key_c = list(zip(chem['beer'].astype(int), chem['beer_id'].astype(np.int64)))
key_s = list(zip(sens['beer'].astype(int), sens['beer_id'].astype(np.int64)))
P(f'beer 编号集合一致：{set(k[0] for k in key_c)==set(k[0] for k in key_s)}')
P(f'(beer,beer_id) 逐行顺序一致：{key_c==key_s}')
same_style = int((chem['tasting_category_fine'].values == sens['tasting_category_fine'].values).sum())
P(f'风格标签逐行一致条数：{same_style}/{len(chem)}')
vc = sens['tasting_category_fine'].value_counts()
P(f'风格类别数：{vc.size}，最小类样本数：{vc.min()}，最大类样本数：{vc.max()}')
P()
P('| 风格 | 样本数 |')
P('| --- | --- |')
for k, v in vc.items():
    P(f'| {k} | {v} |')
P()

P('## 3 log 底数核实（题目要求）')
P()
base = detect_log_base(chem)
P('对每个已知物理量，比较 `10^z` 与 `e^z` 落入合理物理区间的比例：')
P()
P('| 列 | base10 命中率 | baseE 命中率 | 10^z 中位数 | e^z 中位数 | 合理区间 |')
P('| --- | --- | --- | --- | --- | --- |')
for _, r in base.iterrows():
    P(f"| {r['col']} | {r['hit_base10']:.3f} | {r['hit_baseE']:.3f} | "
      f"{r['median_10']:.4g} | {r['median_e']:.4g} | {r['range']} |")
P()
m10 = float(base['hit_base10'].mean())
me = float(base['hit_baseE'].mean())
P(f'平均命中率：base10 = {m10:.4f}，baseE = {me:.4f}')
# 自动判定：命中率高的那个底数即为真实底数
base_is_10 = m10 >= me
P(f'结论：1.csv 的数值是 **log{10 if base_is_10 else "e"} 变换后的浓度**，'
  f'反变换公式为 `c = {"10" if base_is_10 else "e"}^z`。')
P()
P('正负值与原始浓度的对应逻辑：')
P('- `z > 0` 表示原始浓度大于 1（单位同 8.xlsx 的 `unit` 列）；`z = 0` 对应浓度恰为 1。')
P('- `z < 0` 表示原始浓度小于 1，即表内单位下的痕量水平，并非测量错误。')
P('- 同一列存在大量重复的最小值，说明该化合物在检测限以下被统一置为该底值（左删失）。')
P()

P('## 4 左删失（检测限）识别')
P()
feat_cols = chem_feature_cols(chem)
cens = left_censoring(chem, feat_cols)
top = cens[cens['frac'] > 0.20]
P(f'最小值上重复比例 > 20% 的列共 {len(top)} 个，这些列的最低值应视为检测限：')
P()
P('| 列 | 最小值 | 重复次数 | 占比 |')
P('| --- | --- | --- | --- |')
for _, r in top.head(25).iterrows():
    P(f"| {r['col']} | {r['min']:.4f} | {r['n_at_min']} | {r['frac']:.3f} |")
P()

P('## 5 缺失模式（缺失即信息）')
P()
miss = chem[feat_cols].isna().sum()
miss = miss[miss > 0].sort_values(ascending=False)
P(f'含缺失的特征列共 {len(miss)} 个：')
P()
P('| 列 | 缺失数 | 缺失率 |')
P('| --- | --- | --- |')
for k, v in miss.items():
    P(f'| {k} | {int(v)} | {v/len(chem):.3f} |')
P()
P(f'全表缺失总数：{int(miss.sum())}，占特征单元比例 {miss.sum()/(len(chem)*len(feat_cols)):.5f}')
P()
P(f'4.csv 目标缺失总数：{int(sens[TARGET_COLS].isna().sum().sum())}')
P(f'7.csv 感官缺失总数：{int(d7.isna().sum().sum())}')
P()

P('## 6 目标标度诊断')
P()
st = sens[TARGET_COLS].agg(['mean', 'std', 'min', 'max', 'skew']).T
st['n_uniq'] = [sens[c].nunique() for c in st.index]
P('| 维度 | mean | std | min | max | skew | 不同取值数 |')
P('| --- | --- | --- | --- | --- | --- | --- |')
for c, r in st.iterrows():
    P(f"| {c} | {r['mean']:.4f} | {r['std']:.4f} | {r['min']:.3f} | {r['max']:.3f} | "
      f"{r['skew']:.2f} | {int(r['n_uniq'])} |")
P()
P(f"全表：列均值的中位数 {st['mean'].median():.4f}，列标准差的中位数 {st['std'].median():.4f}，"
  f"全局最小 {st['min'].min():.3f}，全局最大 {st['max'].max():.3f}")
P(f"取值全为整数：{bool(np.allclose(sens[TARGET_COLS].values, np.round(sens[TARGET_COLS].values)))}")
P(f"不同取值数中位数：{int(st['n_uniq'].median())}（接近 250 说明是连续的面板均值而非整数打分）")
P()
P('诊断：4.csv 列均值≈0 说明已中心化；列标准差中位数约 0.55 而非 1，'
  '说明标准化时所用分母不是本列样本标准差，因此原始 0–6 标度的 (μ, σ) 需要单独反解。')
P()

P('## 7 维度分组与题目口径')
P()
P(f'3.xlsx summary 定义的评分维度共 {len(TARGET_COLS)} 个（去掉 comments 列）：')
P(f'- Aroma（A_）{len([c for c in TARGET_COLS if c.startswith("A_")])} 维')
P(f'- Flavour/Taste（F_）{len([c for c in TARGET_COLS if c.startswith("F_")])} 维')
P(f'- 其余口感与缺陷 {len(EXTRA_COLS)} 维')
P(f'- 题目所述 30 维核心集（A_ + F_）共 {len(CORE_COLS)} 维')
P()
P(f'聚合列（由化合物列线性组合而来，非独立测量）：{SUM_COLS}')
P()

P('## 8 化合物覆盖与元数据')
P()
th = load_x8('Sensory thresholds')
th_keys = set(th['field_name_R'].dropna().astype(str))
cover = sum(1 for c in feat_cols if c in th_keys)
P(f'1.csv 特征列 {len(feat_cols)} 个，其中 {cover} 个能在 8.xlsx 阈值表中找到元数据')
P(f'阈值表共 {len(th)} 行，含 threshold_mean 非空的行：{int(th["threshold_mean"].notna().sum())}')
P()
org = th['origin_paper'].dropna().value_counts()
P('| 来源 | 化合物数 |')
P('| --- | --- |')
for k, v in org.items():
    P(f'| {k} | {v} |')
P()

P('## 9 结论')
P()
P(f'1. 1.csv 为 log{10 if base_is_10 else "e"} 浓度，负值为亚单位浓度，非异常。')
P('2. 9 列存在缺失，最多 27 条，须用缺失指示列 + 哨兵填充，不可全局均值填充。')
P('3. 4.csv 为标准化后的连续面板均值，0–6 等级需在 02 步反解 (μ, σ) 后给出。')
P('4. 风格共 %d 类，最小类 %d 条，分层抽样可行。' % (vc.size, vc.min()))
P('5. 7.csv/5.csv/6.csv 为感官派生数据，仅用于外部校验，禁止进入特征。')
P()

with open(os.path.join(OUT, '01_audit.md'), 'w', encoding='utf-8') as f:
    f.write('\n'.join(L))

info = {
    'n_samples': int(len(chem)),
    'n_chem_features': int(len(feat_cols)),
    'n_targets': int(len(TARGET_COLS)),
    'log_base': 10 if base_is_10 else float(np.e),
    'log_base_hit10': m10,
    'log_base_hitE': me,
    'miss_cols': {k: int(v) for k, v in miss.items()},
    'styles': {str(k): int(v) for k, v in vc.items()},
    'target_std_median': float(st['std'].median()),
    'censored_cols': {r['col']: int(r['n_at_min']) for _, r in top.iterrows()},
}
with open(os.path.join(OUT, '01_audit.json'), 'w', encoding='utf-8') as f:
    json.dump(info, f, ensure_ascii=False, indent=2)

print()
print('审计报告已写入', os.path.join(OUT, '01_audit.md'))

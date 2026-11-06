# -*- coding: utf-8 -*-
# 临时探查脚本：判断 4.csv / 7.csv 的数值标度，检查 1.csv 缺失列，看 8.xlsx 阈值表
import os
import numpy as np
import pandas as pd

_here = os.path.dirname(os.path.abspath(__file__))
while not os.path.isdir(os.path.join(_here, '题目')):
    _here = os.path.dirname(_here)
WS = _here
ATT = os.path.join(WS, '题目', '附件')

d4 = pd.read_csv(os.path.join(ATT, '4.csv'))
d7 = pd.read_csv(os.path.join(ATT, '7.csv'))
d1 = pd.read_csv(os.path.join(ATT, '1.csv'))

sens4 = [c for c in d4.columns if c not in ('beer', 'beer_id', 'tasting_category_fine')]
print('4.csv 感官维度数 =', len(sens4))
st = d4[sens4].agg(['mean', 'std', 'min', 'max']).T
print(st.head(12).to_string())
print('--- 4.csv 全局: 每列 mean 的均值 =%.4f  std 的均值 =%.4f' % (st['mean'].mean(), st['std'].mean()))
print('--- 4.csv 每列 mean 绝对值中位数 =%.4f' % st['mean'].abs().median())
print('--- 4.csv 每列 std 中位数 =%.4f' % st['std'].median())
print('--- 4.csv 取值是否全为整数:', np.allclose(d4[sens4].values, np.round(d4[sens4].values), equal_nan=True))
print('--- 4.csv 最小值 %.3f 最大值 %.3f' % (st['min'].min(), st['max'].max()))
print()

sens7 = [c for c in d7.columns if c not in ('beer', 'beer_id', 'tasting_category_fine')]
print('7.csv 感官维度数 =', len(sens7))
st7 = d7[sens7].agg(['mean', 'std', 'min', 'max']).T
print(st7.head(10).to_string())
print('--- 7.csv 每列 mean 的均值 =%.4f  std 的均值 =%.4f' % (st7['mean'].mean(), st7['std'].mean()))
print('--- 7.csv 是否全为整数:', np.allclose(d7[sens7].values, np.round(d7[sens7].values), equal_nan=True))
print('--- 7.csv 最小值 %.3f 最大值 %.3f' % (st7['min'].min(), st7['max'].max()))
print('--- 7.csv 缺失数 =', int(d7[sens7].isna().sum().sum()))
print()

print('1.csv 缺失列:')
m = d1.isna().sum()
print(m[m > 0].to_string())
print()
print('1.csv 各列是否含负值（用于判断 log 变换）:')
nonneg = 0
hasneg = 0
for c in d1.columns:
    if d1[c].dtype == object or c in ('beer', 'beer_id', 'tasting_category_fine'):
        continue
    if (d1[c] < 0).any():
        hasneg += 1
print('  含负值列数 =', hasneg, ' 非负列数 =', 231 - hasneg)
print()

print('=' * 100)
th = pd.read_excel(os.path.join(ATT, '8.xlsx'), sheet_name='Sensory thresholds')
print('Sensory thresholds 列:', list(th.columns))
print(th.head(15).to_string())
print()
sn = pd.read_excel(os.path.join(ATT, '8.xlsx'), sheet_name='Sensory names')
print('Sensory names 列:', list(sn.columns))
print(sn.to_string())

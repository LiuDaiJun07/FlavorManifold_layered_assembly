# -*- coding: utf-8 -*-
# 临时探查脚本：判定 4.csv / 7.csv 是否由整数 0-6 标度标准化而来，并判定 1.csv 的数值形态
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

print('===== 4.csv 每列不同取值个数（前 20 列） =====')
sens4 = [c for c in d4.columns if c not in ('beer', 'beer_id', 'tasting_category_fine')]
for c in sens4[:20]:
    u = np.unique(d4[c].values)
    gaps = np.diff(u)
    print(f'{c:20s} n_uniq={len(u):4d} min_gap={gaps.min() if len(gaps) else np.nan:.6f} '
          f'min={u.min():.4f} max={u.max():.4f}')
print()

print('===== 7.csv 每列不同取值个数（前 20 列） =====')
sens7 = [c for c in d7.columns if c not in ('beer', 'beer_id', 'tasting_category_fine')]
for c in sens7[:20]:
    u = np.unique(d7[c].dropna().values)
    gaps = np.diff(u)
    print(f'{c:22s} n_uniq={len(u):4d} min_gap={gaps.min() if len(gaps) else np.nan:.6f} '
          f'min={u.min():.4f} max={u.max():.4f}')
print()

print('===== 1.csv 数值形态抽查 =====')
probe = ['DMS', 'ethyl_acetate', 'isoamyl_alcohol', 'pH', 'bitterness',
         'ethanol..v.v.', 'CO2.PSI.', 'color.EBC.', 'protein.g.L.', 'glycerol',
         'linalool', 'geosmin', 'X4VG', 'esters_sum']
print(d1[probe].describe().T.to_string())
print()

print('===== 1.csv 各列 mean/std 汇总（判断是否已标准化） =====')
num1 = [c for c in d1.columns if d1[c].dtype != object]
desc = d1[num1].describe().T
print('mean 的中位数 =%.4f, std 的中位数 =%.4f' % (desc['mean'].median(), desc['std'].median()))
print('std 最小的 5 列:'); print(desc['std'].nsmallest(5).to_string())
print()
print('===== 8.xlsx Calibrated chemical dataset 与 1.csv 对比 =====')
cal = pd.read_excel(os.path.join(ATT, '8.xlsx'), sheet_name='Calibrated chemical dataset')
print('calibrated shape', cal.shape)
print(cal[probe].describe().T.to_string())
print()
print('===== 8.xlsx GBM importance MDI 前 20 =====')
mdi = pd.read_excel(os.path.join(ATT, '8.xlsx'), sheet_name='GBM importance MDI')
mdi.columns = ['compound', 'mdi']
print(mdi.sort_values('mdi', ascending=False).head(20).to_string())

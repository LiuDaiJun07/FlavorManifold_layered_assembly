# -*- coding: utf-8 -*-
# 临时探查脚本：打印 1/2/4/5/6/7 csv 与 8.xlsx 的列结构
import os
import pandas as pd

_here = os.path.dirname(os.path.abspath(__file__))
while not os.path.isdir(os.path.join(_here, '题目')):
    _here = os.path.dirname(_here)
WS = _here
ATT = os.path.join(WS, '题目', '附件')

for f in ['1.csv', '2.csv', '4.csv', '5.csv', '6.csv', '7.csv']:
    df = pd.read_csv(os.path.join(ATT, f))
    print('=' * 100)
    print(f'{f}  shape={df.shape}')
    print('cols:', list(df.columns))
    print('dtypes:', df.dtypes.value_counts().to_dict())
    print(df.head(3).to_string()[:2000])
    print()

wb = pd.ExcelFile(os.path.join(ATT, '8.xlsx'))
print('=' * 100)
print('8.xlsx sheets:', wb.sheet_names)
for sn in wb.sheet_names:
    d = wb.parse(sn)
    print(f'--- {sn} shape={d.shape} cols={list(d.columns)[:40]}')

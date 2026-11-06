# -*- coding: utf-8 -*-
# 临时探查脚本：打印附件 1-8 的结构，供后续建模设计使用
import os
import sys
import pandas as pd

pd.set_option('display.width', 200)
pd.set_option('display.max_columns', 300)
pd.set_option('display.max_rows', 400)

# 向上寻找工作区根目录（含“题目”文件夹的那一层）
_here = os.path.dirname(os.path.abspath(__file__))
while not os.path.isdir(os.path.join(_here, '题目')):
    _here = os.path.dirname(_here)
WS = _here
BASE = os.path.join(WS, 'main')
ATT = os.path.join(WS, '题目', '附件')

def head(title):
    print('\n' + '=' * 100)
    print(title)
    print('=' * 100)

head('3.xlsx sheet names')
xl = pd.ExcelFile(os.path.join(ATT, '3.xlsx'))
print(xl.sheet_names)

for sn in xl.sheet_names:
    df = xl.parse(sn)
    head(f'3.xlsx / {sn}  shape={df.shape}')
    print('columns:', list(df.columns))
    print(df.head(8).to_string())

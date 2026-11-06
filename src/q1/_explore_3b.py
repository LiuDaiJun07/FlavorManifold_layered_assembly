# -*- coding: utf-8 -*-
# 临时探查脚本：逐单元格打印 3.xlsx 的 beer_1 与 summary，确认真实的 0-6 评分维度定义
import os
import openpyxl

_here = os.path.dirname(os.path.abspath(__file__))
while not os.path.isdir(os.path.join(_here, '题目')):
    _here = os.path.dirname(_here)
WS = _here
ATT = os.path.join(WS, '题目', '附件')

wb = openpyxl.load_workbook(os.path.join(ATT, '3.xlsx'), data_only=True)

for sn in ['beer_1', 'summary']:
    ws = wb[sn]
    print('=' * 100)
    print(f'SHEET {sn}  dims={ws.dimensions}  max_row={ws.max_row} max_col={ws.max_column}')
    print('=' * 100)
    for row in ws.iter_rows():
        cells = []
        for c in row:
            if c.value is not None and str(c.value).strip() != '':
                cells.append(f'{c.coordinate}={c.value!r}')
        if cells:
            print(' | '.join(cells))
    print()

# -*- coding: utf-8 -*-
# 交付两列口径核对：round(连续分) 与 分级 有多少格点不一致，连续分的实际跨度
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'q1'))
from common import ANS, TARGET_COLS  # noqa: E402

new = pd.read_csv(os.path.join(ANS, '问题1_感官评分_分级0到6.csv'))
cs = pd.read_csv(os.path.join(ANS, '问题1_感官评分_连续分.csv'))
A = new[TARGET_COLS].to_numpy(dtype=int)
C = cs[TARGET_COLS].to_numpy(dtype=float)
R = np.clip(np.round(C), 0, 6).astype(int)
print('round(连续分) != 分级 的格点数 =', int((R != A).sum()), '/', A.size)
print('比例 =', round(float((R != A).mean()), 4))
print('连续分 均值 =', round(float(C.mean()), 4), ' 最大 =', round(float(C.max()), 4))
print('连续分 >1.5 的格点 =', int((C > 1.5).sum()))
print('分级=1 但连续分<0.5 的格点 =', int(((A == 1) & (C < 0.5)).sum()))
print('分级=0 但连续分>=0.5 的格点 =', int(((A == 0) & (C >= 0.5)).sum()))

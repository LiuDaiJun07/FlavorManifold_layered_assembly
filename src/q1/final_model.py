# -*- coding: utf-8 -*-
# Q1 最终模型的统一出口：在全样本上拟合 E0 与 E1 两套极端森林并做等级还原
# 只服务问题三的跨风格迁移（生成样本的感官画像读出器）；问题一的答案走 10 折折外结果（G5）
# 树数与并行度统一由 common.trees / NJOB 控制，不再各处散落硬编码
import importlib
import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
FE = importlib.import_module('03_features')
from common import (NJOB, SEED, TARGET_COLS, expect_grade, load_chem, load_scale, load_sensory,
                    median_grade, score_to_grade, trees, z_to_score)

N_EST_FINAL = 1200
MF = 0.5
MSL = 1


# 单套特征口径：训练表的哨兵与缺失指示统计量只从训练表估，再作用到新表
class _View(object):
    def __init__(self, train_df, sens, use_style, n_est=N_EST_FINAL):
        self.keep, _ = FE.prune_cols(train_df)
        self.train_X = train_df[self.keep].to_numpy(float)
        self.use_style = use_style
        self.style_tr = sens['tasting_category_fine'].to_numpy()
        self.cats = sorted(pd.unique(self.style_tr)) if use_style else []
        A, _, _ = FE.sentinel_fill(self.train_X, self.train_X)
        MA, _, _ = FE.miss_indicators(self.train_X, self.train_X)
        A = np.hstack([A, MA])
        if use_style:
            M, _, _ = FE.style_onehot(self.style_tr, self.style_tr)
            A = np.hstack([A, M])
        self.mu, self.sg = load_scale()
        Y = sens[TARGET_COLS].to_numpy(float)
        self.m = ExtraTreesRegressor(n_estimators=trees(n_est), max_features=MF,
                                     min_samples_leaf=MSL,
                                     n_jobs=NJOB, random_state=SEED).fit(A, Y)

    # 对新样本给出标准化预测与等级经验分布
    def predict(self, new_df, new_style=None):
        keep = self.keep
        # 哨兵填充返回 (训练折, 测试折, 哨兵值)，这里要第二个
        _, B, _ = FE.sentinel_fill(self.train_X, new_df[keep].to_numpy(float))
        _, MB, _ = FE.miss_indicators(self.train_X, new_df[keep].to_numpy(float))
        B = np.hstack([B, MB])
        if self.use_style:
            st = np.array([new_style] * len(new_df)) if new_style is not None else \
                new_df['tasting_category_fine'].to_numpy()
            MS = np.zeros((len(B), len(self.cats)))
            for i, s in enumerate(st):
                if s in self.cats:
                    MS[i, self.cats.index(s)] = 1.0
            B = np.hstack([B, MS])
        z = self.m.predict(B)
        _, exp, hist = expect_grade(self.m, B, self.mu, self.sg)
        return z, exp, hist


# 全样本拟合，返回可复用预测器；预测器对原始化合物表给出 (连续分数, 期望等级, 中位数等级)
def fit_final(chem=None, sens=None, n_est=N_EST_FINAL):
    chem = load_chem() if chem is None else chem
    sens = load_sensory() if sens is None else sens
    v0 = _View(chem, sens, False, n_est)
    v1 = _View(chem, sens, True, n_est)
    mu, sg = load_scale()

    def predict(new_df, new_style=None):
        z0, _, h0 = v0.predict(new_df)
        z1, _, h1 = v1.predict(new_df, new_style)
        z = 0.5 * (z0 + z1)
        h = 0.5 * (h0 + h1)
        s = z_to_score(z, mu, sg)
        exp = score_to_grade(z_to_score(np.tensordot(h, np.arange(7.0), axes=([2], [0])), mu, sg))
        med = median_grade(h)
        return s, exp, med

    return predict

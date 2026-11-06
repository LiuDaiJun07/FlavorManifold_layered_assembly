# -*- coding: utf-8 -*-
# 05 单模型消融：E0 化合物 / E1 +风格 / E2 +双曲 / E5 双曲退化
# 协议：分层 10 折 OOF（FOLDS 可覆盖），所有拟合在折内（修 D2）
# 树数由 common.trees() 统一决定（修 D7：不再各处散落 300/800/1200，也不再声称网格搜索）
# 误差在三口径下并列报告（修 D14），主口径 sd
import importlib
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
FE = importlib.import_module('03_features')
from common import (OUT, SEED, TARGET_COLS, calib_params, ensure_dirs, expect_grade,
                    load_chem, load_sensory, load_sensory_grade, macro_r2, mae_caliber,
                    median_grade, r2_groups, split_folds, trees)

N_EST_CV = 300
N_EST_FINAL = 800
MF = 0.5
MSL = 1
NJOB = int(os.environ.get('NJOB', '1'))
CONFIGS = {
    'E0': ('chem', 'miss'),
    'E1': ('chem', 'miss', 'style'),
    'E2': ('chem', 'miss', 'hyp', 'style'),
    'E5': ('chem', 'miss', 'hyp', 'style'),
}
CS = {'E2': 1.0, 'E5': 1e-8}
WANT = [s for s in os.environ.get('ABL_CONFIGS', 'E0,E1,E2,E5').split(',') if s]
L = []


# 追加一行到报告并打印
def P(s=''):
    L.append(str(s))
    print(s)


ensure_dirs()
chem = load_chem()
sens = load_sensory()
Y = sens[TARGET_COLS].to_numpy(dtype=float)
G = load_sensory_grade()
style = sens['tasting_category_fine'].astype(str).to_numpy()
N = len(Y)

# 分层键 = 风格 + 目标均值高低，既保证 22 个风格都进每折，也让折内目标分布接近
gfolds = (G.mean(axis=1) >= 3.0).astype(int)
folds = split_folds(np.char.add(style, np.where(gfolds > 0, '_h', '_l')))
P(f'# 05 单模型消融（{len(folds)} 折 OOF，主口径 sd）')
P()
P(f'- 树数：CV {trees(N_EST_CV)} 棵、留出 {trees(N_EST_FINAL)} 棵（TREE_SCALE={os.environ.get("TREE_SCALE", "1.0")}）')
P(f'- 并行度 NJOB={NJOB}；无网格搜索，超参取开发文档固定值')
P('- 折内拟合：众数占比裁剪只用训练折行（修 D2）；反标准化参数按折内统计量（修 D14）')
P()

oof = {}
info = {}
for name in WANT:
    blocks = CONFIGS[name]
    c = CS.get(name, 1.0)
    pred = np.zeros_like(Y)
    exp = np.zeros_like(Y)
    med = np.zeros(Y.shape, dtype=int)
    hist = np.zeros((N, len(TARGET_COLS), 7))
    for f, (tr, te) in enumerate(folds):
        keep, _ = FE.prune_cols(chem, rows=tr)
        Xtr, Xte, names = FE.build(chem, style, tr, te, blocks=blocks, c=c, feat_cols=keep)
        mu, sg = calib_params(Y[tr], 'sd')
        m = ExtraTreesRegressor(n_estimators=trees(N_EST_CV), max_features=MF,
                                min_samples_leaf=MSL, random_state=SEED + f, n_jobs=NJOB)
        m.fit(Xtr, Y[tr])
        pred[te] = m.predict(Xte)
        g_, exp[te], h_ = expect_grade(m, Xte, mu, sg)
        med[te] = median_grade(h_)
        hist[te] = h_
        print(f'  {name} fold {f + 1}/{len(folds)}  n_feat={Xtr.shape[1]}', flush=True)
    mm = macro_r2(Y, pred)
    mac, weak, strong, _ = r2_groups(Y, pred)
    maes = mae_caliber(Y, pred, Y)
    info[name] = {'macro_r2': mac, 'weak_r2': weak, 'strong_r2': strong,
                  'mae_unit': maes['unit'], 'mae_sd': maes['sd'], 'mae_mm': maes['minmax'],
                  'r2_dim': {TARGET_COLS[j]: float(mm[j]) for j in range(len(TARGET_COLS))}}
    oof[name] = dict(pred=pred, exp=exp, med=med, hist=hist)
    P(f'## {name}')
    P()
    P(f"- 宏 R2 = {mac:.4f}；12 弱维均值 = {weak:.4f}；5 强维均值 = {strong:.4f}")
    P(f"- MAE（unit / sd / minmax）= {maes['unit']:.4f} / {maes['sd']:.4f} / {maes['minmax']:.4f}")
    P()

P('## 汇总')
P()
P('| 配置 | 宏 R2 | 12 弱维 | 5 强维 | MAE(unit) | MAE(sd) | MAE(minmax) |')
P('| --- | --- | --- | --- | --- | --- | --- |')
for name in WANT:
    d = info[name]
    P(f"| {name} | {d['macro_r2']:.4f} | {d['weak_r2']:.4f} | {d['strong_r2']:.4f} | "
      f"{d['mae_unit']:.4f} | {d['mae_sd']:.4f} | {d['mae_mm']:.4f} |")
P()
P('注：v1 曾写「逐维独立建模宏 R2 0.2393 低于联合 0.2507」，该比较未做任何调参，')
P('结论不可靠；本版删除该论断，不再声称逐维建模更差（修 D6）。')
P()

# 70/30 分层留出：仅用于核对留出口径与 10 折 OOF 是否同量级
idx = np.arange(N)
rs = np.random.RandomState(SEED)
hold = np.zeros(N, dtype=bool)
for s in np.unique(style):
    ii = idx[style == s].copy()
    rs.shuffle(ii)
    hold[ii[:max(1, int(round(0.3 * len(ii))))]] = True
tr_h = np.where(~hold)[0]
te_h = np.where(hold)[0]
keep_h, _ = FE.prune_cols(chem, rows=tr_h)
Xtr, Xte, _ = FE.build(chem, style, tr_h, te_h, blocks=CONFIGS['E1'], feat_cols=keep_h)
m_h = ExtraTreesRegressor(n_estimators=trees(N_EST_FINAL), max_features=MF,
                          min_samples_leaf=MSL, random_state=SEED, n_jobs=NJOB)
m_h.fit(Xtr, Y[tr_h])
pred_h = m_h.predict(Xte)
hold_r2 = float(macro_r2(Y[te_h], pred_h).mean())
P('## 70/30 分层留出（E1，仅作口径核对）')
P()
P(f'- 留出宏 R2 = {hold_r2:.4f}（样本量小，波动大，不作为验收依据）')
P()

np.savez_compressed(os.path.join(OUT, '05_oof.npz'),
                    y=Y, g=G, style=style, tr=tr_h, te=te_h, keep=np.array(keep_h),
                    **{f'{k}_{kk}': vv for k, d in oof.items() for kk, vv in d.items()})
with open(os.path.join(OUT, '05_metrics.json'), 'w', encoding='utf-8') as f:
    json.dump({'n_folds': len(folds), 'configs': info, 'holdout_r2_E1': hold_r2,
               'trees_cv': trees(N_EST_CV), 'trees_final': trees(N_EST_FINAL)},
              f, ensure_ascii=False, indent=2)
pd.DataFrame([{'cfg': k, **{kk: vv for kk, vv in v.items() if kk != 'r2_dim'}}
              for k, v in info.items()]).to_csv(
    os.path.join(OUT, '05_metrics.csv'), index=False, encoding='utf-8-sig')
with open(os.path.join(OUT, '05_train_et.md'), 'w', encoding='utf-8') as f:
    f.write('\n'.join(L))
print()
print('05 完成，输出写入', OUT)

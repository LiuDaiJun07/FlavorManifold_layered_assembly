# -*- coding: utf-8 -*-
# 06b2 内层基座：为 LSF/ZITS 的内层选择提供「不含外层验证折」的折外基座预测
# 为什么需要：06b 的 base 是全局 10 折 OOF。外层折 f 的训练行 i 属于折 g，
#   base[i] 由「排除 g、包含 f」的模型给出，已经吃过 f 的标签；
#   拿它当内层选择的基线与残差，等于让外层验证折的标签参与内层选参（漏洞 02）。
# 做法：外层折 f 的训练行 tr 再切 5 折；每折只在 pool 上拟合与 06b 完全同构的五成员，
#   预测 val，拼成 ib_f（只在 tr 上有值，外层测试折留 NaN），供 06c/06d 的内层使用。
# 成本：10 外层 x 5 内层 x 五成员；结果缓存 npz，06c/06d 重跑不必重算。
import importlib
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
FE = importlib.import_module('03_features')
MC = importlib.import_module('06b_member_core')
from common import (OUT, TARGET_COLS, ensure_dirs, load_chem, load_sensory,
                    load_sensory_grade, split_folds, trees)

ensure_dirs()
L = []


# 追加一行到报告并打印
def P(s=''):
    L.append(str(s))
    print(s)


N_INNER = 5

chem = load_chem()
sens = load_sensory()
Y = sens[TARGET_COLS].to_numpy(dtype=float)
G = load_sensory_grade()
style = sens['tasting_category_fine'].astype(str).to_numpy()
N = len(Y)
# 折划分与 06b 完全同源：同一个 split_folds + 同一个分层串 + 同一个 SEED
gfolds = (G.mean(axis=1) >= 3.0).astype(int)
folds = split_folds(np.char.add(style, np.where(gfolds > 0, '_h', '_l')))
fold_id = np.full(N, -1, dtype=int)
for f, (tr, te) in enumerate(folds):
    fold_id[te] = f

P(f'# 06b2 内层基座（外层 {len(folds)} 折 x 内层 {N_INNER} 折）')
P()
P(f'- 内层成员与 06b 完全同构：M1–M5，整表 {trees(MC.N_EST)} 棵、逐维 {trees(MC.N_EST_DIM)} 棵')
P('- ib_f 只在「外层折 f 的训练行」上有值；外层测试行留 NaN，绝不用来选参')
P()

store = {'y': Y, 'g': G, 'fold_id': fold_id, 'n_inner': np.array([N_INNER])}
t_all = time.time()
for f, (tr, te) in enumerate(folds):
    t0 = time.time()
    keep, _ = FE.prune_cols(chem, rows=tr)
    ib = np.full((N, len(TARGET_COLS)), np.nan)
    ibf = np.full(N, -1, dtype=int)
    ifold = split_folds(style[tr], n_folds=N_INNER)
    for i, (atr, ate) in enumerate(ifold):
        pool, val = tr[atr], tr[ate]
        # 特征列沿用外层折的 keep：内层基线要对齐「外层实际使用的基座」，只把拟合缩到 pool
        Atr, Ate, _ = FE.build(chem, style, pool, val, blocks=('chem', 'miss'), feat_cols=keep)
        pt, _ = MC.fit_members(Atr, Ate, Y[pool], len(keep),
                               seed_off=1000 + f * 10 + i, keep_trees=False)
        acc = np.zeros((len(val), len(TARGET_COLS)))
        for k in MC.MEMBERS:
            acc += pt[k]
        ib[val] = acc / len(MC.MEMBERS)
        ibf[val] = i
    store[f'ib_{f}'] = ib
    store[f'ibf_{f}'] = ibf
    n_ok = int(np.isfinite(ib[:, 0]).sum())
    P(f'- 外层折 {f + 1}/{len(folds)}：内层覆盖行 {n_ok}（应等于训练行 {len(tr)}），'
      f'用时 {time.time() - t0:.0f}s')
    if n_ok != len(tr):
        raise RuntimeError(f'外层折 {f} 的内层覆盖不完整：{n_ok} != {len(tr)}')

P()
P(f'- 总用时 {time.time() - t_all:.0f}s')
np.savez_compressed(os.path.join(OUT, '06b2_inner_base.npz'), **store)
with open(os.path.join(OUT, '06b2_inner_base.md'), 'w', encoding='utf-8') as f:
    f.write('\n'.join(L))
with open(os.path.join(OUT, '06b2_inner_base.json'), 'w', encoding='utf-8') as f:
    json.dump({'n_folds': len(folds), 'n_inner': N_INNER,
               'covered': [int(np.isfinite(store[f'ib_{f}'][:, 0]).sum()) for f in range(len(folds))],
               'seconds': float(time.time() - t_all)}, f, ensure_ascii=False, indent=2)
print()
print('06b2 完成，输出写入', OUT)

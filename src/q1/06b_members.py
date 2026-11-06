# -*- coding: utf-8 -*-
# 06b 五成员融合基座：M1–M5 在统一的 10 折 OOF 协议下出折外预测并缓存近邻结构
# M1 ET mf=0.5；M2 ET mf=0.8 + 目标按训练折列 sd 归一化；M3 ET 打目标 PCA30 再逆变换
# M4 ET 逐维折内 |corr| top60；M5 RF 逐维折内 |corr| top60
# 折内拟合一律只用训练折（裁剪、秩变换、标准化、相关系数筛选、邻居池），修 D2/G4
# 五成员定义与直方图生成统一走 06b_member_core：直方图由逐树预测统一算出，
# 不存在「某个成员只写点预测、忘写直方图」的分支（漏洞 01 的结构性修复）
import importlib
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
FE = importlib.import_module('03_features')
MC = importlib.import_module('06b_member_core')
from common import (OUT, TARGET_COLS, WEAK_DIMS, calib_params, ensure_dirs, load_chem,
                    load_sensory, load_sensory_grade, macro_r2, r2_groups, score_to_grade,
                    split_folds, trees, z_to_score)

ensure_dirs()
L = []


# 追加一行到报告并打印
def P(s=''):
    L.append(str(s))
    print(s)


# 近邻上限：06c/06d 从缓存里直接取这么多邻居，多于这个数不再重算
KMAX = 25

chem = load_chem()
sens = load_sensory()
Y = sens[TARGET_COLS].to_numpy(dtype=float)
G = load_sensory_grade()
style = sens['tasting_category_fine'].astype(str).to_numpy()
N = len(Y)
gfolds = (G.mean(axis=1) >= 3.0).astype(int)
folds = split_folds(np.char.add(style, np.where(gfolds > 0, '_h', '_l')))
fold_id = np.full(N, -1, dtype=int)
for f, (tr, te) in enumerate(folds):
    fold_id[te] = f

P(f'# 06b 五成员融合基座（{len(folds)} 折 OOF）')
P()
P(f'- 树数：整表成员 {trees(MC.N_EST)} 棵，逐维成员 {trees(MC.N_EST_DIM)} 棵；近邻上限 K={KMAX}')
P('- 所有拟合在折内：裁剪、秩变换、标准化、|corr| 筛选、邻居池都只看训练折')
P('- 五个成员的等级直方图都来自逐树预测（同一入口），行和恒为 1')
P()

M = {k: np.zeros_like(Y) for k in MC.MEMBERS}
H = {k: np.zeros((N, len(TARGET_COLS), 7)) for k in MC.MEMBERS}
cache = {}
for f, (tr, te) in enumerate(folds):
    keep, _ = FE.prune_cols(chem, rows=tr)
    Xtr, Xte, names = FE.build(chem, style, tr, te, blocks=('chem','miss'), feat_cols=keep)
    n_chem = len(keep)
    mu, sg = calib_params(Y[tr], 'sd')
    Xc_tr = Xtr[:, :n_chem]
    Xc_te = Xte[:, :n_chem]

    # 五成员一次拟合：点预测写 M，逐树预测经同一个入口转成等级直方图写 H
    pt, tz = MC.fit_members(Xtr, Xte, Y[tr], n_chem, seed_off=f, keep_trees=True)
    for k in MC.MEMBERS:
        M[k][te] = pt[k]
        H[k][te] = MC.member_hist(tz[k], mu, sg)

    # 近邻缓存：折内秩变换 + 标准化后算欧氏距离，训练池互相排除自身
    A, B = FE.rank_transform(Xc_tr, Xc_te)
    (Itr, Dtr), (Ite, Dte) = FE.knn_neighbors(A, B, KMAX)
    # 零占比按「折内 calib 还原出来的等级=0」统计，不引用全量 calib 构造的 G，
    # 否则 06c/06d 的开关阈值会间接带上全量标签信息（漏洞 04 的 b 项）
    g_tr = score_to_grade(z_to_score(Y[tr], mu, sg))
    cache[f'mu_{f}'] = mu
    cache[f'sg_{f}'] = sg
    cache[f'zero_{f}'] = np.mean(g_tr == 0, axis=0).astype(float)
    cache[f'knn_tr_idx_{f}'] = tr[Itr]
    cache[f'knn_tr_d_{f}'] = Dtr
    cache[f'knn_te_idx_{f}'] = tr[Ite]
    cache[f'knn_te_d_{f}'] = Dte
    print(f'  fold {f + 1}/{len(folds)}  n_feat={Xtr.shape[1]} 完成', flush=True)

# 直方图必须是概率分布：任何成员漏填都会让行和偏离 1，这里直接炸掉而不是静默继续
for k in MC.MEMBERS:
    ssum = H[k].sum(axis=2)
    if not np.allclose(ssum, 1.0, atol=1e-9):
        raise RuntimeError(f'{k} 直方图行和不为 1：min={ssum.min():.6f} max={ssum.max():.6f}')
mix = np.mean([H[k] for k in MC.MEMBERS], axis=0)
if not np.allclose(mix.sum(axis=2), 1.0, atol=1e-9):
    raise RuntimeError('五成员混合直方图行和不为 1')

base = np.mean([M[k] for k in M], axis=0)
P('## 成员与基座（等权五成员）')
P()
P('| 名称 | 宏 R2 | 12 弱维 | 5 强维 |')
P('| --- | --- | --- | --- |')
for k in ['M1', 'M2', 'M3', 'M4', 'M5', 'BASE']:
    v = base if k == 'BASE' else M[k]
    mac, weak, strong, _ = r2_groups(Y, v)
    P(f'| {k} | {mac:.4f} | {weak:.4f} | {strong:.4f} |')
P()
mac_b, weak_b, strong_b, r2b = r2_groups(Y, base)
zero_mean = np.mean([cache[f'zero_{f}'] for f in range(len(folds))], axis=0)
P(f'- 基座宏 R2 = {mac_b:.4f}（v1 四成员 0.2736，五成员口径下重新给出）')
P(f'- 基座 12 弱维均值 = {weak_b:.4f}，5 强维均值 = {strong_b:.4f}')
P(f'- 零占比 ≥0.40 的维度数 = {int((zero_mean >= 0.40).sum())}，其中落在 12 弱维里的 = '
  f'{int(sum(zero_mean[TARGET_COLS.index(c)] >= 0.40 for c in WEAK_DIMS))}')
P(f'- 直方图自检：五个成员与混合分布的行和 min={mix.sum(axis=2).min():.6f} '
  f'max={mix.sum(axis=2).max():.6f}')
P()

pd.DataFrame({'dim': TARGET_COLS, 'zero_share': zero_mean, 'base_r2': r2b}).to_csv(
    os.path.join(OUT, '06b_zero_share.csv'), index=False, encoding='utf-8-sig')
np.savez_compressed(os.path.join(OUT, '06b_members.npz'), y=Y, g=G, style=style,
                    fold_id=fold_id, base=base,
                    **{f'{k}_pred': v for k, v in M.items()},
                    **{f'{k}_hist': v for k, v in H.items()}, **cache)
with open(os.path.join(OUT, '06b_members.md'), 'w', encoding='utf-8') as f:
    f.write('\n'.join(L))
with open(os.path.join(OUT, '06b_members.json'), 'w', encoding='utf-8') as f:
    json.dump({'macro_r2': mac_b, 'weak_r2': weak_b, 'strong_r2': strong_b,
               'members': {k: float(macro_r2(Y, M[k]).mean()) for k in M},
               'zero_dims_ge_040': int((zero_mean >= 0.40).sum()),
               'n_folds': len(folds), 'n_trees': trees(MC.N_EST)}, f, ensure_ascii=False, indent=2)
print()
print('06b 完成，输出写入', OUT)

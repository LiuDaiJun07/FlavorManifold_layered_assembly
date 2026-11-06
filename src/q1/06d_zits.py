# -*- coding: utf-8 -*-
# 06d ZITS 零膨胀两阶段：逐维开关只由「训练折零占比 ≥ 0.40」决定（G2）
# 阶段 A：ET 分类器给 P(非零)=p（n_estimators=500, max_features=0.3, class_weight='balanced'）
# 阶段 B：只在正样本上回归幅度 m；正样本少时向全局均值收缩，K=20
# 合成 ŷ = p·m；等级分布用逐树经验分布 P(m≥k−0.5)、P(y≥k)=p·P(m≥k−0.5)
# 内层 5 折核对：若某维内层不优于单阶段，则该维回退到基座（不启用 ZITS）
# 零组一律按「分数空间 0 级」判定，幅度与阈值 k−0.5 也都在分数空间；
# 内层比较的基座取 06b2 的 ib_f，不用 06b 的全局 OOF（漏洞 02、03 的修复点）
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import importlib
FE = importlib.import_module('03_features')
from common import (OUT, SEED, TARGET_COLS, calib_params, ensure_dirs, load_chem, load_sensory,
                    macro_r2, r2_groups, score_to_grade, split_folds, trees)

ensure_dirs()
L = []


# 追加一行到报告并打印
def P(s=''):
    L.append(str(s))
    print(s)


NJOB = int(os.environ.get('NJOB', '1'))
ZERO_SWITCH = 0.40
K_SHRINK = 20
N_CLS = 500
N_REG = 150
N_INNER = 60


# 单维两阶段：返回 (z 空间点预测, 分数空间逐树幅度, 非零概率, 正样本数, 全局幅度)
# 零组按「分数空间 0 级」判定：4.csv 每列有 100–250 个不同取值，
# 用 y<=min(y) 只能圈到 1–4 个样本（实测 166 个 0 级的维仅命中 1 个），不能当零组
def zits_fit(Xtr, Xte, ytr, mu_j, sg_j, t_cls=N_CLS, t_reg=N_REG, seed=SEED):
    s = np.clip(mu_j + sg_j * ytr, 0.0, 6.0)
    pos = score_to_grade(s) >= 1
    n_pos = int(pos.sum())
    mglob = float(np.mean(s[pos])) if n_pos > 0 else 0.0
    p_pos = np.zeros(Xte.shape[0])
    if n_pos >= 3 and n_pos < len(s):
        clf = ExtraTreesClassifier(n_estimators=t_cls, max_features=0.3,
                                   class_weight='balanced', min_samples_leaf=1,
                                   random_state=seed, n_jobs=NJOB).fit(Xtr, pos.astype(int))
        if 1 in list(clf.classes_):
            p_pos = clf.predict_proba(Xte)[:, list(clf.classes_).index(1)]
        else:
            p_pos = np.ones(Xte.shape[0])
    elif n_pos >= 3:
        # 训练折全为正样本，存在性无信息，恒判为存在
        p_pos = np.ones(Xte.shape[0])
    if n_pos >= 3:
        reg = ExtraTreesRegressor(n_estimators=t_reg, max_features=0.5, min_samples_leaf=1,
                                  random_state=seed + 1, n_jobs=NJOB).fit(Xtr[pos], s[pos])
        T = np.column_stack([e.predict(Xte) for e in reg.estimators_])
    else:
        T = np.full((Xte.shape[0], 1), mglob)
    mloc = T.mean(axis=1)
    if n_pos < 30:
        # 正样本稀疏时向全局均值收缩，避免小样本回归外推
        m = (n_pos * mloc + K_SHRINK * mglob) / (n_pos + K_SHRINK)
        T = T + (m - mloc)[:, None]
        mloc = m
    # 合成 E[y] = p·m（分数空间），再搬回 z 空间与其余成员同尺度比较
    return (p_pos * mloc - mu_j) / sg_j, T, p_pos, n_pos, mglob


# 由存在概率与逐树幅度写等级分布：P(y=0)=1−p；P(y≥k)=p·P(m≥k−0.5)，k=1..6
# 阈值 k−0.5 只在分数空间上有意义，传进来的 T 必须是分数空间
def zits_hist(T, p_pos):
    n = T.shape[0]
    ge = np.ones((n, 7))
    for k in range(1, 7):
        ge[:, k] = np.clip(p_pos * np.mean(T >= (k - 0.5), axis=1), 0.0, 1.0)
    ge = np.minimum.accumulate(ge, axis=1)
    hist = np.diff(np.column_stack([np.zeros(n), ge]), axis=1)
    return np.clip(hist, 0.0, 1.0)


chem = load_chem()
sens = load_sensory()
Y = sens[TARGET_COLS].to_numpy(dtype=float)
style = sens['tasting_category_fine'].astype(str).to_numpy()
Z = np.load(os.path.join(OUT, '06b_members.npz'), allow_pickle=False)
IB = np.load(os.path.join(OUT, '06b2_inner_base.npz'), allow_pickle=False)
base = Z['base']
fold_id = Z['fold_id']
nf = int(fold_id.max()) + 1
folds = [(np.where(fold_id != f)[0], np.where(fold_id == f)[0]) for f in range(nf)]

P('# 06d ZITS 零膨胀两阶段')
P()
P(f'- 开关：训练折零占比 ≥ {ZERO_SWITCH}（唯一硬判据，R2/AUC 只写报告）')
P(f'- 分类器 {trees(N_CLS)} 棵 mf=0.3 balanced；回归器 {trees(N_REG)} 棵；K={K_SHRINK}')
P('- 零组 = 分数空间 0 级（不是 z 列最小值）；幅度与阈值 k−0.5 均在分数空间')
P('- 内层核对的基座取 06b2 的 ib_f（不含外层验证折信息）')
P()

pred = base.copy()
hist_z = np.zeros((len(Y), len(TARGET_COLS), 7))
on = np.zeros(len(TARGET_COLS), dtype=bool)
# 逐折记录哪些维在内层核对里赢过基座；只有「每一折都赢」的维才允许整体替换，
# 否则某一折的该维会退回基座点预测却没有对应直方图，06e 的分布就残缺了
used = np.zeros((nf, len(TARGET_COLS)), dtype=bool)
zstore = {}
rows = []
zero_mean = np.mean([Z[f'zero_{f}'] for f in range(nf)], axis=0)
for f, (tr, te) in enumerate(folds):
    keep, _ = FE.prune_cols(chem, rows=tr)
    Xtr, Xte, _ = FE.build(chem, style, tr, te, blocks=('chem', 'miss'), feat_cols=keep)
    zero = Z[f'zero_{f}']
    ib = IB[f'ib_{f}']
    mu_f = Z[f'mu_{f}']
    sg_f = Z[f'sg_{f}']
    if not np.all(np.isfinite(ib[tr])):
        raise RuntimeError(f'外层折 {f} 的内层基座在训练行上不完整')
    # 内层 5 折的裁剪与特征只建一次，所有开关维复用；否则 35 个维会重复建 175 次特征
    inner = []
    for atr, ate in split_folds(style[tr], n_folds=5):
        pool, val = tr[atr], tr[ate]
        kp, _ = FE.prune_cols(chem, rows=pool)
        Atr, Ate, _ = FE.build(chem, style, pool, val, blocks=('chem', 'miss'), feat_cols=kp)
        inner.append((val, Atr, Ate, pool))
    for j, c in enumerate(TARGET_COLS):
        if zero[j] < ZERO_SWITCH:
            continue
        pj, T, pj_pos, n_pos, mglob = zits_fit(Xtr, Xte, Y[tr, j], mu_f[j], sg_f[j])
        # 内层核对：内层 5 折比较 ZITS 与基座在本维的 R2（特征已在外层折里预建好）
        z_in = np.zeros(len(tr))
        b_in = np.zeros(len(tr))
        p_in = np.zeros(len(tr))
        y_in = np.zeros(len(tr), dtype=int)
        for val, Atr, Ate, pool in inner:
            pos_in = np.searchsorted(tr, val)
            mi, si = calib_params(Y[pool, :][:, [j]], 'sd')
            zz, _, pp, _, _ = zits_fit(Atr, Ate, Y[pool, j], float(mi[0]), float(si[0]),
                                       t_cls=trees(N_INNER), t_reg=trees(N_INNER))
            z_in[pos_in] = zz
            p_in[pos_in] = pp
            b_in[pos_in] = ib[val, j]
            y_in[pos_in] = score_to_grade(np.clip(float(mi[0]) + float(si[0]) * Y[val, j], 0.0, 6.0))
        rz = 1.0 - np.sum((Y[tr, j] - z_in) ** 2) / max(np.sum((Y[tr, j] - Y[tr, j].mean()) ** 2), 1e-12)
        rb = 1.0 - np.sum((Y[tr, j] - b_in) ** 2) / max(np.sum((Y[tr, j] - Y[tr, j].mean()) ** 2), 1e-12)
        use = rz > rb
        try:
            auc = float(roc_auc_score((y_in >= 1).astype(int), p_in))
        except ValueError:
            auc = float('nan')
        rows.append({'fold': f, 'dim': c, 'zero_share': float(zero[j]), 'n_pos': n_pos,
                     'm_glob': mglob, 'auc': auc,
                     'inner_r2_zits': float(rz), 'inner_r2_base': float(rb),
                     'on': bool(use)})
        if use:
            used[f, j] = True
            zstore[(f, j)] = (pj, zits_hist(T, pj_pos))
    print(f'  fold {f + 1}/{nf} 完成', flush=True)

# 只有全折一致的维才整体替换：部分折赢、部分折输的维按基座交付（G5 同口径）
on = used.all(axis=0)
nf_part = int(((used.any(axis=0)) & (~on)).sum())
for (f, j), (pj, hz) in zstore.items():
    if on[j]:
        te = folds[f][1]
        pred[te, j] = pj
        hist_z[te, j] = hz

T = pd.DataFrame(rows)
T.to_csv(os.path.join(OUT, '06d_zits.csv'), index=False, encoding='utf-8-sig')
mac, weak, strong, r2 = r2_groups(Y, pred)
mac_b, weak_b, strong_b, r2_b = r2_groups(Y, base)
P('## 结果')
P()
P('| 方案 | 宏 R2 | 12 弱维 | 5 强维 |')
P('| --- | --- | --- | --- |')
P(f'| 基座 | {mac_b:.4f} | {weak_b:.4f} | {strong_b:.4f} |')
P(f'| 基座 + ZITS（替换开启维） | {mac:.4f} | {weak:.4f} | {strong:.4f} |')
P()
P(f'- 零占比 ≥0.40 的维度 = {int((zero_mean >= 0.40).sum())}；内层核对后真正开启 = {int(on.sum())}')
P(f"- 开启维：{', '.join(TARGET_COLS[j] for j in np.where(on)[0])}")
P(f'- 只有部分折赢过基座的维度 = {nf_part}（这些维按基座交付，避免点预测与直方图口径不一致）')
P()
P('## 逐维 AUC（内层折外，AUC<0.60 只是「该维存在性与化合物关系弱」的结论，不作开关判据）')
P()
P('| 维度 | 零占比 | AUC | 内层 R2（ZITS） | 内层 R2（基座） | 是否开启 |')
P('| --- | --- | --- | --- | --- | --- |')
for c in T['dim'].unique():
    g = T[T['dim'] == c]
    P(f"| {c} | {g['zero_share'].mean():.2f} | {g['auc'].mean():.3f} | "
      f"{g['inner_r2_zits'].mean():+.4f} | {g['inner_r2_base'].mean():+.4f} | "
      f"{'是' if on[TARGET_COLS.index(c)] else '否'} |")
P()
P(f"- AUC 中位数 = {T['auc'].median():.3f}，AUC<0.60 的维数 = {int((T.groupby('dim')['auc'].mean() < 0.60).sum())}")
P()

agg = T.groupby('dim').agg(zero_share=('zero_share', 'mean'), n_pos=('n_pos', 'mean'),
                           auc=('auc', 'mean'),
                           inner_r2_zits=('inner_r2_zits', 'mean'),
                           inner_r2_base=('inner_r2_base', 'mean'),
                           on=('on', 'mean')).reset_index()
agg['base_r2'] = [float(r2_b[TARGET_COLS.index(c)]) for c in agg['dim']]
agg['zits_r2'] = [float(r2[TARGET_COLS.index(c)]) for c in agg['dim']]
# 是否开启以「全折一致」为准，不用逐折均值（部分折开启不算开启）
agg['on'] = [bool(on[TARGET_COLS.index(c)]) for c in agg['dim']]
agg.to_csv(os.path.join(OUT, '06d_zits_dim.csv'), index=False, encoding='utf-8-sig')
np.savez_compressed(os.path.join(OUT, '06d_zits.npz'), y=Y, base=base, zits=pred,
                    hist_z=hist_z, on=on, fold_id=fold_id, zero_mean=zero_mean)
with open(os.path.join(OUT, '06d_zits.json'), 'w', encoding='utf-8') as f:
    json.dump({'base': [mac_b, weak_b, strong_b], 'zits': [mac, weak, strong],
               'n_switch': int((zero_mean >= 0.40).sum()), 'n_on': int(on.sum()),
               'dims_on': [TARGET_COLS[j] for j in np.where(on)[0]],
               'n_partial': nf_part}, f, ensure_ascii=False, indent=2)
with open(os.path.join(OUT, '06d_zits.md'), 'w', encoding='utf-8') as f:
    f.write('\n'.join(L))
print()
print('06d 完成，输出写入', OUT)

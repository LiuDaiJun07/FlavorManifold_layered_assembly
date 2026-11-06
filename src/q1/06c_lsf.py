# -*- coding: utf-8 -*-
# 06c LSF 局部收缩融合：三种 KNN 能力叠在五成员基座上
# ① 弱维局部收缩（只对训练折零占比≥0.40 的维；λ∈{0,.10,...,.35}，内层 5 折选）
# ② 残差邻域校正（残差取折外预测，β∈{0,.1,.2}，内层增益≥0.002 才启用）
# ③ 孤立度分层偏差（k 近邻平均距离三分位，减 0.5·b_layer，只对零占比≥0.20 的维）
# 同时产出两列：内层选出的逐维配置 / 全固定默认值（λ=0.15, β=0.1, k=10），供 G3 对照
# 内层选择用的基线一律取 06b2 的「不含外层验证折」内层基座 ib_f：
# 06b 的全局 OOF base 在训练行上来自「排除本行所在折、但包含本外层验证折」的模型，
# 拿它当内层基线与残差，等于让外层验证折的标签参与选参（漏洞 02 的修复点）
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import importlib
FE = importlib.import_module('03_features')
from common import (OUT, SEED, TARGET_COLS, ensure_dirs, load_chem, load_sensory,
                    macro_r2, r2_groups, split_folds)

ensure_dirs()
L = []


# 追加一行到报告并打印
def P(s=''):
    L.append(str(s))
    print(s)


KS = [10, 15, 20, 25]
LAMS = [0.0, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35]
BETAS = [0.0, 0.10, 0.20]
K_DEF, LAM_DEF, BETA_DEF = 10, 0.15, 0.10
GAIN_MIN = 0.002
ZERO_SWITCH = 0.40
ZERO_LAYER = 0.20


# 邻居加权均值：w=1/(d+eps)，取前 k 列
def local_mean(idx, dist, Yref, k):
    w = 1.0 / (dist[:, :k] + 1e-6)
    v = Yref[idx[:, :k], :]
    return (w[:, :, None] * v).sum(axis=1) / w.sum(axis=1, keepdims=True)


# 逐维 R2，用于内层选择
def r2_col(yt, yp, j):
    ss_res = float(np.sum((yt[:, j] - yp[:, j]) ** 2))
    ss_tot = float(np.sum((yt[:, j] - yt[:, j].mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0


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

P('# 06c LSF 局部收缩融合')
P()
P(f'- 基座 = 五成员等权；KS={KS}，LAMS={LAMS}，BETAS={BETAS}')
P('- kNN 只以三种能力出现：弱维局部收缩、残差邻域校正、孤立度分层偏差；')
P('  不做 kNN 等权并入基座，也不把 kNN 预测当作第六个成员——它的偏差与树模型强相关，等权并入只会稀释。')
P(f'- 启用阈值：内层增益 ≥ {GAIN_MIN}；局部收缩只对零占比 ≥ {ZERO_SWITCH} 的维')
P('- 内层基线与残差取 06b2 的 ib_f（只用外层训练行做 5 折内层），不含外层验证折信息')
P()

sel = base.copy()
dfn = base.copy()
rows = []
for f, (tr, te) in enumerate(folds):
    zero = Z[f'zero_{f}']
    keep, _ = FE.prune_cols(chem, rows=tr)
    Xc = chem[keep].to_numpy(dtype=float)

    # 内层 5 折：每次只用内层训练行建秩变换与邻居池，内层验证行只做查询
    ifold = split_folds(style[tr], n_folds=5)
    inner = []
    for atr, ate in ifold:
        pool, val = tr[atr], tr[ate]
        A, B = FE.rank_transform(Xc[pool], Xc[val])
        (_, _), (Iv, Dv) = FE.knn_neighbors(A, B, max(KS))
        inner.append((pool, val, pool[Iv], Dv))

    # 内层折编号必须与 06b2 的切分逐行一致，否则 ib_f 的行会错位
    ib = IB[f'ib_{f}']
    ibf = IB[f'ibf_{f}']
    for i, (pool, val, Iv, Dv) in enumerate(inner):
        if not np.all(ibf[val] == i):
            raise RuntimeError(f'外层折 {f} 的内层折 {i} 与 06b2 的切分不一致')
    if not np.all(np.isfinite(ib[tr])):
        raise RuntimeError(f'外层折 {f} 的内层基座在训练行上不完整')

    # 外层次：训练池自身三分位孤立度，用内层折外残差估分层偏差
    A_tr, _ = FE.rank_transform(Xc[tr], Xc[tr])
    (Itr, Dtr), _ = FE.knn_neighbors(A_tr, Xc[tr][:1], max(KS))
    iso_tr = Dtr[:, :K_DEF].mean(axis=1)
    q1, q2 = np.quantile(iso_tr, [1 / 3, 2 / 3])
    lay_tr = np.digitize(iso_tr, [q1, q2])
    # 残差只在外层训练行上有定义，其余行留在 0，邻居池本身只含训练行
    res_tr = Y[tr] - ib[tr]
    RG = np.zeros_like(Y)
    RG[tr] = res_tr

    # 外层测试行的近邻与孤立度（来自 06b 缓存）
    Ite = Z[f'knn_te_idx_{f}']
    Dte = Z[f'knn_te_d_{f}']
    iso_te = Dte[:, :K_DEF].mean(axis=1)
    lay_te = np.digitize(iso_te, [q1, q2])

    Ptr_sel = sel[te].copy()
    Ptr_def = dfn[te].copy()
    for j, c in enumerate(TARGET_COLS):
        Yj = Y[:, [j]]
        RGj = RG[:, [j]]
        bte = base[te, j]

        # ① 局部收缩：内层网格选 (k, λ)；邻居下标是全局行号，参考值也按全局下标取
        best = (-np.inf, K_DEF, 0.0)
        r0 = r2_col(Y[tr], ib[tr][:, [j]], 0)
        if zero[j] >= ZERO_SWITCH:
            for k in KS:
                for lam in LAMS:
                    pv = np.zeros(len(tr))
                    for pool, val, Iv, Dv in inner:
                        loc = local_mean(Iv, Dv, Yj, k)[:, 0]
                        pos = np.searchsorted(tr, val)
                        pv[pos] = ib[val, j] + lam * (loc - ib[val, j])
                    r = r2_col(Y[tr], pv[:, None], 0)
                    if r > best[0]:
                        best = (r, k, lam)
            gain1 = best[0] - r0
            if gain1 >= GAIN_MIN and best[2] > 0:
                loc = local_mean(Ite, Dte, Yj, best[1])[:, 0]
                Ptr_sel[:, j] = bte + best[2] * (loc - bte)
            k_s, lam_s = best[1], best[2]
        else:
            gain1, k_s, lam_s = 0.0, K_DEF, 0.0

        # 默认值列：不分维网格，一律 k=10, λ=0.15
        if zero[j] >= ZERO_SWITCH:
            loc = local_mean(Ite, Dte, Yj, K_DEF)[:, 0]
            Ptr_def[:, j] = bte + LAM_DEF * (loc - bte)

        # ② 残差邻域校正：β 在内层选；残差取折外预测（G1），不是样本内残差
        bestb = (r0, 0.0)
        for beta in BETAS:
            pv = np.zeros(len(tr))
            for pool, val, Iv, Dv in inner:
                locr = local_mean(Iv, Dv, RGj, K_DEF)[:, 0]
                pos = np.searchsorted(tr, val)
                pv[pos] = ib[val, j] + beta * locr
            r = r2_col(Y[tr], pv[:, None], 0)
            if r > bestb[0]:
                bestb = (r, beta)
        gain2 = bestb[0] - r0
        beta_s = bestb[1] if gain2 >= GAIN_MIN else 0.0
        if beta_s > 0:
            locr = local_mean(Ite, Dte, RGj, K_DEF)[:, 0]
            Ptr_sel[:, j] += beta_s * locr
        locr_def = local_mean(Ite, Dte, RGj, K_DEF)[:, 0]
        Ptr_def[:, j] += BETA_DEF * locr_def

        # ③ 孤立度分层偏差
        use_lay = zero[j] >= ZERO_LAYER
        if use_lay:
            bl = np.array([res_tr[lay_tr == t, j].mean() if np.any(lay_tr == t) else 0.0
                           for t in range(3)])
            Ptr_sel[:, j] -= 0.5 * bl[lay_te]
            Ptr_def[:, j] -= 0.5 * bl[lay_te]

        rows.append({'fold': f, 'dim': c, 'zero_share': float(zero[j]),
                     'gain_shrink': float(gain1), 'k_sel': int(k_s), 'lam_sel': float(lam_s),
                     'shrink_on': bool(gain1 >= GAIN_MIN and lam_s > 0),
                     'gain_resid': float(gain2), 'beta_sel': float(beta_s),
                     'layer_on': bool(use_lay)})
    sel[te] = Ptr_sel
    dfn[te] = Ptr_def
    print(f'  fold {f + 1}/{nf} 完成', flush=True)

mac_s, weak_s, strong_s, r2_s = r2_groups(Y, sel)
mac_d, weak_d, strong_d, r2_d = r2_groups(Y, dfn)
mac_b, weak_b, strong_b, r2_b = r2_groups(Y, base)
P('## 结果')
P()
P('| 方案 | 宏 R2 | 12 弱维 | 5 强维 |')
P('| --- | --- | --- | --- |')
P(f'| 基座（五成员） | {mac_b:.4f} | {weak_b:.4f} | {strong_b:.4f} |')
P(f'| LSF 内层选参 | {mac_s:.4f} | {weak_s:.4f} | {strong_s:.4f} |')
P(f'| LSF 全默认 | {mac_d:.4f} | {weak_d:.4f} | {strong_d:.4f} |')
P()
P(f'- 两列宏 R2 差 = {abs(mac_s - mac_d):.4f}（G3 阈值 0.010，超过即判选择过拟合，以默认值为准）')
P()

T = pd.DataFrame(rows)
T.to_csv(os.path.join(OUT, '06c_lsf.csv'), index=False, encoding='utf-8-sig')
agg = T.groupby('dim').agg(zero_share=('zero_share', 'mean'), shrink_on=('shrink_on', 'mean'),
                           k_sel=('k_sel', 'median'), lam_sel=('lam_sel', 'mean'),
                           gain_shrink=('gain_shrink', 'mean'), beta_sel=('beta_sel', 'mean'),
                           gain_resid=('gain_resid', 'mean'), layer_on=('layer_on', 'mean'),
                           base_r2=('fold', 'size')).reset_index()
agg['base_r2'] = [float(r2_b[TARGET_COLS.index(c)]) for c in agg['dim']]
agg['lsf_r2'] = [float(r2_s[TARGET_COLS.index(c)]) for c in agg['dim']]
agg.to_csv(os.path.join(OUT, '06c_lsf_dim.csv'), index=False, encoding='utf-8-sig')
np.savez_compressed(os.path.join(OUT, '06c_lsf.npz'), y=Y, base=base, lsf_sel=sel, lsf_def=dfn,
                    fold_id=fold_id)
with open(os.path.join(OUT, '06c_lsf.json'), 'w', encoding='utf-8') as f:
    json.dump({'base': [mac_b, weak_b, strong_b], 'sel': [mac_s, weak_s, strong_s],
               'def': [mac_d, weak_d, strong_d],
               'n_shrink_on': int(T.groupby('dim')['shrink_on'].max().sum()),
               'n_beta_on': int(T.groupby('dim')['beta_sel'].max().gt(0).sum())},
              f, ensure_ascii=False, indent=2)
with open(os.path.join(OUT, '06c_lsf.md'), 'w', encoding='utf-8') as f:
    f.write('\n'.join(L))
print()
print('06c 完成，输出写入', OUT)

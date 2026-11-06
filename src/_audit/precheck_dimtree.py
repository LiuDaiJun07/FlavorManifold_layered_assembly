import os
import sys
import json
import importlib
import numpy as np

# 维度树预检：不是建模，只回答两个问题
# 一、3.xlsx 名字里那棵「all -> 4 个子维」的树，在 4.csv 里到底成不成立
# 二、50 个感官维度之间的距离结构，双曲空间是否比欧氏空间嵌得更准
HERE = os.path.dirname(os.path.abspath(__file__))
Q1 = os.path.join(os.path.dirname(HERE), 'q1')
sys.path.insert(0, Q1)
import common as C
HYP = importlib.import_module('04_hyperbolic')
FE = importlib.import_module('03_features')

C.ensure_dirs()
sens = C.load_sensory()
Y = sens[C.TARGET_COLS].to_numpy(float)
cols = list(C.TARGET_COLS)
ix = {c: i for i, c in enumerate(cols)}

# 六个家族：父维 + 四个子维
# 注意 3.xlsx 的 summary 页把它写成 A_ester_fruity（单数），4.csv 里是 A_esters_fruity，按 4.csv 走
FAM = [
    ('A_malt', 'A_malt_all', ['A_malt_grain', 'A_malt_bread', 'A_malt_cara', 'A_malt_burn']),
    ('A_hops', 'A_hops_all', ['A_hops_citrus', 'A_hops_tropical', 'A_hops_noble', 'A_hops_woody']),
    ('A_esters', 'A_esters_all', ['A_esters_ethac', 'A_esters_isoaa', 'A_esters_flower', 'A_esters_fruity']),
    ('F_malt', 'F_malt_all', ['F_malt_grain', 'F_malt_bread', 'F_malt_cara', 'F_malt_burn']),
    ('F_hops', 'F_hops_all', ['F_hops_citrus', 'F_hops_tropical', 'F_hops_noble', 'F_hops_woody']),
    ('F_esters', 'F_esters_all', ['F_esters_ethac', 'F_esters_isoaa', 'F_esters_flower', 'F_esters_fruity']),
]
# 先确认名字全部对得上，避免 KeyError 掩盖真问题
for _, p, ch in FAM:
    assert p in ix, p
    for c in ch:
        assert c in ix, c

# 逐维 R2 低于 0.11 的 12 个弱维，用来判断「树能不能救弱维」
WEAK = ['clove', 'F_esters_flower', 'A_esters_flower', 'orange', 'stale_hops', 'A_esters_fruity',
        'A_malt_grain', 'barnyard', 'dms', 'diacetyl', 't2n', 'F_malt_grain']

# 带截距的线性回归，返回样本内 R2
def lin_r2(xmat, yvec):
    X = np.column_stack([np.ones(len(yvec)), xmat])
    coef, *_ = np.linalg.lstsq(X, yvec, rcond=None)
    pred = X @ coef
    ss_res = float(np.sum((yvec - pred) ** 2))
    ss_tot = float(np.sum((yvec - yvec.mean()) ** 2))
    return 1.0 - ss_res / max(ss_tot, 1e-12), coef


# 按家族切分做 5 折，验证「子维预测父维」不是靠样本内过拟合
def cv_r2(xmat, yvec, n_splits=5):
    n = len(yvec)
    order = np.arange(n)
    fold = order % n_splits
    out = np.zeros(n)
    for f in range(n_splits):
        tr = fold != f
        te = ~tr
        Xtr = np.column_stack([np.ones(tr.sum()), xmat[tr]])
        coef, *_ = np.linalg.lstsq(Xtr, yvec[tr], rcond=None)
        out[te] = np.column_stack([np.ones(te.sum()), xmat[te]]) @ coef
    ss_res = float(np.sum((yvec - out) ** 2))
    ss_tot = float(np.sum((yvec - yvec.mean()) ** 2))
    return 1.0 - ss_res / max(ss_tot, 1e-12)


def corr(a, b):
    a = a - a.mean()
    b = b - b.mean()
    d = float(np.sqrt((a * a).sum() * (b * b).sum()))
    return float((a * b).sum() / d) if d > 1e-12 else 0.0


print('=' * 78)
print('一、六个家族的父子关系')
print('=' * 78)
print(f"{'family':10s} {'r(p,mean)':>9s} {'r(p,max)':>9s} {'r(p,sum)':>9s} {'R2子->父':>9s} {'CV_R2':>7s}")
fam_rows = []
for name, par, ch in FAM:
    yp = Y[:, ix[par]]
    M = Y[:, [ix[c] for c in ch]]
    r_mean = corr(yp, M.mean(1))
    r_max = corr(yp, M.max(1))
    r_sum = corr(yp, M.sum(1))
    r2_lin, _ = lin_r2(M, yp)
    r2_cv = cv_r2(M, yp)
    print(f'{name:10s} {r_mean:9.4f} {r_max:9.4f} {r_sum:9.4f} {r2_lin:9.4f} {r2_cv:7.4f}')
    fam_rows.append(dict(family=name, parent=par, children=ch,
                         r_mean=r_mean, r_max=r_max, r_sum=r_sum,
                         r2_lin=r2_lin, r2_cv=r2_cv))

print()
print('=' * 78)
print('二、兄弟维 vs 跨家族子维的平均相关（树的判别特征）')
print('=' * 78)
sib, cross = [], []
all_children = [c for _, _, ch in FAM for c in ch]
for name, par, ch in FAM:
    for i in range(4):
        for j in range(i + 1, 4):
            sib.append(corr(Y[:, ix[ch[i]]], Y[:, ix[ch[j]]]))
for i, a in enumerate(all_children):
    for b in all_children[i + 1:]:
        fam_a = [n for n, _, ch in FAM if a in ch][0]
        fam_b = [n for n, _, ch in FAM if b in ch][0]
        if fam_a != fam_b:
            cross.append(corr(Y[:, ix[a]], Y[:, ix[b]]))
sib = np.array(sib)
cross = np.array(cross)
print(f'兄弟维相关：均值 {sib.mean():.4f}  中位 {np.median(sib):.4f}  n={len(sib)}')
print(f'跨家族子维相关：均值 {cross.mean():.4f}  中位 {np.median(cross):.4f}  n={len(cross)}')
ratio = float(sib.mean() / max(abs(cross.mean()), 1e-9))
print(f'比值（兄弟/跨家族）= {ratio:.2f}  （>1.5 说明家族结构真实）')

print()
print('=' * 78)
print('三、50 个维度的距离结构：双曲 vs 欧氏 谁嵌得更准')
print('=' * 78)
ct = np.corrcoef(Y.T)
D = 1.0 - ct
D = np.maximum(D, 0.0)
np.fill_diagonal(D, 0.0)
# 先把距离统一归一，否则两种嵌入的应力不可比
nz = D[~np.eye(len(D), dtype=bool)]
D = D / float(nz.mean())
n = len(D)
D2 = D * D
J = np.eye(n) - np.ones((n, n)) / n
B = -0.5 * J @ D2 @ J
w, V = np.linalg.eigh(B)
order = np.argsort(w)[::-1]
eps = 1e-6
mds_rows = []
for d in (2, 4, 8):
    Xe = V[:, order[:d]] * np.sqrt(np.maximum(w[order[:d]], 0.0))[None, :]
    De = np.sqrt(np.maximum(((Xe[:, None, :] - Xe[None, :, :]) ** 2).sum(-1), 0.0))
    se = float(np.mean((De - D) ** 2 / (D * D + eps)))
    Zh, lh = HYP.fit_poincare(D, dim=d, c=1.0, iters=800, lr=0.02, seed=C.SEED)
    Dh = HYP.poincare_dist(Zh, Zh, 1.0, same=True)
    sh = float(np.mean((Dh - D) ** 2 / (D * D + eps)))
    mds_rows.append(dict(dim=d, stress_euclid=se, stress_hyp=sh, ratio=se / max(sh, 1e-12)))
    print(f'd={d}: 欧氏应力 {se:.5f}   双曲应力 {sh:.5f}   欧氏/双曲 = {se / max(sh, 1e-12):.3f}')

print()
print('=' * 78)
print('四、弱维能不能被父维救（r(子,父) 与 r(子,兄弟均值)）')
print('=' * 78)
weak_rows = []
for name, par, ch in FAM:
    for c in ch:
        yc = Y[:, ix[c]]
        sibs = [Y[:, ix[x]] for x in ch if x != c]
        rc = corr(yc, Y[:, ix[par]])
        rs = corr(yc, np.mean(sibs, 0))
        weak_rows.append(dict(dim=c, family=name, parent=par,
                              r_with_parent=rc, r_with_siblings=rs, is_weak=c in WEAK))
        flag = ' <== 弱维' if c in WEAK else ''
        print(f'{c:18s} r(子,父)={rc:6.3f}  r(子,兄弟均值)={rs:6.3f}  子维自身R2={-1:.0f}{flag}'.replace('子维自身R2=-1', ''))

wr = [r for r in weak_rows if r['is_weak']]
print()
print(f'12 个弱维里属于某个家族子维的有 {len(wr)} 个：')
for r in wr:
    print(f"  {r['dim']:18s} 父={r['parent']:14s} r(子,父)={r['r_with_parent']:.3f}")

print()
print('=' * 78)
print('判定')
print('=' * 78)
tree_min = min(r['r_mean'] for r in fam_rows)
tree_ok = tree_min >= 0.80
hyp_ok = any(r['ratio'] >= 1.30 for r in mds_rows)
best_ratio = max(r['ratio'] for r in mds_rows)
print(f'树成立判据：六个家族 min r(父,子均值) = {tree_min:.4f} -> {"成立" if tree_ok else "不成立"}（阈值 0.80）')
print(f'双曲更优判据：max(欧氏应力/双曲应力) = {best_ratio:.3f} -> {"双曲更准" if hyp_ok else "双曲不占优"}（阈值 1.30）')
weak_help = np.mean([r['r_with_parent'] for r in wr]) if wr else 0.0
print(f'弱维可救性：{len(wr)} 个弱维子维的 r(子,父) 均值 = {weak_help:.3f}（>0.5 才有信息可借）')

out = dict(families=fam_rows,
           sibling=dict(mean=float(sib.mean()), median=float(np.median(sib)), n=len(sib)),
           cross=dict(mean=float(cross.mean()), median=float(np.median(cross)), n=len(cross)),
           ratio_sib_cross=ratio,
           mds=mds_rows,
           weak_dims=weak_rows,
           verdict=dict(tree_min=tree_min, tree_ok=bool(tree_ok),
                        best_stress_ratio=best_ratio, hyp_ok=bool(hyp_ok),
                        weak_parent_corr_mean=float(weak_help), n_weak_in_tree=len(wr)))
p = os.path.join(C.OUT, '10_dimtree_check.json')
json.dump(out, open(p, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print()
print('已写', p)

import os
import sys
import json
import importlib
import numpy as np

# 直接测「父子收缩」到底能不能救弱维：用已有的折外预测做，不重训任何模型
# 对每个家族子 C，把 C 的折外预测往「父维预测的线性重标定」拉一点，看在没见过的折上 R2 是否真的上升
HERE = os.path.dirname(os.path.abspath(__file__))
Q1 = os.path.join(os.path.dirname(HERE), 'q1')
sys.path.insert(0, Q1)
import common as C

sens = C.load_sensory()
Y = sens[C.TARGET_COLS].to_numpy(float)
cols = list(C.TARGET_COLS)
ix = {c: i for i, c in enumerate(cols)}
d = np.load(os.path.join(C.OUT, '05_oof.npz'), allow_pickle=True)
P = d['E0_pred']

FAM = [
    ('A_malt', 'A_malt_all', ['A_malt_grain', 'A_malt_bread', 'A_malt_cara', 'A_malt_burn']),
    ('A_hops', 'A_hops_all', ['A_hops_citrus', 'A_hops_tropical', 'A_hops_noble', 'A_hops_woody']),
    ('A_esters', 'A_esters_all', ['A_esters_ethac', 'A_esters_isoaa', 'A_esters_flower', 'A_esters_fruity']),
    ('F_malt', 'F_malt_all', ['F_malt_grain', 'F_malt_bread', 'F_malt_cara', 'F_malt_burn']),
    ('F_hops', 'F_hops_all', ['F_hops_citrus', 'F_hops_tropical', 'F_hops_noble', 'F_hops_woody']),
    ('F_esters', 'F_esters_all', ['F_esters_ethac', 'F_esters_isoaa', 'F_esters_flower', 'F_esters_fruity']),
]
WEAK = ['clove', 'F_esters_flower', 'A_esters_flower', 'orange', 'stale_hops', 'A_esters_fruity',
        'A_malt_grain', 'barnyard', 'dms', 'diacetyl', 't2n', 'F_malt_grain']


def r2(y, p):
    ss = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - float(np.sum((y - p) ** 2)) / max(ss, 1e-12)


# 交叉拟合：a,b 与 lambda 都只在训练折上定，测试折只用一次
def shrink_child(parent, child, lam_grid, n_splits=5):
    yp = P[:, ix[parent]]
    yc = Y[:, ix[child]]
    yh = P[:, ix[child]]
    n = len(yc)
    fold = np.arange(n) % n_splits
    out = yh.copy()
    lams = []
    for f in range(n_splits):
        tr = fold != f
        te = ~tr
        Xtr = np.column_stack([np.ones(tr.sum()), yp[tr]])
        coef, *_ = np.linalg.lstsq(Xtr, yc[tr], rcond=None)
        cal = np.column_stack([np.ones(te.sum()), yp[te]]) @ coef
        # 在训练折上选 lambda：用同样的拟合再对训练折内部预测
        cal_tr = Xtr @ coef
        best, best_r2 = 0.0, -9.9
        for lam in lam_grid:
            pr = (1.0 - lam) * yh[tr] + lam * cal_tr
            v = r2(yc[tr], pr)
            if v > best_r2:
                best_r2, best = v, lam
        lams.append(best)
        out[te] = (1.0 - best) * yh[te] + best * cal
    return r2(yc, yh), r2(yc, out), out, lams


GRID = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
print('=' * 84)
print('父子收缩：子维预测 <- 父维预测（线性重标定后按 lambda 混合）')
print('=' * 84)
print(f"{'子维度':18s} {'原R2':>8s} {'收缩后':>8s} {'变化':>8s} {'lambda均':>9s}  备注")
rows = []
for name, par, ch in FAM:
    for c in ch:
        r0, r1, out, lams = shrink_child(par, c, GRID)
        rows.append(dict(family=name, parent=par, child=c, r2_before=r0, r2_after=r1,
                         delta=r1 - r0, lam_mean=float(np.mean(lams)), is_weak=c in WEAK))
        tag = '弱维' if c in WEAK else ''
        print(f'{c:18s} {r0:8.4f} {r1:8.4f} {r1 - r0:+8.4f} {np.mean(lams):9.3f}  {tag}')

# 把收缩后的预测拼回完整 250x50，看宏 R2 的变化
P2 = P.copy()
for r in rows:
    if r['delta'] > 0.002:
        _, _, out, _ = shrink_child(r['parent'], r['child'], GRID)
        P2[:, ix[r['child']]] = out


def macro_r2(Yt, Yp):
    ss = ((Yt - Yt.mean(0)) ** 2).sum(0)
    sr = ((Yt - Yp) ** 2).sum(0)
    return np.where(ss < 1e-12, 0.0, 1.0 - sr / ss)


m0 = macro_r2(Y, P)
m1 = macro_r2(Y, P2)
print()
print('=' * 84)
print('宏 R2')
print('=' * 84)
print(f'基线 E0           : {m0.mean():.4f}')
print(f'只对 delta>0.002 的 {int((np.array([r["delta"] for r in rows]) > 0.002).sum())} 个子维做收缩 : {m1.mean():.4f}   (变化 {m1.mean() - m0.mean():+.4f})')
print()
print('弱维明细：')
for c in WEAK:
    if c in [r['child'] for r in rows]:
        r = [x for x in rows if x['child'] == c][0]
        print(f"  {c:18s} {r['r2_before']:.4f} -> {r['r2_after']:.4f}  ({r['delta']:+.4f})")
    else:
        print(f'  {c:18s} 不在任何家族里，父子收缩无法覆盖')

p = os.path.join(C.OUT, '10_dimtree_shrink.json')
json.dump(dict(rows=rows, macro_before=float(m0.mean()), macro_after=float(m1.mean()),
               applied=int((np.array([r['delta'] for r in rows]) > 0.002).sum())),
          open(p, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print()
print('已写', p)

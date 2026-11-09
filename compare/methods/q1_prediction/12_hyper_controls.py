"""C3: curvature/dimension/subblock/noise controls, fold-local geometry cache."""
from protocol import *
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from scipy.spatial.distance import cdist


def geometry(tr, te, f, c=1., dim=4, euclidean=False,corrected=False):
    path = OUT / f'12_geom_f{f}_c{c:g}_d{dim}_e{int(euclidean)}_fix{int(corrected)}.npz'
    if path.exists():
        d = np.load(path); return d['a'], d['b'], d['names'].tolist()
    keep, _ = FE.prune_cols(CHEM, rows=tr)
    x = CHEM[keep].to_numpy(float)
    a, b, _ = FE.sentinel_fill(x[tr], x[te])
    if corrected:
        from geometry_corrected import H as fixed
        dist=fixed.knn_graph_distance(a,k=10)
        zz,_=fixed.fit_poincare(dist,dim=dim,c=c,iters=500,lr=.02)
        rr=fixed.fit_tangent_ridge(a,zz,c=c,alpha=1.)
        zte=fixed.transform_tangent_ridge(rr,b,c=c)
        feats,_=fixed.hyperbolic_features(zz,zte,STYLE[tr],STYLE[te],c=c)
        names=list(feats);matrix=np.column_stack([feats[n] for n in names])
        a,b=matrix[:len(tr)],matrix[len(tr):]
    elif not euclidean:
        a, b, names, _, _ = FE.hyperbolic_block(a, b, STYLE[tr], STYLE[te], c=c, dim=dim)
    else:
        dist = H.knn_graph_distance(a, k=10)
        j = np.eye(len(a)) - 1 / len(a)
        vals, vecs = np.linalg.eigh(-.5 * j @ dist ** 2 @ j)
        z = vecs[:, -dim:] * np.sqrt(np.maximum(vals[-dim:], 0))
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        zte = make_pipeline(StandardScaler(), Ridge(alpha=1)).fit(a, z).predict(b)
        cats = sorted(set(STYLE[tr])); zz = np.vstack([z, zte])
        cents = np.vstack([z[STYLE[tr] == s].mean(0) for s in cats])
        dc = cdist(zz, cents)
        own = np.array([cats.index(s) if s in cats else 0 for s in np.r_[STYLE[tr], STYLE[te]]])
        od = dc[np.arange(len(zz)), own]
        mu = np.array([od[:len(tr)][STYLE[tr] == s].mean() for s in cats])
        sd = np.array([max(od[:len(tr)][STYLE[tr] == s].std(), 1e-9) for s in cats])
        other = dc.copy(); other[np.arange(len(zz)), own] = np.inf
        den = cdist(zz, z); den[np.arange(len(tr)), np.arange(len(tr))] = np.inf
        a = np.column_stack([dc, od, (od-mu[own])/sd[own], od/np.maximum(other.min(1),1e-9),
                             np.sort(den, axis=1)[:, :5].mean(1), np.linalg.norm(zz, axis=1), zz])
        a, b = a[:len(tr)], a[len(tr):]
        names = ['hyp_dist_' + s for s in cats] + ['hyp_typ_dist','hyp_typ_z','hyp_typ_ratio',
                  'hyp_dens5','hyp_radius'] + [f'hyp_tan{i+1}' for i in range(dim)]
    np.savez_compressed(path, a=a, b=b, names=np.array(names), fold_id=FID)
    return a, b, names


def main():
    e1 = original('E1')
    rows = [metrics('E1', e1, e1)]
    configs = [('euclidean', 1., 4, 'all', -1)]
    configs += [(f'c{c:g}', c, 4, 'all', -1) for c in [1e-8, .1, 1., 10.]]
    configs += [(f'corrected_c{c:g}', c, 4, 'all', -1) for c in [1e-8, .1, 1., 10.]]
    configs += [(f'd{d}_{part}', 1., d, part, -1) for d in [2, 4, 8] for part in ['dist','typ','tan']]
    configs += [(f'perm{i}', 1., 4, 'all', i) for i in range(20)]
    preds = {}
    for name, c, dim, part, perm in configs:
        def fitter(tr, te, f):
            a, b, base_names = design(tr, te)
            ga, gb, names = geometry(tr, te, f, c, dim, name == 'euclidean', name.startswith('corrected'))
            if part != 'all':
                ix = [i for i, n in enumerate(names) if n.startswith('hyp_' + part)]
                ga, gb = ga[:, ix], gb[:, ix]
            if perm >= 0:
                rng = np.random.default_rng(C.SEED + 1000 * perm + f)
                # Independent column permutations, separately within training and test partitions.
                ga = np.column_stack([rng.permutation(v) for v in ga.T])
                gb = np.column_stack([rng.permutation(v) for v in gb.T])
            cut=next((i for i,n in enumerate(base_names) if n.startswith('style__')),a.shape[1])
            aa,bb=np.c_[a[:,:cut],ga,a[:,cut:]],np.c_[b[:,:cut],gb,b[:,cut:]]
            return ExtraTreesRegressor(n_estimators=C.trees(300), max_features=.5,
                random_state=C.SEED + f, n_jobs=1).fit(aa, Y[tr]).predict(bb)
        p = cv('12_' + name, fitter=fitter); preds[name] = p
        rows.append(metrics(name, p, e1))
        report(OUT / '12_hyper_controls.md', 'C3 双曲严谨对照（增量）', {'metrics': pd.DataFrame(rows)},
               '全部几何拟合仅用训练折；子块按实际字段选择。实际dim4完整块为31列，设计文档46列是误记。'
               'HC4每列独立打乱且训练/验证分开。待全部配置完成后汇总条件性结论。')
    e2 = preds['c1']
    bucket = []
    isolation = np.zeros(len(Y))
    for f, (_, te) in enumerate(FOLDS):
        isolation[te] = BASE[f'knn_te_d_{f}'].mean(1) if NF == 10 else 0
    grouping=[('style', STYLE), ('mean_z', np.where(Y.mean(1) >= np.median(Y.mean(1)), 'high','low')),
              ('isolation', pd.qcut(isolation, 3, labels=False, duplicates='drop'))]
    for mode in ['unit','sd','minmax']:
        mu,sg=C.calib_params(Y,mode)
        grade=C.score_to_grade(C.z_to_score(Y,mu,sg)).mean(1)
        grouping.append(('mean_grade_'+mode,np.where(grade>np.median(grade),'high','low')))
    for field, labels in grouping:
        for label in np.unique(labels):
            ix = labels == label
            bucket.append(dict(bucket=field, label=label, n=int(ix.sum()),
                mse_E1=float(np.mean((Y[ix]-e1[ix])**2)), mse_E2=float(np.mean((Y[ix]-e2[ix])**2))))
    r = pd.DataFrame(rows)
    from common_stats import permutation_p
    observed = C.macro_r2(Y, e2).mean()
    null = [C.macro_r2(Y, preds[f'perm{i}']).mean() for i in range(20)]
    p = permutation_p(observed, null)
    delta = observed - C.macro_r2(Y, e1).mean()
    conclusion = f'完整双曲块相对E1宏R²差={delta:.6f}；噪声块经验p={p:.6f}（20次，最小p=1/21）。'
    eu=C.macro_r2(Y,preds['euclidean']).mean();base=C.macro_r2(Y,e1).mean()
    conclusion+=f'HC1同维欧氏块宏R²={eu:.6f}，E1={base:.6f}，E2={observed:.6f}。'
    conclusion+='欧氏块也降分，损失并非双曲曲率独有。' if eu<base-.005 else '欧氏块未出现明显同幅降分，当前双曲表示/实现需要单独定位。'
    conclusion+='未支持双曲块信息高于随机噪声。' if p>=.05 else '双曲块优于20个噪声块的探索性比较，仍需同时看它相对E1的绝对损益。'
    conclusion += '原模块c≠1的log/exp不互逆且距离缺1/sqrt(c)，原扫描仅复现实现行为；corrected_c行修复公式后才是曲率一致对照。'
    conclusion += '平均预测有害。分桶结果仅事后探索，不能据此宣称可靠条件增益。' if delta < -.005 else '平均效果接近无害或有益，需配对检验共同判断。'
    conclusion+='HC5等级分桶按unit/sd/minmax三口径各自样本均值中位数划分；分组仅作事后误差分析，不参与拟合。'
    report(OUT / '12_hyper_controls.md', 'C3 双曲严谨对照', {'metrics':r,'buckets':pd.DataFrame(bucket)}, conclusion)


if __name__ == '__main__': main()

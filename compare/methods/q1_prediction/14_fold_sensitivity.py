from protocol import *
from sklearn.model_selection import StratifiedKFold, KFold


def main():
    rows = []
    schemes = dict(current=FOLDS,
        style=list(StratifiedKFold(NF, shuffle=True, random_state=C.SEED).split(Y, STYLE)),
        random=list(KFold(NF, shuffle=True, random_state=C.SEED).split(Y)))
    for scheme, folds in schemes.items():
        fid = np.full(len(Y), -1)
        for f, (_, te) in enumerate(folds): fid[te] = f
        for cfg in ['E1', 'BASE5']:
            p = original(cfg) if scheme == 'current' else cv(f'14_{scheme}_{cfg}', cfg, folds)
            rows.append(dict(scheme=scheme, **metrics(cfg, p, original(cfg),fid=fid),
                             same_as_current=bool(np.array_equal(fid, FID))))
    d = pd.DataFrame(rows)
    deltas={cfg:float(d[(d.scheme=='style')&(d.model==cfg)].macro_r2.iloc[0]-d[(d.scheme=='current')&(d.model==cfg)].macro_r2.iloc[0]) for cfg in ['E1','BASE5']}
    report(OUT / '14_fold_sensitivity.md', 'C5 折划分敏感性', {'metrics': d},
           '原split_folds在最小分层样本数不足时整体退回KFold；因此current与random本应完全相同。'
           'style使用StratifiedKFold（允许稀有类别警告）以真正测试仅风格分层。'
           '历史BASE5缓存与当前相同折/种子重跑有小幅差异，不能把它归因于折划分；18复现核查将其定位于M3目标PCA分支，其余成员及E1可精确复现。历史PCA状态未保存，无法唯一追溯更深原因。'
           f'仅风格分层相对历史当前折：E1差={deltas["E1"]:.6f}，BASE5差={deltas["BASE5"]:.6f}；均小于0.005。'
           '这只支持本次划分差异有限，不能证明任意划分无影响。未触发按设计阈值要求换折重跑主线。')


if __name__ == '__main__': main()

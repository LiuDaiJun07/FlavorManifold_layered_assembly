"""C2: repeated paired CV, fixed-fold seed family, Friedman/Nemenyi."""
from protocol import *
from common_stats import paired, friedman_nemenyi, wilcoxon
from sklearn.model_selection import StratifiedKFold, KFold
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    rows, dims, repeat_rows = [], [], []
    strata = np.char.add(STYLE, np.where(G.mean(1) >= 3, '_h', '_l'))
    for rep in range(3):
        # StratifiedKFold allows rare strata (<5); retain exact key and disclose warning.
        folds = list(StratifiedKFold(5, shuffle=True, random_state=C.SEED + rep).split(Y, strata))
        pp = {k: cv(f'11_rep{rep}_{k}', cfg=k, folds=folds, seed=rep * 1000)
              for k in ['E1', 'BASE5']}
        for f, (_, te) in enumerate(folds):
            rows.append(dict(repeat=rep, fold=f, **{k: C.macro_r2(Y[te], p[te]).mean() for k, p in pp.items()}))
        dims.append(C.macro_r2(Y, pp['BASE5']) - C.macro_r2(Y, pp['E1']))
        repeat_rows.append(dict(repeat=rep,**{k:float(C.macro_r2(Y,p).mean()) for k,p in pp.items()}))
    df = pd.DataFrame(rows)
    seeds = []
    for seed in range(3):
        existing=OUT/'14_random_BASE5.npz'
        target=OUT/'11_seed0.npz'
        if seed==0 and existing.exists() and not target.exists():
            cache=np.load(existing)
            if np.array_equal(cache['fold_id'],FID):
                np.savez_compressed(target,**{k:cache[k] for k in cache.files},reused_from=str(existing.name))
        p = cv(f'11_seed{seed}', cfg='BASE5', seed=seed)
        seeds.append(metrics(str(seed), p))
    stat = paired(df.BASE5, df.E1)
    stat['p_dim_mean_repeat'] = wilcoxon(np.mean(dims, axis=0), np.zeros(50))
    stat['seed_range'] = float(pd.DataFrame(seeds).macro_r2.max() - pd.DataFrame(seeds).macro_r2.min())
    stat['E1_mean'], stat['E1_std'] = df.E1.mean(), df.E1.std()
    stat['BASE5_mean'], stat['BASE5_std'] = df.BASE5.mean(), df.BASE5.std()
    repeats=pd.DataFrame(repeat_rows)
    for k in ['E1','BASE5']:
        stat[k+'_pooled_repeat_mean']=repeats[k].mean()
        stat[k+'_pooled_repeat_std']=repeats[k].std()
    frames = {'folds': df, 'statistics': pd.DataFrame([stat]), 'seeds': pd.DataFrame(seeds),'pooled_repeats':repeats}
    names = ['B0', 'B1', 'B2_med', 'B3_med', 'B4', 'B5_15', 'B6', 'B7']
    if all((OUT / f'10_{n}.npz').exists() for n in names):
        names += ['E1', 'BASE5']
        preds = [np.load(OUT / f'10_{n}.npz')['pred'] if n.startswith('B') and n != 'BASE5'
                 else original(n) for n in names]
        scores = np.column_stack([C.macro_r2(Y, p) for p in preds])
        st = friedman_nemenyi(scores)
        frames['nemenyi'] = pd.DataFrame(st['p'], columns=names).assign(model=names)
        frames['friedman'] = pd.DataFrame([dict(p=st['friedman_p'], critical_difference=st['cd'])])
        order = np.argsort(st['ranks'])
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.scatter(st['ranks'][order], np.arange(len(names)))
        ax.set_yticks(np.arange(len(names)), np.array(names)[order])
        ax.set_xlabel('Mean rank across 50 target dimensions (dependent blocks)')
        ax.plot([1, 1 + st['cd']], [len(names), len(names)], color='black')
        ax.text(1, len(names) + .2, f"CD={st['cd']:.2f}")
        fig.tight_layout(); fig.savefig(OUT / 'fig11_cd.png', dpi=180); plt.close(fig)
    report(OUT / '11_repeated_cv.md', 'C2 重复CV与统计检验', frames,
           f'BASE5−E1折级差={stat["delta_mean"]:.6f}±{stat["delta_std"]:.6f}，配对Wilcoxon p={stat["wilcoxon_p"]:.6g}；种子极差={stat["seed_range"]:.6f}。稀有分层无法保证每折包含全部风格。'
           '固定折 seed_off=f+{0,1,2}。CV折并非独立重复，检验仅探索性；未以p>0.05证明等效。'
           '同时报告每次重复全250行OOF拼接后的宏R²均值/标准差（3次），与15个验证折的分数分布分开，二者因验证集方差分母不同不必相等。'
           'Friedman使用同折OOF的50个目标作区组，其相关性限制推断。')


if __name__ == '__main__': main()

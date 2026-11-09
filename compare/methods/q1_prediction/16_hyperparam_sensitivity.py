from protocol import *
import itertools
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    rows=[];ref=original('E1')
    # Design says 12 but enumerates 4×3×3=36: run full Cartesian grid.
    for mf,leaf,n in itertools.product([.3,.5,.7,1.],[1,2,5],[100,300,800]):
        name=f'16_mf{mf}_leaf{leaf}_n{n}'
        p=cv(name,max_features=mf,min_samples_leaf=leaf,n_estimators=C.trees(n))
        rows.append(dict(max_features=mf,min_leaf=leaf,trees=C.trees(n),**metrics(name,p,ref)))
    df=pd.DataFrame(rows);fig,axes=plt.subplots(1,3,figsize=(12,4),constrained_layout=True)
    for ax,n in zip(axes,sorted(df.trees.unique())):
        z=df[df.trees==n].pivot(index='min_leaf',columns='max_features',values='macro_r2')
        im=ax.imshow(z,aspect='auto',vmin=df.macro_r2.min(),vmax=df.macro_r2.max())
        ax.set_xticks(range(4),z.columns);ax.set_yticks(range(3),z.index);ax.set_title(f'{n} trees');ax.set_xlabel('max_features')
        for i in range(3):
            for j in range(4):ax.text(j,i,f'{z.iloc[i,j]:.3f}',ha='center',va='center',fontsize=8,
                color='black' if z.iloc[i,j]>(df.macro_r2.min()+df.macro_r2.max())/2 else 'white')
    axes[0].set_ylabel('min_samples_leaf')
    fig.colorbar(im,ax=axes.tolist(),label='OOF macro R²');fig.savefig(OUT/'fig16_hyperparams.png',dpi=180);plt.close(fig)
    best=df.loc[df.macro_r2.idxmax()];near=int((df.macro_r2>=best.macro_r2-.005).sum())
    report(OUT/'16_hyperparam_sensitivity.md','C7 超参敏感性',{'metrics':df},
           f'宏R²范围={df.macro_r2.min():.6f}–{df.macro_r2.max():.6f}；距最优0.005内有{near}/{len(df)}组（描述性平台阈值）。'
           f'最高点max_features={best.max_features}, min_leaf={best.min_leaf}, trees={best.trees}，不将该外层扫描最高点当作无偏新模型性能。'
           '设计所列网格实际36组而非12组，执行完整笛卡尔积。外层OOF扫描用于敏感性描述，不能当作无偏最优模型评估。')


if __name__=='__main__':main()

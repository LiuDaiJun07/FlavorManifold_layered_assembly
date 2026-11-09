from shared import *
import itertools


def main():
    rows=[]
    common=set.intersection(*[set(FE.prune_cols(CHEM,rows=tr)[0]) for tr,_ in FOLDS])
    columns=[c for c in C.chem_feature_cols(CHEM) if c in common]
    full=CHEM[columns].fillna(CHEM[columns].median()).to_numpy(float)
    mu,sd=full.mean(0),np.maximum(full.std(0),1e-9);real=(full-mu)/sd
    for q,kappa in itertools.product([40,77,120,144],[1,5,10]):
        stats=[];arrays={};pooled=np.zeros_like(real)
        for f,(tr,te) in enumerate(FOLDS):
            gen=Generator(tr,q=q,kappa=kappa);a=gen.real(te);b=np.zeros_like(a)
            rng=np.random.default_rng(C.SEED+f)
            for c in np.unique(STYLE[te]):
                ix=np.where(STYLE[te]==c)[0]
                b[ix]=gen.sample('MAIN',c,100,rng)[:len(ix)]
            idx=[gen.keep.index(c) for c in columns]
            pooled[te]=((b*gen.sd+gen.mu)[:,idx]-mu)/sd
            stats.append(quality(a,b,permutations=0))
            arrays[f'real_f{f}']=a;arrays[f'generated_f{f}']=b
        np.savez_compressed(QOUT/f'09_q{q}_k{kappa}.npz',**arrays,fold_id=FID,pred=pooled,real=real,r2_dim=np.full(50,np.nan))
        pool_stats=quality(real,pooled,permutations=0)
        rows.append(dict(q=q,kappa=kappa,**{'fold_'+key:np.mean([x[key] for x in stats]) for key in ['ks_dim_mean','mmd2','energy']},
                         **{'pooled_'+key:pool_stats[key] for key in ['ks_dim_mean','ks_pooled','mmd2','energy']}))
    data=pd.DataFrame(rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,metric,title in zip(axes,['pooled_ks_dim_mean','pooled_mmd2'],['Pooled mean KS','Pooled MMD²']):
        z=data.pivot(index='q',columns='kappa',values=metric);im=ax.imshow(z,aspect='auto')
        ax.set_xticks(range(3),z.columns);ax.set_yticks(range(4),z.index);ax.set(xlabel='kappa',ylabel='q',title=title)
        for i in range(4):
            for j in range(3):ax.text(j,i,f'{z.iloc[i,j]:.4f}',ha='center',va='center',color='white',fontsize=8)
        fig.colorbar(im,ax=ax)
    fig.tight_layout();fig.savefig(QOUT/'fig09_latent_sensitivity.png',dpi=180);plt.close(fig)
    report(QOUT/'09_latent_sensitivity.md','C21 潜维数与收缩敏感性',{'grid':data},
           f'池化逐维KS均值范围={data.pooled_ks_dim_mean.min():.6f}–{data.pooled_ks_dim_mean.max():.6f}，MMD²范围={data.pooled_mmd2.min():.6f}–{data.pooled_mmd2.max():.6f}。'
           '12组同折、同种子族；q在训练样本可行范围内截断。折均值与全250行池化指标分开列出；池化坐标只用于评价，不进入生成拟合。'
           '敏感性扫描不能充作选择最优q/κ后的无偏验证。')


if __name__=='__main__':main()

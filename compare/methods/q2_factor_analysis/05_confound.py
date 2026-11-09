from shared import *
from scipy.stats import rankdata
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    raw=spearman(X,Y)
    a,b=X.copy(),Y.copy()
    for s in np.unique(STYLE):
        ix=STYLE==s;a[ix]-=a[ix].mean(0);b[ix]-=b[ix].mean(0)
    k1=spearman(a,b)
    cov=['ethanol..v.v.','color.EBC.','bitterness','pH']
    v=np.c_[np.ones(len(Y)),rankdata(CHEM[cov].fillna(CHEM[cov].median()).to_numpy(),axis=0)]
    rx,ry=rankdata(X,axis=0),rankdata(Y,axis=0)
    k2=corr(rx-v@np.linalg.lstsq(v,rx,rcond=None)[0],ry-v@np.linalg.lstsq(v,ry,rcond=None)[0])
    zs=[];weights=[];styles=[];valid=[]
    for s in np.unique(STYLE):
        ix=STYLE==s
        if ix.sum()<8:continue
        zs.append(np.arctanh(np.clip(spearman(X[ix],Y[ix]),-.999999,.999999)))
        weights.append(ix.sum()-3);styles.append(s)
        valid.append((X[ix].std(0)>1e-12)[:,None]&(Y[ix].std(0)>1e-12)[None,:])
    zs=np.array(zs);valid=np.asarray(valid);w=np.array(weights)[:,None,None]*valid
    sw=w.sum(0);effective=valid.sum(0)
    fixed=(w*zs).sum(0)/np.maximum(sw,1e-12);q=(w*(zs-fixed)**2).sum(0)
    tau=np.maximum(0,(q-np.maximum(effective-1,0))/np.maximum(sw-(w*w).sum(0)/np.maximum(sw,1e-12),1e-12))
    rw=np.where(valid,1/(1/np.maximum(w,1e-12)+tau),0)
    k3=np.tanh((rw*zs).sum(0)/np.maximum(rw.sum(0),1e-12))
    i2=np.where(effective>=2,np.maximum(0,(q-(effective-1))/np.maximum(q,1e-12)),np.nan)
    original=top(raw);seat=[];summary=[]
    for name,m in [('K1_style_demean',k1),('K2_partial_spearman',k2),('K3_random_effects',k3)]:
        idx=top(m)
        for j,dim in enumerate(C.TARGET_COLS):
            summary.append(dict(method=name,dim=dim,jaccard=jaccard(original[:,j],idx[:,j])))
            for rank in range(3):
                old,new=original[rank,j],idx[rank,j]
                seat.append(dict(method=name,dim=dim,rank=rank+1,original=FEAT[old],adjusted=FEAT[new],
                    replaced=bool(old!=new),raw_rho=raw[new,j],adjusted_rho=m[new,j],I2=i2[new,j],valid_styles=int(effective[new,j])))
    d=pd.DataFrame(summary);t=pd.DataFrame(seat)
    fig,ax=plt.subplots(figsize=(6,6));ix=original.ravel();jj=np.tile(np.arange(50),3)
    ax.scatter(raw[ix,jj],k1[ix,jj],s=12);ax.plot([-1,1],[-1,1],'k--')
    ax.set(xlabel='Raw rho (univariate top3)',ylabel='Within-style rho');fig.tight_layout()
    fig.savefig(QOUT/'fig05_confound.png',dpi=180);plt.close(fig)
    np.savez_compressed(QOUT/'05_confound.npz',raw=raw,k1=k1,k2=k2,k3=k3,I2=i2,fold_id=FID,
                        feat=np.array(FEAT),r2_dim=np.full(50,np.nan),pred=np.empty((0,50)))
    means=d.groupby('method',as_index=False).jaccard.mean()
    report(QOUT/'05_confound.md','C9 混杂控制',{'summary':means,'seats':t,'dimensions':d},
           '原始对照采用同一候选池的单变量Top3，避免把综合排序变化混入去混杂效应。'
           'K1为原值风格内去均值后Spearman；K2为秩残差偏相关；K3为样本≥8风格的Fisher-z随机效应合并（Spearman方差近似n−3）。常数变量组不计入权重，少于2个有效风格的I²不定义。'
           f'K1平均Jaccard={means.iloc[0].jaccard:.4f}；席位变化不能被解释为因果关系。'
           'K2协变量自身的残差方差为0，不参与化合物排名；近常数列的相关系数记0。')


if __name__=='__main__':main()

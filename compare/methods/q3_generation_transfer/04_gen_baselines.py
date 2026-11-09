from shared import *


def main():
    # Global holdout excludes all held-out observations from imputation, scale, PCA and calibration.
    # Common evaluation coordinates fit once to each training partition, then map raw outputs
    # to a fixed scale for pooling. Fixed scale only defines evaluation units, never fitting.
    keeps=[set(FE.prune_cols(CHEM,rows=tr)[0]) for tr,_ in FOLDS]
    common=set.intersection(*keeps)
    cols=[c for c in C.chem_feature_cols(CHEM) if c in common]
    raw=CHEM[cols].to_numpy(float)
    med=np.nanmedian(raw,0);full=np.where(np.isnan(raw),med,raw)
    mu,sd=full.mean(0),np.maximum(full.std(0),1e-9)
    pp={k:np.zeros_like(full) for k in ['MAIN','G0','G1','G2','G3']}
    def task(item):
        f,(tr,te)=item;g=Generator(tr);out={k:np.zeros((len(te),len(cols))) for k in pp}
        for k in pp:
            rng=np.random.default_rng(C.SEED+f)
            for c in np.unique(STYLE[te]):
                ix=np.where(STYLE[te]==c)[0]
                # Generate reference batch 100/style so empirical calibration is meaningful.
                zz=g.sample(k,c,max(100,len(ix)),rng)[:len(ix)]
                idx=[g.keep.index(c) for c in cols]
                arr=(zz*g.sd+g.mu)[:,idx]
                out[k][ix]=(arr-mu)/sd
        return te,out
    for te,out in parallel(task,list(enumerate(FOLDS))):
        for k in pp:pp[k][te]=out[k]
    real=(full-mu)/sd;rows=[];perstyle=[]
    for k,p in pp.items():
        rows.append(dict(method=k,**quality(real,p,classifier=True)))
        for c in sorted(set(STYLE)):
            ix=STYLE==c;perstyle.append(dict(method=k,style=c,n=int(ix.sum()),**quality(real[ix],p[ix])))
        np.savez_compressed(QOUT/f'04_{k}.npz',pred=p,real=real,fold_id=FID,r2_dim=np.full(50,np.nan))
    # Explicit LOO for styles n<10: each real point compared to training-only reference batch.
    loo=[]
    for i in range(len(Y)):
        if np.sum(STYLE==STYLE[i])>=10:continue
        g=Generator(np.delete(np.arange(len(Y)),i))
        for k in pp:
            generated=g.sample(k,STYLE[i],30,np.random.default_rng(C.SEED+i))
            loo.append(dict(method=k,style=STYLE[i],beer=int(CHEM.beer.iloc[i]),
                            **mmd_energy(g.real([i]),generated,500,C.SEED+i)))
    df=pd.DataFrame(perstyle);weighted=[]
    for k in pp:
        d=df[df.method==k]
        weighted.append(dict(method=k,**{metric:np.average(d[metric],weights=d.n) for metric in
            ['mmd2','energy','ks_dim_mean']},equal_style_mmd=float(d.mmd2.mean())))
    mainrow=next(r for r in rows if r['method']=='MAIN');bootrow=next(r for r in rows if r['method']=='G2')
    report(QOUT/'04_gen_baselines.md','C15/C16 生成基线与小样本协议',
           {'metrics':pd.DataFrame(rows),'per_style':df,'stratified':pd.DataFrame(weighted),'small_style_LOO':pd.DataFrame(loo)},
           f'全部方法同一折与训练支持域；评估仅限各训练折共同保留的{len(cols)}列，避免用未建模列常数回填人为抬高判别准确率。主模型在折内重估PCA/收缩/经验校准；G0全局残差对角方差，G1 Ledoit-Wolf，G2风格内残差bootstrap，G3全局单因子。'
           '池化评价坐标使用全体特征定义量纲，仅用于评价；未进入生成拟合。'
           '每个小风格样本另做LOO，单点MMD/能量是相似性描述，不能可靠估计其分布。'
           f"主模型MMD²={mainrow['mmd2']:.6g}，残差bootstrap={bootrow['mmd2']:.6g}；主模型未在MMD上胜出，但后者最近邻距离比={bootrow['nearest_ratio']:.6g}，暴露样本复用。"
           'permutation=500，CV模型共享训练样本使可交换性仅近似；所有p值为探索性，不能证明分布相同。'
           '分类器含折内PCA，平衡真实/生成样本数，相同谱按数值8位舍入分组、5折StratifiedGroupKFold，避免复制谱跨真假分类训练/验证；报告balanced accuracy。'
           '各风格MMD、能量、KS和最近邻比均附500次行标签置换p值；最近邻比为|log(ratio)|双侧偏离。训练生成器与判别器划分仍非完全独立外部验证。')


if __name__=='__main__':main()

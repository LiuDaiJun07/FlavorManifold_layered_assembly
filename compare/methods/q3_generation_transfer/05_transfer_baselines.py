from shared import *
from sklearn.linear_model import Ridge
from scipy.special import logsumexp


def sinkhorn_map(a,b,epsilon=.5,iterations=2000):
    cost=cdist(a,b)**2;cost/=max(np.median(cost),1e-12)
    logk=-cost/epsilon;u=np.zeros(len(a));v=np.zeros(len(b))
    for _ in range(iterations):
        u=-np.log(len(a))-logsumexp(logk+v[None,:],axis=1)
        v=-np.log(len(b))-logsumexp(logk+u[:,None],axis=0)
    plan=np.exp(logk+u[:,None]+v[None,:]);mapped=plan@b/plan.sum(1,keepdims=True)
    return mapped,float(max(abs(plan.sum(0)-1/len(b)).max(),abs(plan.sum(1)-1/len(a)).max()))


def main():
    d,keep,x,z,cats,mus,lam,src,dst,res=transfer_data()
    # No paired source-target beers exist: identifiable marginal location-scale affine baseline.
    source,target=z[src],z[dst]
    affine=(source-source.mean(0))*target.std(0)/np.maximum(source.std(0),1e-6)+target.mean(0)
    ot,error=sinkhorn_map(source,target)
    zs=dict(T0=np.tile(target.mean(0),(len(src),1)),T1=affine,T2=ot,
            T3=mus[cats.index('Tripel')]+res)
    reader=LegacyReadout()
    mu,sg=C.load_scale();real_s=C.z_to_score(Y[dst],mu,sg).mean(0)
    rows=[];sample_rows=[];nondef=[i for i,c in enumerate(keep) if c not in DEFECT]
    save=dict(source=x[src],target=x[dst],src_index=src,dst_index=dst,keep=np.array(keep),fold_id=FID,
              r2_dim=np.full(50,np.nan))
    save['before_z']=reader.predict_z(pd.DataFrame(x[src],columns=keep),'Pils/Lager')
    save['before_sensory']=C.z_to_score(save['before_z'],mu,sg)
    for k,v in zs.items():
        raw=v*d['sd_x']+d['mu_x'];guarded,n=guard(raw,x[src],keep)
        for state,a in [('raw',raw),('guarded',guarded)]:
            zp=reader.predict_z(pd.DataFrame(a,columns=keep),'Tripel')
            scores=C.z_to_score(zp,mu,sg)
            metrics_row=dict(method=k,state=state,rmse_portrait=float(np.sqrt(np.mean((scores.mean(0)-real_s)**2))),
                corr=float(np.corrcoef(scores.mean(0),real_s)[0,1]),guard_fixes=n if state=='guarded' else 0,
                ks_nondef=float(np.mean([ks_2samp(a[:,j],x[dst,j]).statistic for j in nondef])),
                defect_increases=sum(int((a[:,keep.index(c)]>x[src,keep.index(c)]+1e-9).sum()) for c in DEFECT if c in keep))
            az=(a-d['mu_x'])/d['sd_x']
            metrics_row['individual_variance_vs_source']=float(np.var(az[:,nondef],axis=0).sum()/max(np.var(source[:,nondef],axis=0).sum(),1e-12))
            metrics_row['variance_vs_target']=float(np.var(az[:,nondef],axis=0).sum()/max(np.var(target[:,nondef],axis=0).sum(),1e-12))
            metrics_row.update(mmd_energy((x[dst]-d['mu_x'])/d['sd_x'],(a-d['mu_x'])/d['sd_x'],500))
            for mode in ['unit','sd','minmax']:
                cm,cs=C.calib_params(Y,mode)
                truth=C.z_to_score(Y[dst],cm,cs).mean(0)
                image=C.z_to_score(zp,cm,cs).mean(0)
                metrics_row['rmse_'+mode]=float(np.sqrt(np.mean((image-truth)**2)))
                metrics_row['mae_'+mode]=float(np.mean(abs(image-truth)))
                metrics_row['corr_'+mode]=float(np.corrcoef(image,truth)[0,1])
            rows.append(metrics_row);save[f'{k}_{state}']=a;save[f'{k}_{state}_sensory']=scores
            save[f'{k}_{state}_z']=zp
            for ii,beer in enumerate(CHEM.beer.iloc[src]):
                sample_rows.append(dict(method=k,state=state,beer=beer,rmse=float(np.sqrt(np.mean((scores[ii]-real_s)**2)))))
    save['pred']=save['T3_guarded_sensory']
    np.savez_compressed(QOUT/'05_transfer_baselines.npz',**save)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots()
    samples=pd.DataFrame(sample_rows)
    for k in zs:
        dd=samples[(samples.method==k)&(samples.state=='guarded')]
        ax.scatter(range(len(dd)),dd.rmse,label=k)
    ax.set(xlabel='Source sample',ylabel='Sensory RMSE to target style (sd calibration)');ax.legend()
    fig.tight_layout();fig.savefig(QOUT/'fig05_transfer.png',dpi=180);plt.close(fig)
    completed=pd.DataFrame(rows);guarded_rows=completed[completed.state=='guarded']
    winner=guarded_rows.loc[guarded_rows.rmse_portrait.idxmin()]
    report(QOUT/'05_transfer_baselines.md','C17 迁移基线',{'metrics':completed,'samples':samples},
           f'T1原设计无法拟合：不存在250组源→目标配对谱。改用源/目标边缘均值方差定义的逐维仿射映射，已明确区分。Sinkhorn边缘误差={error:.3g}。'
           'T0–T3同一11源样本、同一原读出器、同一7列缺陷守护（不在支持集的列不改写）。'
           '主方法保留旧链路，不额外支持域截断。目标样本参与风格分布估计，属描述性迁移，非外部泛化检验。'
           f'守护后画像RMSE最小为{winner.method}（{winner.rmse_portrait:.6f}）；T3并非由高相关系数即可证明优于朴素映射。'
           'unit/sd/minmax的RMSE/MAE已并列。后续独立读出器使用各自自然空间，避免混同量纲。')


if __name__=='__main__':main()

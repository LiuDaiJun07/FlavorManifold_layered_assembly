from shared import *
from sklearn.manifold import MDS
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def stress(target,pred):
    ix=np.triu_indices(len(target),1);a,b=target[ix],pred[ix]
    scale=float(a@b/max(b@b,1e-12))
    return float(np.sqrt(np.sum((a-scale*b)**2)/np.sum(a*a)))


def main():
    d,keep,x,z,cats,mus,lam,src,dst,res=transfer_data()
    mu,sg=C.load_scale();scores=C.z_to_score(Y,mu,sg)
    cent=np.vstack([scores[STYLE==c].mean(0) for c in cats]);cent=(cent-cent.mean(0))/np.maximum(cent.std(0),1e-9)
    distance=np.maximum(1-np.corrcoef(cent),0);np.fill_diagonal(distance,0)
    hyp,_=H.fit_poincare(distance,dim=2,iters=800,c=1)
    eu=MDS(n_components=2,dissimilarity='precomputed',random_state=C.SEED,n_init=8,
           normalized_stress=True,max_iter=600).fit_transform(distance)
    dh=H.poincare_dist(hyp,hyp);de=cdist(eu,eu)
    hs,es=stress(distance,dh),stress(distance,de)
    calibrations=[]
    for mode in ['unit','sd','minmax']:
        cm,cs=C.calib_params(Y,mode);ss=C.z_to_score(Y,cm,cs)
        cc=np.vstack([ss[STYLE==c].mean(0) for c in cats]);cc=(cc-cc.mean(0))/np.maximum(cc.std(0),1e-9)
        dd=np.maximum(1-np.corrcoef(cc),0);np.fill_diagonal(dd,0)
        if np.allclose(dd,distance,atol=1e-12,rtol=0):mh,me=hs,es
        else:
            hz,_=H.fit_poincare(dd,dim=2,iters=800,c=1)
            ez=MDS(n_components=2,dissimilarity='precomputed',random_state=C.SEED,n_init=8,
                normalized_stress=True,max_iter=600).fit_transform(dd)
            mh,me=stress(dd,H.poincare_dist(hz,hz)),stress(dd,cdist(ez,ez))
        calibrations.append(dict(calibration=mode,hyperbolic=mh,euclidean=me,ratio=me/max(mh,1e-12),
                                 max_distance_change=float(abs(dd-distance).max())))
    # Use a common numeric bandwidth after matching distance scales (mean positive distance=1).
    hscale=dh[np.triu_indices(len(cats),1)].mean();escale=de[np.triu_indices(len(cats),1)].mean()
    dnorm=dh/hscale;np.fill_diagonal(dnorm,np.inf);bandwidth=float(np.median(dnorm.min(1)))
    i,j=cats.index('Pils/Lager'),cats.index('Tripel')
    rows=[];weights=[];mid={}
    for method in ['euclidean','main_origin_tangent','true_geodesic']:
        for t in [.25,.5,.75]:
            if method=='euclidean':dd=cdist(((1-t)*eu[i]+t*eu[j])[None,:],eu)[0]/escale
            else:
                if method=='main_origin_tangent':
                    point=H.exp_map0((1-t)*H.log_map0(hyp[i:i+1])+t*H.log_map0(hyp[j:j+1]))
                else:
                    delta=H.mobius_add(-hyp[i:i+1],hyp[j:j+1]);norm=np.linalg.norm(delta)
                    scaled=delta*np.tanh(t*np.arctanh(min(norm,1-1e-9)))/max(norm,1e-12)
                    point=H.mobius_add(hyp[i:i+1],scaled)
                dd=H.poincare_dist(point,hyp)[0]/hscale
            logw=-dd*dd/(2*bandwidth**2);w=np.exp(logw-logw.max());w/=w.sum()
            raw=(w@mus+res)*d['sd_x']+d['mu_x'];raw=np.clip(raw,x.min(0),x.max(0))
            gd,n=guard(raw,x[src],keep);std=(gd-d['mu_x'])/d['sd_x']
            row=dict(method=method,t=t,nearest_distance=float(cdist(std,z).min(1).mean()),guard_fixes=n,
                     **mmd_energy(z,std,500))
            rows.append(row);mid[f'{method}_{t}']=gd
            for c,wc in zip(cats,w):weights.append(dict(method=method,t=t,style=c,weight=wc))
    df=pd.DataFrame(rows);fig,ax=plt.subplots()
    for method,dd in df.groupby('method'):ax.plot(dd.t,dd.nearest_distance,'-o',label=method)
    ax.set(xlabel='t',ylabel='Mean nearest real-sample distance');ax.legend();fig.tight_layout()
    fig.savefig(QOUT/'fig08_path.png',dpi=180);plt.close(fig)
    np.savez_compressed(QOUT/'08_path_controls.npz',**mid,fold_id=FID,r2_dim=np.full(50,np.nan),pred=np.empty((0,50)))
    report(QOUT/'08_path_controls.md','C20 路径任务对照与应力复核',
           {'metrics':df,'weights':pd.DataFrame(weights),'common_stress':pd.DataFrame([dict(hyperbolic=hs,euclidean=es,ratio=es/max(hs,1e-12))]),
            'calibration_stress':pd.DataFrame(calibrations)},
           f'同口径stress-1：双曲={hs:.6f}，欧氏={es:.6f}，欧氏/双曲={es/max(hs,1e-12):.4f}。原应力优势不再成立。'
           '现行路径是原点切空间线性插值，并非一般两点间测地线，另补真正Mobius测地线。'
           '两空间先归一平均成对距离，再共享双曲侧带宽，保证带宽数值可比。'
           '两拟合器优化目标不同，此为现行实现的比较，并非两类几何的最优性能界。'
           '三口径应力另表并列；路径任务主表沿用现行sd坐标。'
           '最近邻与MMD可能给出不同排序，当前结果不能支持双曲路径全面优于欧氏路径。'
           '中间态最近邻包含源训练点且以全部真实样本为参照，是描述性真实感，不能证明存在真实中间风格。')


if __name__=='__main__':main()

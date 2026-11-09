from shared import *
from sklearn.ensemble import ExtraTreesRegressor,RandomForestRegressor
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    loaded=np.load(ROOT/'Output/Q2/02c_scores.npz');d={k:loaded[k] for k in loaded.files};assert X.shape== (250,240)
    rho,coef,graph=d['rho'],d['coef'],d['imp_graph']
    # Validation-fold permutation importance of a 200-tree joint forest proxy, per design budget.
    def permutation_fold(item):
        f,(tr,te)=item;local_perm=np.zeros_like(rho)
        cache=QOUT/f'07_perm_fold{f}.npz'
        if cache.exists():
            with np.load(cache) as stored:return stored['importance']
        a,b,_=FE.sentinel_fill(RAW[tr],RAW[te]);ma,mb,_=FE.miss_indicators(RAW[tr],RAW[te])
        # Reconcile fold-local missing indicator set with canonical full candidate indices.
        local=np.r_[np.arange(len(CAND)),len(CAND)+np.searchsorted(MIDX,np.where(np.isnan(RAW[tr]).any(0))[0])]
        aa,bb=np.c_[a,ma],np.c_[b,mb]
        model=ExtraTreesRegressor(n_estimators=C.trees(200),max_features=.5,n_jobs=1,
                                 random_state=C.SEED+f).fit(aa,Y[tr])
        base_prediction=model.predict(bb)
        baseline=((Y[te]-base_prediction)**2).mean(0)
        # Candidate nomination from training correlation only, union across dimensions.
        training_rho=spearman(aa,Y[tr]);nom=np.unique(np.argsort(-abs(training_rho),axis=0)[:5])
        rng=np.random.default_rng(C.SEED+f)
        for i in nom:
            delta=[]
            for rep in range(5):
                cp=bb.copy();cp[:,i]=rng.permutation(cp[:,i]);delta.append(((Y[te]-model.predict(cp))**2).mean(0)-baseline)
            local_perm[local[i]]=np.mean(delta,axis=0)/len(FOLDS)
        save_npz(cache,importance=local_perm,pred=base_prediction,fold_id=FID,test_index=te,
                 r2_dim=C.macro_r2(Y[te],base_prediction))
        return local_perm
    perm=np.sum(parallel(permutation_fold,list(enumerate(FOLDS))),axis=0)
    def per_target(j):
        cache=QOUT/f'07_chain_target{j}.npz'
        if cache.exists():
            with np.load(cache) as stored:return stored['gain'],stored['hit']
        rng=np.random.default_rng(C.SEED+j);gain=np.zeros(X.shape[1]);hit=np.zeros(X.shape[1])
        def local_matrix(ix):
            a,_,_=FE.sentinel_fill(RAW[ix],RAW[ix])
            return np.c_[a,np.isnan(RAW[ix][:,MIDX]).astype(float)]
        for b in range(30):
            ix=rng.integers(0,len(Y),len(Y))
            gain+=RandomForestRegressor(n_estimators=C.trees(200),max_features='sqrt',n_jobs=1,
                random_state=C.SEED+b).fit(local_matrix(ix),Y[ix,j]).feature_importances_/30
        for b in range(200):
            ix=rng.choice(len(Y),int(.7*len(Y)),replace=False)
            subset=local_matrix(ix);sc=StandardScaler().fit(subset);yy=Y[ix,j];yy=(yy-yy.mean())/max(yy.std(),1e-9)
            m=Lasso(alpha=float(d['alpha'][j]),max_iter=20000).fit(sc.transform(subset),yy)
            hit+=(abs(m.coef_)>1e-10)/200
        save_npz(cache,gain=gain,hit=hit,fold_id=FID,pred=np.empty((0,50)),r2_dim=np.full(50,np.nan))
        return gain,hit
    vals=parallel(per_target,list(range(50)));gain=np.column_stack([v[0] for v in vals]);stability=np.column_stack([v[1] for v in vals])
    methods=dict(spearman=abs(rho),lasso=abs(coef),diffusion=graph,permutation=np.maximum(perm,0),RF=gain,stability=stability)
    tops={k:top(v,5) for k,v in methods.items()};names=list(methods);matrix=np.eye(6);votes=[]
    for i,a in enumerate(names):
        for j,b in enumerate(names):matrix[i,j]=np.mean([jaccard(tops[a][:,t],tops[b][:,t]) for t in range(50)])
    for j,dim in enumerate(C.TARGET_COLS):
        counts={i:sum(i in tops[k][:,j] for k in names) for i in ELIG}
        eligible=sorted([i for i in ELIG if counts[i]>=3],key=lambda i:(-counts[i],-abs(rho[i,j])))[:3]
        for rank,i in enumerate(eligible):votes.append(dict(dim=dim,rank=rank+1,factor=FEAT[i],votes=counts[i]))
    fig,ax=plt.subplots(figsize=(7,6));im=ax.imshow(matrix,vmin=0,vmax=1)
    ax.set_xticks(range(6),names,rotation=40,ha='right');ax.set_yticks(range(6),names);fig.colorbar(im,ax=ax)
    fig.tight_layout();fig.savefig(QOUT/'fig07_methods.png',dpi=180);plt.close(fig)
    np.savez_compressed(QOUT/'07_method_matrix.npz',**methods,fold_id=FID,r2_dim=np.full(50,np.nan),pred=np.empty((0,50)))
    report(QOUT/'07_method_matrix.md','C11 六链方法一致性',
           {'jaccard':pd.DataFrame(matrix,columns=names).assign(method=names),'votes':pd.DataFrame(votes)},
           f'六链Top5两两Jaccard均值={matrix[np.triu_indices(6,1)].mean():.4f}；50维中{len(set(r["dim"] for r in votes))}维有至少一项≥3链支持，共{len(votes)}个投票席位。'
           '候选=231化学/理化/汇总列+9缺失指示，共240；正式化合物榜排除理化、汇总和缺失列。'
           'permutation是按设计预算使用的200树小森林代理，并非BASE5重要性；训练折提名候选，验证折5次置换。'
           'RF为30次bootstrap，Lasso为200次70%子采样；固定已有alpha，稳定性属描述性，无错误发现率保证。'
           '投票≥3/6才列出，少于3项不强行补足。六链相互相关，投票不等于独立证据。')


if __name__=='__main__':main()

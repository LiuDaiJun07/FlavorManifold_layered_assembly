"""C10: training-only top3 selection versus 1000 random triples."""
from shared import *
from sklearn.linear_model import Ridge, Lasso, lasso_path
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import ExtraTreesRegressor
from common_stats import permutation_p
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def composite_selection(tr,f):
    cache=QOUT/f'06_selection_f{f}.npz'
    if cache.exists():return np.load(cache)['selected']
    raw=RAW[tr];n=len(tr)
    def local(a,b):
        aa,bb,_=FE.sentinel_fill(a,b)
        aa=np.c_[aa,np.isnan(a[:,MIDX]).astype(float)]
        bb=np.c_[bb,np.isnan(b[:,MIDX]).astype(float)]
        sc=StandardScaler().fit(aa)
        return sc.transform(aa),sc.transform(bb)
    # Fold-local inner selection of one alpha per target, with independent preprocessing.
    grid=np.logspace(3,-3,13);loss=np.zeros((13,50))
    for itr,ite in KFold(5,shuffle=True,random_state=C.SEED+f).split(tr):
        aa,bb=local(raw[itr],raw[ite]);ys=StandardScaler().fit(Y[tr[itr]])
        ya=ys.transform(Y[tr[itr]]);yb=ys.transform(Y[tr[ite]])
        for j in range(50):
            _,co,_=lasso_path(np.asfortranarray(aa),ya[:,j],alphas=grid,max_iter=20000)
            loss[:,j]+=np.mean((bb@co-yb[:,j,None])**2,axis=0)
    alpha=grid[loss.argmin(0)];a,_=local(raw,raw);ys=StandardScaler().fit_transform(Y[tr])
    co=np.column_stack([Lasso(alpha=alpha[j],max_iter=20000).fit(a,ys[:,j]).coef_ for j in range(50)])
    rho=spearman(a,Y[tr]);hit=np.zeros_like(co);rng=np.random.default_rng(C.SEED+f)
    for _ in range(200):
        ix=rng.integers(0,n,n);aa,_=local(raw[ix],raw[ix]);yy=StandardScaler().fit_transform(Y[tr[ix]])
        for j in range(50):
            coef=Lasso(alpha=alpha[j],max_iter=20000).fit(aa,yy[:,j]).coef_
            hit[:,j]+=((abs(coef)>0)&(np.sign(coef)==np.sign(co[:,j])))/200
    w=abs(spearman(a[:,:len(CAND)],a[:,:len(CAND)]));np.fill_diagonal(w,0)
    w/=np.maximum(w.sum(1,keepdims=True),1e-12)
    imp=abs(co);imp[:len(CAND)]=.5*imp[:len(CAND)]+.5*w@imp[:len(CAND)]
    score=.4*rank01(abs(rho))+.4*rank01(imp)+.2*hit
    selected=top(score)
    np.savez_compressed(cache,selected=selected,alpha=alpha,coef=co,rho=rho,boot=hit,
                        train_index=tr,fold_id=FID,r2_dim=np.full(50,np.nan),pred=np.empty((0,50)))
    return selected


def main():
    if os.getenv('TOPK_SELECTION_ONLY',''):
        parallel(lambda item:composite_selection(item[1][0],item[0]),list(enumerate(FOLDS)))
        return
    nrandom=int(os.getenv('RANDOM_TRIPLES','1000'))
    # Same random triples across folds and targets; no target-based random-feature selection.
    rng=np.random.default_rng(C.SEED)
    triples=np.array([rng.choice(ELIG[ELIG<len(CAND)],3,replace=False) for _ in range(nrandom)])
    def task(item):
        f,(tr,te)=item
        imp=SimpleImputer(strategy='median',keep_empty_features=True)
        a,b=imp.fit_transform(RAW[tr]),imp.transform(RAW[te])
        sc=StandardScaler().fit(a);a,b=sc.transform(a),sc.transform(b)
        # Inference top3 must be selected anew on each training fold; full-data list is leaky.
        indices=composite_selection(tr,f)
        preds={k:np.zeros((len(te),50)) for k in ['top_ridge','all_ridge','top_ET','all_ET']}
        preds['all_ridge']=Ridge(alpha=1).fit(a,Y[tr]).predict(b)
        preds['all_ET']=ExtraTreesRegressor(n_estimators=C.trees(200),max_features=.5,
                            n_jobs=1,random_state=C.SEED+f).fit(a,Y[tr]).predict(b)
        for j in range(50):
            ix=indices[:,j]
            preds['top_ridge'][:,j]=Ridge(alpha=1).fit(a[:,ix],Y[tr,j]).predict(b[:,ix])
            preds['top_ET'][:,j]=ExtraTreesRegressor(n_estimators=C.trees(200),max_features=1.,
                n_jobs=1,random_state=C.SEED+f).fit(a[:,ix],Y[tr,j]).predict(b[:,ix])
        random=np.zeros((nrandom,len(te),50))
        for r,ix in enumerate(triples):
            random[r]=Ridge(alpha=1).fit(a[:,ix],Y[tr]).predict(b[:,ix])
        return te,preds,random,indices
    pp={k:np.zeros_like(Y) for k in ['top_ridge','all_ridge','top_ET','all_ET']}
    random_path=QOUT/'06_random_triples.npz'
    cached=all((QOUT/f'06_{k}.npz').exists() for k in pp) and random_path.exists()
    if cached:
        with np.load(random_path) as saved:
            cached=saved['pred'].shape[0]==nrandom and np.array_equal(saved['fold_id'],FID)
            if cached:rp,triples,selected=saved['pred'],saved['triples'],saved['selected']
    if cached:
        pp={k:np.load(QOUT/f'06_{k}.npz')['pred'] for k in pp}
    else:
        rp=np.zeros((nrandom,len(Y),50));selected=[]
        for te,p,r,ix in parallel(task,list(enumerate(FOLDS))):
            for k in pp:pp[k][te]=p[k]
            rp[:,te]=r;selected.append(ix)
    rr=np.array([C.macro_r2(Y,p) for p in rp]);rows=[]
    for k,p in pp.items():
        r=C.macro_r2(Y,p)
        np.savez_compressed(QOUT/f'06_{k}.npz',pred=p,fold_id=FID,r2_dim=r)
    rt=C.macro_r2(Y,pp['top_ridge']);ra=C.macro_r2(Y,pp['all_ridge'])
    et=C.macro_r2(Y,pp['top_ET']);ea=C.macro_r2(Y,pp['all_ET'])
    for j,dim in enumerate(C.TARGET_COLS):
        rows.append(dict(dim=dim,r2_top3=rt[j],r2_all=ra[j],random_mean=rr[:,j].mean(),
            random_p=permutation_p(rt[j],rr[:,j]),retention=rt[j]/ra[j] if ra[j]>0 else np.nan,
            r2_top3_ET=et[j],r2_all_ET=ea[j],retention_ET=et[j]/ea[j] if ea[j]>0 else np.nan))
    df=pd.DataFrame(rows)
    solver_warnings=sum(path.read_text(encoding='utf-8',errors='replace').count('ConvergenceWarning')
                        for path in (ROOT/'compare/provenance/logs').glob('C10*.log'))
    if not cached:np.savez_compressed(random_path,pred=rp,fold_id=FID,r2_dim=rr,triples=triples,selected=selected)
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,key,label in zip(axes,['retention','retention_ET'],['Ridge','ExtraTrees']):
        ax.hist(df[key].dropna(),bins=20);ax.set(xlabel='Top3/full R² (full R² > 0)',ylabel='Target count',title=label)
    fig.tight_layout();fig.savefig(QOUT/'fig06_retention.png',dpi=180);plt.close(fig)
    report(QOUT/'06_topk_predictive.md','C10 Top3预测验证',
           {'dimensions':df,'metrics':pd.DataFrame([metrics(k,p,pp['all_ridge' if k.endswith('ridge') else 'all_ET']) for k,p in pp.items()])},
           f'每个训练折重新执行Spearman+Lasso+图扩散+200次bootstrap综合Top3选择；图也由训练折重新估计，未使用全样本2.csv或6.csv。'
           'Lasso alpha在内层5折按13点网格选；原综合算法的选择权重保持0.4/0.4/0.2。验证对象是可重训选择算法，不是已经看过全样本标签的固定原名单。'
           f'{nrandom}个随机三元组固定跨折；经验p衡量优于随机子集，不是因果或标签置换检验。'
           f'岭回归Top3宏R²={rt.mean():.4f}，全列={ra.mean():.4f}；全模型R²≤0的保留率无定义。'
           f'ET Top3宏R²={et.mean():.4f}，全列={ea.mean():.4f}；两个模型家族的保留率均单列。'
           '全列模型含理化指标和汇总列，Top3与随机三元组仅含单一化合物；因此保留率是给定候选口径下的预测保留率，不能直接解读为纯化学信息的集中度。'
           '随机子集检验未作50维多重比较校正；固定alpha=1，不以验证折选参。')
    if solver_warnings:
        path=QOUT/'06_topk_predictive.md'
        with path.open('a',encoding='utf-8') as handle:
            handle.write(f'\n\n数值限制：保存的C10运行日志记录{solver_warnings}次Lasso候选/筛选求解收敛警告（见logs/C10_selection_precompute.log）。预测有限性检查通过，但不能据此认定所有候选已精确收敛；特征排序的数值敏感性仍需谨慎解释。\n')


if __name__=='__main__':main()

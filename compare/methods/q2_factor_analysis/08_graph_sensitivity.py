from shared import *
from scipy.sparse.csgraph import minimum_spanning_tree
import itertools


def main():
    d=np.load(ROOT/'Output/Q2/02c_scores.npz');coef=abs(d['coef']);rho=abs(d['rho']);hit=d['boot']
    cols,r=C.load_graph_corr();idx=[cols.index(c) for c in CAND];r=abs(r[np.ix_(idx,idx)]);np.fill_diagonal(r,0)
    def scores(w,alpha):
        w=w/np.maximum(w.sum(1,keepdims=True),1e-12);imp=coef.copy()
        imp[:len(CAND)]=(1-alpha)*coef[:len(CAND)]+alpha*w@coef[:len(CAND)]
        return .4*rank01(rho)+.4*rank01(imp)+.2*hit
    def knn(k):
        mask=np.zeros_like(r,dtype=bool);ix=np.argsort(-r,axis=1)[:,:k]
        mask[np.arange(len(r))[:,None],ix]=True
        return r*(mask|mask.T)
    reference=top(scores(knn(8),.5));orig=top(d['S']);rows=[]
    configs=[(f'knn{k}',a,knn(k)) for k,a in itertools.product([5,8,10,15,20],[.3,.5,.7])]
    cost=np.maximum(1-r,1e-12);np.fill_diagonal(cost,10)
    mst=minimum_spanning_tree(cost).toarray();mask=(mst>0)|(mst.T>0)
    configs.append(('MST',.5,r*mask))
    # Partial correlation graph from shrinkage precision of rank chemistry.
    from sklearn.covariance import LedoitWolf
    from scipy.stats import rankdata
    precision=LedoitWolf().fit(StandardScalerSafe(X[:,:len(CAND)])).precision_
    partial=-precision/np.sqrt(np.diag(precision)[:,None]*np.diag(precision)[None,:]);np.fill_diagonal(partial,0)
    configs.append(('partial_gt03',.5,abs(partial)*(abs(partial)>.3)))
    for name,alpha,w in configs:
        t=top(scores(w,alpha))
        for j,dim in enumerate(C.TARGET_COLS):
            rows.append(dict(graph=name,alpha=alpha,dim=dim,edges=int((w>0).sum()//2),jaccard=jaccard(reference[:,j],t[:,j]),
                flip_fraction=1-len(set(reference[:,j])&set(t[:,j]))/3,
                original_fullgraph_jaccard=jaccard(orig[:,j],t[:,j])))
    df=pd.DataFrame(rows)
    np.savez_compressed(QOUT/'08_graph_sensitivity.npz',top_reference=reference,original_fullgraph_top=orig,
        fold_id=FID,pred=np.empty((0,50)),r2_dim=np.full(50,np.nan))
    report(QOUT/'08_graph_sensitivity.md','C12 建图扩散敏感性',
           {'summary':df.groupby(['graph','alpha'],as_index=False)[['jaccard','flip_fraction','original_fullgraph_jaccard','edges']].mean(),'dimensions':df},
           f'15组k/α网格的平均席位翻转率范围={df[df.graph.str.startswith("knn")].groupby(["graph","alpha"]).flip_fraction.mean().min():.2%}–{df[df.graph.str.startswith("knn")].groupby(["graph","alpha"]).flip_fraction.mean().max():.2%}（包含参照组自身0翻转）。'
           '现行04_final实际在完整加权图扩散，没有用01_graph的kNN图；因此分别比较knn8/α=.5参照与原完整图。'
           '翻转率为未保留席位比例，不混同Jaccard距离。偏相关图采用秩变量Ledoit-Wolf精度矩阵，避免p≈n奇异。'
           '图来自全样本相关矩阵，属于解释性敏感性，不能作为无泄漏预测特征。')


def StandardScalerSafe(a):
    from sklearn.preprocessing import StandardScaler
    from scipy.stats import rankdata
    return StandardScaler().fit_transform(rankdata(a,axis=0))


if __name__=='__main__':main()

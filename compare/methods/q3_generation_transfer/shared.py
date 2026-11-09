import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'q1_prediction'))
from protocol import *
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.stats import ks_2samp
QOUT=ArtifactStore('Q3',TAG.strip('/'));QOUT.mkdir(parents=True,exist_ok=True)


def mmd_energy(a,b,permutations=0,seed=C.SEED,distribution_stats=False):
    z=np.vstack([a,b]);dist=cdist(z,z);d2=dist**2
    bw=max(float(np.median(d2[np.triu_indices(len(z),1)])),1e-12)
    kernel=np.exp(-d2/(2*bw));n=len(a)
    def stat(mat,ii,jj):return mat[np.ix_(ii,ii)].mean()+mat[np.ix_(jj,jj)].mean()-2*mat[np.ix_(ii,jj)].mean()
    ii=np.arange(n);jj=np.arange(n,len(z));mmd=stat(kernel,ii,jj)
    energy=-stat(dist,ii,jj)
    p=np.nan;ep=np.nan;extra={}
    if distribution_stats:
        sorted_rows=np.argsort(z,axis=0)
        sorted_values=np.take_along_axis(z,sorted_rows,axis=0)
        ends=np.vstack([np.diff(sorted_values,axis=0)!=0,np.ones((1,z.shape[1]),bool)])
        flatorder=np.argsort(z.ravel());flatrows=flatorder//z.shape[1]
        flatvalues=z.ravel()[flatorder];flatends=np.r_[np.diff(flatvalues)!=0,True]
        def descriptive(group):
            na,nb=group.sum(),(~group).sum();ha=group[sorted_rows]
            ds=np.max(np.where(ends,abs(np.cumsum(ha,axis=0)/na-np.cumsum(~ha,axis=0)/nb),0),axis=0)
            fa=group[flatrows]
            pooled=np.max(np.where(flatends,abs(np.cumsum(fa)/(na*z.shape[1])-np.cumsum(~fa)/(nb*z.shape[1])),0))
            ai,bi=np.where(group)[0],np.where(~group)[0]
            da=dist[np.ix_(ai,ai)].copy();np.fill_diagonal(da,np.inf)
            ratio=dist[np.ix_(bi,ai)].min(1).mean()/max(da.min(1).mean(),1e-12) if na>1 else np.nan
            return np.array([pooled,np.median(ds),np.mean(ds),abs(np.log(max(ratio,1e-12)))])
        observed=descriptive(np.arange(len(z))<n)
    if permutations:
        rng=np.random.default_rng(seed);null=[];energy_null=[];descriptive_null=[]
        for _ in range(permutations):
            order=rng.permutation(len(z));null.append(stat(kernel,order[:n],order[n:]))
            energy_null.append(-stat(dist,order[:n],order[n:]))
            if distribution_stats:
                group=np.zeros(len(z),bool);group[order[:n]]=True;descriptive_null.append(descriptive(group))
        from common_stats import permutation_p
        p=permutation_p(mmd,null);ep=permutation_p(energy,energy_null)
        if distribution_stats:
            null=np.asarray(descriptive_null)
            for j,key in enumerate(['ks_pooled_p','ks_dim_median_p','ks_dim_mean_p','nearest_ratio_p']):
                extra[key]=permutation_p(observed[j],null[:,j]) if np.isfinite(observed[j]) else np.nan
    return dict(mmd2=float(mmd),energy=float(energy),mmd_p=p,energy_p=ep,**extra)


def quality(a,b,permutations=500,classifier=False):
    stats=mmd_energy(a,b,permutations,distribution_stats=True)
    ks=[ks_2samp(a[:,j],b[:,j]).statistic for j in range(a.shape[1])]
    stats.update(ks_pooled=float(ks_2samp(a.ravel(),b.ravel()).statistic),ks_dim_median=float(np.median(ks)),
                 ks_dim_mean=float(np.mean(ks)))
    dd=cdist(a,a);np.fill_diagonal(dd,np.inf)
    stats['nearest_ratio']=float(cdist(b,a).min(1).mean()/max(dd.min(1).mean(),1e-12)) if len(a)>1 else np.nan
    if classifier:
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score,StratifiedGroupKFold
        model=make_pipeline(StandardScaler(),PCA(n_components=min(30,a.shape[1],len(a)-1),
                            svd_solver='full',random_state=C.SEED),LogisticRegression(max_iter=1000,random_state=C.SEED))
        combined=np.vstack([a,b]);_,groups=np.unique(np.round(combined,8),axis=0,return_inverse=True)
        stats['fake_classifier_accuracy']=float(cross_val_score(model,combined,
            np.r_[np.zeros(len(a)),np.ones(len(b))],cv=StratifiedGroupKFold(5,shuffle=True,random_state=C.SEED),
            groups=groups,scoring='balanced_accuracy').mean())
        stats['classifier_unique_groups']=int(len(np.unique(groups)))
    return stats


class Generator:
    def __init__(self,tr,q=None,kappa=5):
        from sklearn.covariance import LedoitWolf
        self.keep,_=FE.prune_cols(CHEM,rows=tr)
        raw=CHEM.iloc[tr][self.keep].to_numpy(float)
        self.med=np.nanmedian(raw,0);raw=np.where(np.isnan(raw),self.med,raw)
        self.mu=raw.mean(0);self.sd=np.maximum(raw.std(0),1e-9)
        self.lo,self.hi=raw.min(0),raw.max(0)
        self.z=(raw-self.mu)/self.sd;self.style=STYLE[tr];self.kappa=kappa
        self.cats=sorted(set(self.style));self.mus={};self.means={}
        for c in self.cats:
            a=self.z[self.style==c];self.means[c]=a.mean(0)
            self.mus[c]=(a.sum(0)+kappa*self.z.mean(0))/(len(a)+kappa)
        r=np.vstack([z-self.mus[s] for z,s in zip(self.z,self.style)])
        u,s,v=np.linalg.svd(r,full_matrices=False)
        if q is None:q=int(np.searchsorted(np.cumsum(s*s)/np.sum(s*s),.9)+1)
        q=min(q,len(s));self.q=q
        self.lam=v[:q].T*(s[:q]/np.sqrt(len(r)-1))
        self.sig=np.maximum(np.var(r-(u[:,:q]*s[:q])@v[:q],0),1e-6)
        self.cov=LedoitWolf().fit(r).covariance_
        self.eig,self.vec=np.linalg.eigh(self.cov)
        self.diag=np.var(r,0)
        ug,sg,vg=np.linalg.svd(self.z-self.z.mean(0),full_matrices=False)
        self.global_lam=vg[:1].T*sg[:1]/np.sqrt(len(r)-1)
        self.global_sig=np.maximum(np.var(self.z-(ug[:,:1]*sg[:1])@vg[:1],0),1e-6)

    def real(self,indices):
        a=CHEM.iloc[indices][self.keep].to_numpy(float)
        return (np.where(np.isnan(a),self.med,a)-self.mu)/self.sd

    def sample(self,method,c,n,rng):
        ix=self.style==c
        center=self.mus.get(c,self.z.mean(0));mean=self.means.get(c,self.z.mean(0))
        if method=='G0':g=mean+rng.normal(size=(n,len(mean)))*np.sqrt(self.diag)
        elif method=='G1':g=center+(rng.normal(size=(n,len(mean)))*np.sqrt(np.maximum(self.eig,0)))@self.vec.T
        elif method=='G2':
            residual=self.z[ix]-mean if ix.any() else self.z-self.z.mean(0)
            g=mean+residual[rng.integers(len(residual),size=n)]
        elif method=='G3':
            g=self.z.mean(0)+rng.normal(size=(n,1))@self.global_lam.T+rng.normal(size=(n,len(mean)))*np.sqrt(self.global_sig)
        else:
            g=center+rng.normal(size=(n,self.q))@self.lam.T+rng.normal(size=(n,len(mean)))*np.sqrt(self.sig)
            if ix.sum()>=5:
                # Calibrate against training style empirical CDF, with independent ranks.
                for j in range(g.shape[1]):
                    ref=np.sort(self.z[ix,j]);r=(np.argsort(np.argsort(g[:,j]))+.5)/n
                    g[:,j]=ref[np.minimum((r*len(ref)).astype(int),len(ref)-1)]
        return np.clip(g,(self.lo-self.mu)/self.sd,(self.hi-self.mu)/self.sd)


DEFECT=['DMS','X2.3.butanedione','X4VG','X.E..2.nonenal','humulene.epoxide.II',
        'X14.hydroxy.alpha.humulene','X.E..dihydro.apofarnesal']


def transfer_data():
    d=np.load(ROOT/'Output/Q3/01_generative.npz');keep=d['keep'].tolist()
    x=CHEM[keep].fillna(CHEM[keep].median()).to_numpy(float)
    z=(x-d['mu_x'])/d['sd_x'];cats=d['cats'].tolist();mus=d['mu_s'];lam=d['lam']
    src=np.where(STYLE=='Pils/Lager')[0];dst=np.where(STYLE=='Tripel')[0]
    residual=np.linalg.lstsq(lam,(z[src]-mus[cats.index('Pils/Lager')]).T,rcond=None)[0].T@lam.T
    return d,keep,x,z,cats,mus,lam,src,dst,residual


def guard(xnew,xsrc,keep):
    out=xnew.copy();count=0
    for c in DEFECT:
        if c in keep:
            j=keep.index(c);count+=int((out[:,j]>xsrc[:,j]).sum());out[:,j]=np.minimum(out[:,j],xsrc[:,j])
    return out,count


class LegacyReadout:
    """Exact E0/E1 point readout, exposing z before clipping for three calibrations."""
    def __init__(self):
        fm=importlib.import_module('final_model')
        self.views=[fm._View(CHEM,SENS,False,1200),fm._View(CHEM,SENS,True,1200)]

    def predict_z(self,frame,style):
        outputs=[]
        for view in self.views:
            raw=frame[view.keep].to_numpy(float)
            _,a,_=FE.sentinel_fill(view.train_X,raw)
            _,missing,_=FE.miss_indicators(view.train_X,raw);a=np.c_[a,missing]
            if view.use_style:
                st=np.zeros((len(a),len(view.cats)))
                if style in view.cats:st[:,view.cats.index(style)]=1
                a=np.c_[a,st]
            outputs.append(view.m.predict(a))
        return np.mean(outputs,axis=0)

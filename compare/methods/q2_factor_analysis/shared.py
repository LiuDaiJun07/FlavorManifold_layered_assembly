import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'q1_prediction'))
from protocol import *
from scipy.stats import rankdata
QOUT=ArtifactStore('Q2',TAG.strip('/'));QOUT.mkdir(parents=True,exist_ok=True)
CAND=C.chem_feature_cols(CHEM)
RAW=CHEM[CAND].to_numpy(float)
X,_,_=FE.sentinel_fill(RAW,RAW)
MI,_,MIDX=FE.miss_indicators(RAW,RAW)
X=np.c_[X,MI]
FEAT=CAND+[f'miss__{i}' for i in MIDX]
ELIG=np.array([i for i,n in enumerate(FEAT) if n not in C.PHYS_COLS+C.SUM_COLS and not n.startswith('miss__')])


def corr(a,b):
    a=a-a.mean(0);b=b-b.mean(0)
    return np.nan_to_num((a.T@b)/np.maximum(np.sqrt((a*a).sum(0)[:,None]*(b*b).sum(0)[None,:]),1e-12))


def spearman(a,b):return corr(rankdata(a,axis=0),rankdata(b,axis=0))


def top(m,k=3):return ELIG[np.argsort(-np.abs(m[ELIG]),axis=0)[:k]]


def jaccard(a,b):return len(set(a)&set(b))/len(set(a)|set(b))


def rank01(a):return np.argsort(np.argsort(a,axis=0),axis=0)/max(len(a)-1,1)

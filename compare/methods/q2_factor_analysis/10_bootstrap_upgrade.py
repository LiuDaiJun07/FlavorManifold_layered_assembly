"""C13: 1000 subsamples per fraction; full-sample BCa coefficient interval."""
from shared import *
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from common_stats import bca_from_replicates


def main():
    loaded=np.load(ROOT/'Output/Q2/02c_scores.npz');d={k:loaded[k] for k in loaded.files};rank1=top(d['S'],1)[0]
    B=int(os.getenv('BOOTSTRAPS','1000'))
    def task(j):
        path=QOUT/f'10_boot_target{j}.npz'
        if path.exists():
            with np.load(path) as cached:
                if len(cached['boot'])==B:return cached['rows']
        rng=np.random.default_rng(C.SEED+j);yy=Y[:,j];alpha=float(d['alpha'][j]);factor=int(rank1[j])
        def fit(ix):
            sc=StandardScaler().fit(X[ix]);y=yy[ix];y=(y-y.mean())/max(y.std(),1e-9)
            return Lasso(alpha=alpha,max_iter=20000).fit(sc.transform(X[ix]),y).coef_
        rows=[]
        for frac in [.5,.7,.9]:
            indicators=[]
            for b in range(B):
                co=fit(rng.choice(len(Y),round(frac*len(Y)),replace=False))
                indicators.append(float(abs(co[factor])>1e-10))
            p=np.mean(indicators);z=1.96;den=1+z*z/B;center=(p+z*z/(2*B))/den
            half=z*np.sqrt(p*(1-p)/B+z*z/(4*B*B))/den
            rows.append([j,frac,p,center-half,center+half,np.nan,np.nan])
        # BCa applies to rank-1 standardized Lasso coefficient, not selection-frequency MC trials.
        boot=np.array([fit(rng.integers(0,len(Y),len(Y)))[factor] for _ in range(B)])
        jack=np.array([fit(np.delete(np.arange(len(Y)),i))[factor] for i in range(len(Y))])
        point=fit(np.arange(len(Y)))[factor];low,high=bca_from_replicates(point,boot,jack)
        rows.append([j,1.,np.mean(abs(boot)>1e-10),np.nan,np.nan,low,high])
        rows=np.asarray(rows);np.savez_compressed(path,rows=rows,boot=boot,jack=jack,fold_id=FID,
            pred=np.empty((0,50)),r2_dim=np.full(50,np.nan));return rows
    results=np.vstack(parallel(task,list(range(50))))
    df=pd.DataFrame(results,columns=['target_index','fraction','selection_frequency','mc_wilson_low','mc_wilson_high','coefficient_bca_low','coefficient_bca_high'])
    df['dim']=[C.TARGET_COLS[int(j)] for j in df.target_index]
    df['rank1']=[FEAT[int(rank1[int(j)])] for j in df.target_index]
    frequency_bounds=[]
    for i,row in df.iterrows():
        successes=int(round(row.selection_frequency*B))
        indicators=np.r_[np.ones(successes),np.zeros(B-successes)]
        rng=np.random.default_rng(C.SEED+int(i))
        # Exact resampling distribution of the mean of B observed Bernoulli trials.
        replicates=rng.binomial(B,successes/B,size=1000)/B
        jack=(successes-indicators)/(B-1)
        frequency_bounds.append(bca_from_replicates(successes/B,replicates,jack))
        prob=successes/B;den=1+1.96**2/B
        center=(prob+1.96**2/(2*B))/den
        half=1.96*np.sqrt(prob*(1-prob)/B+1.96**2/(4*B*B))/den
        df.at[i,'mc_wilson_low']=max(0.,center-half);df.at[i,'mc_wilson_high']=min(1.,center+half)
    bounds=np.asarray(frequency_bounds)
    df['frequency_bca_mc_low']=bounds[:,0];df['frequency_bca_mc_high']=bounds[:,1]
    df['frequency_mc_trials']=B
    np.savez_compressed(QOUT/'10_bootstrap_upgrade.npz',frequency=df.selection_frequency.to_numpy(),
        frequency_bca_mc=bounds,coefficient_bca=df[['coefficient_bca_low','coefficient_bca_high']].to_numpy(),
        target_index=df.target_index.to_numpy(int),fraction=df.fraction.to_numpy(),fold_id=FID,
        pred=np.empty((0,50)),r2_dim=np.full(50,np.nan))
    averages=df.groupby('fraction').selection_frequency.mean()
    whole=df[df.fraction==1]
    excludes=int(((whole.coefficient_bca_low>0)|(whole.coefficient_bca_high<0)).sum())
    report(QOUT/'10_bootstrap_upgrade.md','C13 Bootstrap协议升级',{'stability':df},
           f'每维每比例{B}次无放回子采样，频率区间为Wilson蒙特卡洛误差区间。另用{B}次全样本bootstrap+250次jackknife给rank1标准化Lasso系数BCa区间。'
           '不把对重抽样0/1结果再次bootstrap得到的区间冒充原数据不确定性。原数据BCa目标为系数，选择频率的原数据BCa仍需嵌套重抽样，未声称已完成。'
           '另提供frequency_bca_mc区间：它仅表示固定数据集上条件入选概率的蒙特卡洛误差，与系数的原数据BCa分列；频率0或1时该BCa会退化，应同时看非退化的Wilson区间。'
           f'50%/70%/90%子采样平均入选频率={averages[.5]:.4f}/{averages[.7]:.4f}/{averages[.9]:.4f}；全样本系数BCa不含0有{excludes}/50维。'
           'alpha与rank1沿用原全样本选择，区间不修正选择后推断，只作描述性稳健性。')


if __name__=='__main__':main()

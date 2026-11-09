from protocol import *
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.ensemble import ExtraTreesRegressor
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    reference = original('E1'); rows = [metrics('sentinel', reference, reference)]
    configs = [(x,0.,False) for x in ['median','knn','zero','drop']]
    configs += [('sentinel',r,w) for r in [.05,.1,.2] for w in [False,True]]
    for method,rate,weighted in configs:
        name = f'{method}_{rate:g}_{int(weighted)}'
        def fitter(tr,te,f):
            # Missingness injection only on training observations. Validation stays original.
            data = CHEM.copy()
            if rate:
                cols = C.chem_feature_cols(data)
                a = data.iloc[tr][cols].to_numpy(float)
                rng = np.random.default_rng(C.SEED+f)
                prob = np.ones(a.shape[1])
                if weighted:
                    prob = np.isnan(a).mean(0) + 1/len(tr)
                    prob /= prob.mean()
                # Calibrate clipped probabilities to attain requested expected observed-cell rate.
                observed = np.isfinite(a); low,high=0.,100.
                for _ in range(50):
                    mid=(low+high)/2
                    actual=np.sum(np.minimum(1,mid*prob)*observed)/observed.sum()
                    if actual < rate: low=mid
                    else: high=mid
                mask=(rng.random(a.shape)<np.minimum(1,high*prob)) & observed
                a[mask]=np.nan
                data.loc[data.index[tr],cols]=a
            if method == 'sentinel': return fit_one(tr,te,chem=data,seed_off=f)
            keep,_=FE.prune_cols(data,rows=tr)
            a,b=data.iloc[tr][keep].to_numpy(float),data.iloc[te][keep].to_numpy(float)
            if method=='drop':
                ix=np.isfinite(a).all(0)
                a,b=a[:,ix],b[:,ix]
                # Any validation-only missing value uses training median, never test statistics.
                imp=SimpleImputer(strategy='median'); a=imp.fit_transform(a); b=imp.transform(b)
            else:
                imp=KNNImputer(n_neighbors=10) if method=='knn' else SimpleImputer(
                    strategy='constant' if method=='zero' else 'median',fill_value=0)
                a,b=imp.fit_transform(a),imp.transform(b)
            sa,sb,_=FE.style_onehot(STYLE[tr],STYLE[te])
            return ExtraTreesRegressor(n_estimators=C.trees(300),max_features=.5,
                    random_state=C.SEED+f,n_jobs=1).fit(np.c_[a,sa],Y[tr]).predict(np.c_[b,sb])
        p=cv('15_'+name,fitter=fitter)
        rows.append(dict(method=method,rate=rate,weighted=weighted,**metrics(name,p,reference)))
    df=pd.DataFrame(rows)
    fig,ax=plt.subplots()
    for w in [False,True]:
        z=df[(df.method=='sentinel')&(df.weighted==w)].sort_values('rate')
        ax.plot(np.r_[0,z.rate],np.r_[rows[0]['macro_r2'],z.macro_r2],'-o',label=str(w))
    ax.set(xlabel='Added training missing rate',ylabel='OOF macro R²');ax.legend(title='Weighted by column missingness')
    fig.tight_layout();fig.savefig(OUT/'fig15_missing.png',dpi=180);plt.close(fig)
    report(OUT/'15_missing_stress.md','C6 缺失机制压力测试',{'metrics':df},
           f'天然缺失下，哨兵宏R²={rows[0]["macro_r2"]:.6f}，常规填法范围={df[df.rate==0].macro_r2.min():.6f}–{df[df.rate==0].macro_r2.max():.6f}，未显示哨兵包的稳定优势。'
           f'训练观测额外缺失20%时，均匀掩码R²={df[(df.rate==.2)&(df.weighted==False)].macro_r2.iloc[0]:.6f}，加权掩码={df[(df.rate==.2)&(df.weighted==True)].macro_r2.iloc[0]:.6f}，性能明显下降。'
           '压力仅作用训练折观测单元；测试折保持真实缺失模式。加缺失后重新折内裁剪；两种掩码独立于标签。'
           '人工掩码造成整列训练观测消失时，从化合物块删除该列，避免无定义哨兵；缺失指示仍由训练折决定。'
           'kNN填充采用原量纲欧氏距离，作为常规填法对照，量纲影响须保留。')


if __name__=='__main__':main()

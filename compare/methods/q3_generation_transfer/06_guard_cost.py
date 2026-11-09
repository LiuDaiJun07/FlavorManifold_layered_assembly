from shared import *


def main():
    d=np.load(QOUT/'05_transfer_baselines.npz');keep=d['keep'].tolist();rows=[];details=[]
    _,_,reference,_,_,_,_,_,dst,_=transfer_data()
    lo,hi=reference.min(0),reference.max(0)
    tolerance=1e-9*np.maximum(1,np.abs(reference).max(0))
    non=[i for i,c in enumerate(keep) if c not in DEFECT]
    quality_table=pd.read_csv(QOUT/'05_transfer_baselines_metrics.csv').set_index(['method','state'])
    for k in ['T0','T1','T2','T3']:
        raw,gd=d[k+'_raw'],d[k+'_guarded']
        shift=float(np.abs(gd[:,non]-raw[:,non]).sum())
        raw_metrics=quality_table.loc[(k,'raw')];guarded_metrics=quality_table.loc[(k,'guarded')]
        rows.append(dict(method=k,nondef_absolute_change=shift,
            affected_nondef_fraction=float(np.mean(abs(gd[:,non]-raw[:,non])>1e-12)),
            sensory_RMSE_change=float(np.sqrt(np.mean((d[k+'_guarded_sensory'].mean(0)-d[k+'_raw_sensory'].mean(0))**2))),
            target_RMSE_raw=raw_metrics.rmse_portrait,target_RMSE_guarded=guarded_metrics.rmse_portrait,
            target_MMD_raw=raw_metrics.mmd2,target_MMD_guarded=guarded_metrics.mmd2))
        rows[-1]['support_violation_raw']=float(np.mean((raw<lo-tolerance)|(raw>hi+tolerance)))
        rows[-1]['support_violation_guarded']=float(np.mean((gd<lo-tolerance)|(gd>hi+tolerance)))
        for mode in ['unit','sd','minmax']:
            rows[-1]['target_RMSE_raw_'+mode]=raw_metrics['rmse_'+mode]
            rows[-1]['target_RMSE_guarded_'+mode]=guarded_metrics['rmse_'+mode]
            cm,cs=C.calib_params(Y,mode)
            before=C.z_to_score(d[k+'_raw_z'],cm,cs).mean(0)
            after=C.z_to_score(d[k+'_guarded_z'],cm,cs).mean(0)
            rows[-1]['sensory_change_'+mode]=float(np.sqrt(np.mean((after-before)**2)))
        for c in DEFECT:
            if c not in keep:continue
            j=keep.index(c);details.append(dict(method=k,compound=c,source=d['source'][:,j].mean(),
                unguarded=raw[:,j].mean(),guarded=gd[:,j].mean(),n_clipped=int((gd[:,j]!=raw[:,j]).sum())))
    np.savez_compressed(QOUT/'06_guard_cost.npz',**{key:d[key] for key in d.files},metrics=np.array([r['nondef_absolute_change'] for r in rows]))
    report(QOUT/'06_guard_cost.md','C18 守护代价',{'cost':pd.DataFrame(rows),'compounds':pd.DataFrame(details)},
           f"全部方法非缺陷修改量最大={max(r['nondef_absolute_change'] for r in rows):.6g}；只修改缺陷列是构造结果，不等于感官或联合分布无代价。"
           '目标画像RMSE（三口径）及MMD均在本表直接并列；守护能精确限制缺陷浓度，同时可能牺牲目标风格分布贴合度。'
           '另报原始/守护谱超出观测支持域的单元比例；观测支持域并非完整物理约束，零非缺陷改变量不能证明化合物谱物理合法。'
           '这些是固定11个源样本的描述性代价，不是独立外部泛化或因果检验。')


if __name__=='__main__':main()

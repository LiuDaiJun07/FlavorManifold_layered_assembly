from shared import *
from sklearn.neighbors import KNeighborsRegressor


def main():
    d,keep,x,z,cats,mus,lam,src,dst,res=transfer_data()
    transfer=np.load(QOUT/'05_transfer_baselines.npz')
    # 6.csv is already full-sample cross-correlation, used only as descriptive external reader.
    six=pd.read_csv(Path(C.ATT)/'6.csv',index_col=0)
    mapping=C.chem_name_map();rev={v:k for k,v in mapping.items()}
    sn=C.load_x8('Sensory names');sn=sn[sn.dataset.astype(str).str.contains('Trained')]
    dm={str(r.R_name):str(r.CleanName_PanelFeedback) for _,r in sn.iterrows()}
    w=np.zeros((len(keep),50));coverage=0
    for i,c in enumerate(keep):
        if c not in rev or rev[c] not in six.index:continue
        for j,t in enumerate(C.TARGET_COLS):
            if dm.get(t) in six.columns:
                value=six.loc[rev[c],dm[t]]
                try:w[i,j]=float(value);coverage+=1
                except (TypeError,ValueError):pass
    w=np.nan_to_num(w);w/=np.maximum(np.sqrt((w*w).sum(0)),1e-9)
    knn=KNeighborsRegressor(n_neighbors=10).fit(z,Y)
    mu,sg=C.load_scale();target_score=C.z_to_score(Y[dst],mu,sg).mean(0)
    linear_real=z@w
    linear_mean,linear_sd=linear_real.mean(0),np.maximum(linear_real.std(0),1e-9)
    def linear_readout(a):return (a@w-linear_mean)/linear_sd*Y.std(0)+Y.mean(0)
    if 'before_z' not in transfer:
        before_original=LegacyReadout().predict_z(pd.DataFrame(x[src],columns=keep),'Pils/Lager')
    else:before_original=transfer['before_z']
    rows=[];arrays={};legacy=[]
    for method in ['T0','T1','T2','T3']:
        a=(transfer[method+'_guarded']-d['mu_x'])/d['sd_x']
        for reader,pred,target,before in [
            ('6csv_direction',linear_readout(a),Y[dst].mean(0),linear_readout(z[src]).mean(0)),
            ('knn10',knn.predict(a),Y[dst].mean(0),knn.predict(z[src]).mean(0)),
            ('original_z',transfer[method+'_guarded_z'],Y[dst].mean(0),before_original.mean(0))]:
            rmse=float(np.sqrt(np.mean((pred.mean(0)-target)**2)))
            base=float(np.sqrt(np.mean((before-target)**2)))
            rows.append(dict(method=method,reader=reader,rmse=rmse,before_rmse=base,gain=base-rmse,
                corr=float(np.corrcoef(pred.mean(0),target)[0,1])))
            arrays[f'{method}_{reader}']=pred
        for mode in ['unit','sd','minmax']:
            cm,cs=C.calib_params(Y,mode)
            score=C.z_to_score(transfer[method+'_guarded_z'],cm,cs).mean(0)
            truth=C.z_to_score(Y[dst],cm,cs).mean(0)
            legacy.append(dict(method=method,calibration=mode,rmse=float(np.sqrt(np.mean((score-truth)**2))),
                               corr=float(np.corrcoef(score,truth)[0,1])))
    np.savez_compressed(QOUT/'07_readout_robust.npz',**arrays,fold_id=FID,pred=np.empty((0,50)),r2_dim=np.full(50,np.nan))
    result=pd.DataFrame(rows);main=result[result.method=='T3']
    consistent=bool((main.gain>0).all())
    report(QOUT/'07_readout_robust.md','C19 读出器独立性',{'metrics':result,'calibrations':pd.DataFrame(legacy)},
           f'6.csv读出覆盖{coverage}个交叉项，方向矩阵列L2归一后仅按真实谱/标签全局均值与标准差校准尺度，未拟合回归系数。'
           '三读出器主表均对真实目标风格z标签评价；另并列unit/sd/minmax的RMSE/相关作为标度敏感性，避免跨维反标准化偏移造成高相关假象。'
           'kNN包含源点自身作为邻居，迁移前对照可能偏乐观；6.csv为全样本，两个读出均非独立外部数据。'
           f'T3在三个读出器中迁移增益全为正：{consistent}；这支持增益方向的读出形式稳健性，但不证明T3优于其他迁移方法。')


if __name__=='__main__':main()

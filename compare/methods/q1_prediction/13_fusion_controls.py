from protocol import *
from sklearn.model_selection import KFold
from sklearn.linear_model import RidgeCV
from common_stats import paired


def main():
    if NF==10 and SCALE==1:
        outer = np.stack([BASE[f'{k}_pred'] for k in MC.MEMBERS], axis=2)
    else:
        outer=np.zeros((len(Y),50,5))
        def outer_fit(item):
            f,(tr,te)=item;keep,_=FE.prune_cols(CHEM,rows=tr)
            a,b,_=FE.build(CHEM,STYLE,tr,te,blocks=('chem','miss'),feat_cols=keep)
            points,_=MC.fit_members(a,b,Y[tr],len(keep),seed_off=f)
            return te,np.stack([points[k] for k in MC.MEMBERS],axis=2)
        for te,points in parallel(outer_fit,list(enumerate(FOLDS))):outer[te]=points
    ref=outer.mean(2)
    if NF==10:
        loaded=np.load(ROOT/'Output/Q1/06b2_inner_base.npz')
        inner_protocol={k:loaded[k] for k in loaded.files}
    else:inner_protocol=None
    if inner_protocol is not None:assert np.array_equal(inner_protocol['fold_id'],FID)
    rows = [metrics('equal', ref, ref)]
    for k, name in enumerate(MC.MEMBERS):
        p = np.delete(outer, k, axis=2).mean(2)
        np.savez_compressed(OUT / f'13_drop_{name}.npz', pred=p, fold_id=FID, r2_dim=C.macro_r2(Y,p))
        rows.append(metrics('drop_' + name, p, ref))
    # Existing inner cache contains only mean, not five members. Refit genuine nested OOF.
    predictions = {k: np.zeros_like(Y) for k in ['inverse_error','inner_select','ridge_stack','leaky_select']}
    def prepare(item):
        f,(tr,te)=item
        path = OUT / f'13_inner_members_f{f}.npz'
        if path.exists(): inner = np.load(path)['pred']
        else:
            inner = np.zeros((len(tr), 50, 5))
            if inner_protocol is not None:inner_id=inner_protocol[f'ibf_{f}'][tr]
            else:
                inner_id=np.full(len(tr),-1)
                for g,(_,ite) in enumerate(C.split_folds(STYLE[tr],n_folds=5)):inner_id[ite]=g
            for g in np.unique(inner_id):
                itr,ite=np.where(inner_id!=g)[0],np.where(inner_id==g)[0]
                tt, vv = tr[itr], tr[ite]
                keep, _ = FE.prune_cols(CHEM, rows=tt)
                a,b,_ = FE.build(CHEM,STYLE,tt,vv,blocks=('chem','miss'),feat_cols=keep)
                pts,_ = MC.fit_members(a,b,Y[tt],len(keep),seed_off=1000+f*10+int(g))
                inner[ite] = np.stack([pts[k] for k in MC.MEMBERS],axis=2)
            np.savez_compressed(path,pred=inner,train_index=tr,fold_id=FID[tr],inner_fold_id=inner_id)
        print('13 inner',f+1,'/',len(FOLDS),flush=True)
        return f,inner
    items=list(enumerate(FOLDS))
    subset=os.getenv('FUSION_FOLD_SUBSET','')
    if subset:
        selected={int(x) for x in subset.split(',')}
        items=[item for item in items if item[0] in selected]
    prepared=parallel(prepare,items)
    if subset:return
    for f,inner in prepared:
        tr,te=FOLDS[f]
        for j in range(50):
            # 1/(1-R²) is inverse residual variance; literal 1/R² is undefined for R²<=0.
            mse = ((inner[:,j]-Y[tr,j,None])**2).mean(0)
            weights = 1/np.maximum(mse,1e-12); weights /= weights.sum()
            predictions['inverse_error'][te,j] = outer[te,j] @ weights
            predictions['inner_select'][te,j] = outer[te,j,np.argmin(mse)]
            model = RidgeCV(alphas=np.logspace(-3,3,13)).fit(inner[:,j],Y[tr,j])
            predictions['ridge_stack'][te,j] = model.predict(outer[te,j])
    best = ((outer-Y[:,:,None])**2).mean(0).argmin(1)
    predictions['leaky_select'] = np.column_stack([outer[:,j,best[j]] for j in range(50)])
    for k,p in predictions.items():
        np.savez_compressed(OUT / f'13_{k}.npz',pred=p,fold_id=FID,r2_dim=C.macro_r2(Y,p))
        rows.append(metrics(k,p,ref))
    existing=[('06c_lsf.npz','lsf_def'),('06d_zits.npz','zits')] if NF==10 and SCALE==1 else []
    for file,key in existing:
        result=metrics(key,np.load(ROOT/'Output/Q1'/file)[key],ref)
        from scipy.stats import t
        se=result['delta_std']/np.sqrt(len(FOLDS));margin=.005
        result['ni_margin']=margin
        result['ni_lower95']=result['delta_mean']-t.ppf(.95,len(FOLDS)-1)*se
        result['ni_one_sided_p']=float(t.sf((result['delta_mean']+margin)/se,len(FOLDS)-1)) if se>0 else float(result['delta_mean']<=-margin)
        result['ni_exploratory_pass']=result['ni_lower95']>-margin
        rows.append(result)
    result_table=pd.DataFrame(rows).set_index('model')
    optimistic=result_table.loc['leaky_select','macro_r2']-result_table.loc['inner_select','macro_r2']
    report(OUT/'13_fusion_controls.md','C4 融合层对照',{'metrics':pd.DataFrame(rows)},
           f'等权宏R²={result_table.loc["equal","macro_r2"]:.6f}，内层选成员={result_table.loc["inner_select","macro_r2"]:.6f}，全量泄漏选成员={result_table.loc["leaky_select","macro_r2"]:.6f}；选择过拟合造成{optimistic:.6f}的乐观差距。'
           f'去M3仅提升到{result_table.loc["drop_M3","macro_r2"]:.6f}，逐维p={result_table.loc["drop_M3","p_dim"]:.6g}，不足以认定删除M3改善泛化。'
           '去成员结果直接复用同折成员预测，数学上等于重训同配置，避免无意义重复。'
           'inverse_error采用1/(1-R²)同义的逆MSE权重，避免1/R²对负值无定义。'
           'leaky_select明确为泄漏上界，不能交付；内层成员重新拟合。'
           'LSF/ZITS另给δ=.005、单侧95%折级t下界及非劣p值；CV折相关使其仅为探索性非劣性检查。'
           'LSF未显示显著收益，且本次单侧下界未建立其δ=.005非劣性；ZITS与基座相同，由零维开启的构造决定。'
           'p>0.05本身不能证明非劣性，正式确证仍需独立重复。')


if __name__ == '__main__': main()

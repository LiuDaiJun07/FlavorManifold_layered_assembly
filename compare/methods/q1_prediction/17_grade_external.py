from protocol import *


def main():
    p=original('BASE5')
    df=pd.DataFrame(dict(dim=C.TARGET_COLS,r2=C.macro_r2(Y,p),mae_z=np.abs(Y-p).mean(0)))
    path=Path(C.ATT)/'3.xlsx';x=pd.ExcelFile(path);cards=[]
    for sheet in x.sheet_names:
        raw=pd.read_excel(path,sheet_name=sheet,header=None)
        cells=[dict(row=i+1,col=j+1,value=str(v)) for i,row in enumerate(raw.to_numpy())
               for j,v in enumerate(row) if pd.notna(v)]
        cards.append(dict(sheet=sheet,nonempty=len(cells),numeric_cells=sum(
            isinstance(v,(float,int,np.number)) for v in raw.to_numpy().ravel() if pd.notna(v))))
        pd.DataFrame(cells).to_csv(OUT/f'17_card_{sheet}.csv',index=False,encoding='utf-8-sig')
    np.savez_compressed(OUT/'17_grade_external.npz',pred=p,fold_id=FID,r2_dim=C.macro_r2(Y,p),mae_z=np.abs(Y-p).mean(0))
    summary=pd.read_excel(path,sheet_name='summary')
    score_cols=[c for c in summary.columns if c not in ['Beer','comments']]
    all_zero=bool((summary[score_cols].fillna(0).to_numpy()==0).all())
    mapping_path=OUT/'17_card_mapping.csv'
    if not mapping_path.exists():
        pd.DataFrame(columns=['card_beer','dataset_beer','mapping_verified','scores_verified']).to_csv(mapping_path,index=False,encoding='utf-8-sig')
    mapping=pd.read_csv(mapping_path)
    usable=mapping[mapping.mapping_verified.astype(str).str.lower().isin(['true','1']) &
                   mapping.scores_verified.astype(str).str.lower().isin(['true','1'])]
    external=[]
    if len(usable):
        from sklearn.metrics import cohen_kappa_score
        card_scores=summary.rename(columns={'A_ester_fruity':'A_esters_fruity','F_ester_fruity':'F_esters_fruity','4vg':'X4vg'}).set_index('Beer')
        if usable.card_beer.duplicated().any():raise ValueError('Ambiguous card mapping')
        targets=[];predictions={mode:[] for mode in ['unit','sd','minmax']}
        for r in usable.itertuples():
            ix=np.flatnonzero(SENS.beer.to_numpy()==r.dataset_beer)
            if len(ix)!=1 or r.card_beer not in card_scores.index:raise ValueError('Unmatched verified card/beer')
            i=int(ix[0]);truth=card_scores.loc[r.card_beer,C.TARGET_COLS].to_numpy(float)
            if not np.isfinite(truth).all() or not np.all((truth>=0)&(truth<=6)) or not np.allclose(truth,np.round(truth)):
                raise ValueError('Verified cards must contain complete integer grades 0..6')
            targets.append(truth.astype(int))
            for mode in predictions:
                mu,sg=C.calib_params(Y[FID!=FID[i]],mode)
                predictions[mode].append(C.score_to_grade(C.z_to_score(p[i:i+1],mu,sg))[0])
        truth=np.asarray(targets)
        for mode,values in predictions.items():
            pred=np.asarray(values)
            external.append(dict(calibration=mode,n_cards=len(usable),n_beers=usable.dataset_beer.nunique(),
                n_cells=truth.size,accuracy=float(np.mean(pred==truth)),mae_grade=float(np.mean(abs(pred-truth))),
                kappa=cohen_kappa_score(truth.ravel(),pred.ravel(),labels=np.arange(7)),
                quadratic_kappa=cohen_kappa_score(truth.ravel(),pred.ravel(),labels=np.arange(7),weights='quadratic')))
        np.savez_compressed(OUT/'17_card_validation.npz',truth=truth,**{k:np.asarray(v) for k,v in predictions.items()},
            pred=p,fold_id=FID,r2_dim=C.macro_r2(Y,p),dataset_beer=usable.dataset_beer.to_numpy())
    report(OUT/'17_grade_external.md','C8 目标循环性与评分卡核查',
           {'z_metrics':df,'card_inventory':pd.DataFrame(cards),
            'external_validation':pd.DataFrame(external)},
           'z空间R²和MAE不依赖0–6反标准化；unit/sd/minmax MAE仅评价映射定义下的一致性。'
           f'3.xlsx实际有12张beer卡；summary的12×50评分全部为0：{all_zero}。'+
           ('现有附件为未填写模板，未提供核验后的映射/实填评分时，这些0不能当作12条真实全零观测，外部验证前提不成立。' if not len(usable)
            else '仅对显式核验的卡片映射和实填评分计算三口径一致率/Kappa；小样本且标签是否来自独立面板需另行披露，不是外部队列泛化证明。')+
           'Beer#是表内编号，不能假定等于数据集beer编号。17_card_mapping.csv保留可施行的后续核验入口；本次未使用Kaggle标签。')


if __name__=='__main__':main()

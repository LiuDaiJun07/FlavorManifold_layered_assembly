"""C14 prerequisite audit: do not manufacture unit conversions or threshold sources."""
from shared import *
import itertools


def main():
    path=QOUT/'threshold_sources.csv'
    columns=['compound','source_id','source_url','threshold','threshold_unit','matrix',
             'concentration_unit','concentration_scale','independence_group','conversion_verified']
    if not path.exists():
        pd.DataFrame(columns=columns).to_csv(path,index=False,encoding='utf-8-sig')
    data=pd.read_csv(path)
    if data.empty:
        for col in columns:
            if col not in data:data[col]=pd.Series(dtype=object)
        data.to_csv(path,index=False,encoding='utf-8-sig')
    verified=data[data.conversion_verified.astype(str).str.lower().isin(['true','1'])]
    if len(verified)==0:
        report(QOUT/'09_oav_sensitivity.md','C14 OAV来源敏感性：前提不足',{'sources':data},
               '未执行数值OAV稳健性检验：现有阈值表混合单位，原数据浓度单位未逐列证实，且没有两套独立、同基质且单位可核验的阈值。'
               'threshold_sources.csv仅为待核验数据结构，不把阈值倍数扰动冒充独立来源。检索记录见threshold_source_candidates.md。'
               '脚本已实现输入齐备后的单位换算、来源共同覆盖集Jaccard与翻转表；independence_group应填写原始阈值测量研究，不能用转载文献伪装独立来源。')
        return
    for col in columns:
        if col not in verified or verified[col].isna().any():raise ValueError('Missing verified source field: '+col)
    if verified.duplicated(['source_id','compound']).any():raise ValueError('Each source must have one threshold per compound')
    units={'g/l':1.,'mg/l':1e-3,'ug/l':1e-6,'ng/l':1e-9}
    def unit(value):
        key=str(value).strip().lower().replace('µ','u').replace('μ','u').replace(' ','')
        if key not in units:raise ValueError('Unsupported unit (no density/molarity assumptions): '+key)
        return units[key]
    rows=[]
    for r in verified.itertuples():
        if r.compound not in CHEM:raise ValueError('Unknown compound: '+r.compound)
        if str(r.concentration_scale) not in ['raw','log10']:raise ValueError('Explicit raw/log10 scale required')
        if not np.isfinite(float(r.threshold)) or float(r.threshold)<=0:raise ValueError('Threshold must be finite and positive')
        values=CHEM[r.compound].to_numpy(float)
        values=10**values if r.concentration_scale=='log10' else values
        observed=values[~np.isnan(values)]
        if not len(observed) or not np.isfinite(observed).all() or (observed<0).any():
            raise ValueError('Concentrations must be observed, finite and nonnegative; verify scale/units')
        concentration=float(np.nanmedian(values))*unit(r.concentration_unit)
        threshold=float(r.threshold)*unit(r.threshold_unit)
        rows.append(dict(compound=r.compound,source_id=r.source_id,source_url=r.source_url,
            independence_group=r.independence_group,matrix=r.matrix,concentration_gL=concentration,
            threshold_gL=threshold,oav=concentration/threshold,above_one=concentration>threshold))
    numeric=pd.DataFrame(rows);pairs=[];changes=[]
    for a,b in itertools.combinations(numeric.source_id.unique(),2):
        left=numeric[numeric.source_id==a].set_index('compound');right=numeric[numeric.source_id==b].set_index('compound')
        common=left.index.intersection(right.index)
        common=[c for c in common if left.at[c,'independence_group']!=right.at[c,'independence_group']
                and str(left.at[c,'matrix']).lower()==str(right.at[c,'matrix']).lower()]
        sa={c for c in common if left.at[c,'above_one']};sb={c for c in common if right.at[c,'above_one']}
        pairs.append(dict(source_a=a,source_b=b,n_independent_same_matrix_common=len(common),
            jaccard=len(sa&sb)/len(sa|sb) if sa|sb else (1. if common else np.nan),
            n_flips=len(sa^sb),only_a_coverage=len(left.index.difference(right.index)),only_b_coverage=len(right.index.difference(left.index))))
        for c in common:changes.append(dict(source_a=a,source_b=b,compound=c,oav_a=left.at[c,'oav'],
            oav_b=right.at[c,'oav'],flipped=(c in sa)!=(c in sb)))
    np.savez_compressed(QOUT/'09_oav_sensitivity.npz',oav=numeric.oav.to_numpy(),
        compounds=numeric.compound.to_numpy(str),sources=numeric.source_id.to_numpy(str),
        fold_id=FID,pred=np.empty((0,50)),r2_dim=np.full(50,np.nan))
    report(QOUT/'09_oav_sensitivity.md','C14 OAV来源敏感性',
        {'oav':numeric,'pairs':pd.DataFrame(pairs),'flips':pd.DataFrame(changes)},
        '仅比较单位、原始浓度标度和来源独立性已核验且基质相同的共同覆盖化合物。'
        '缺失覆盖不计作阈值翻转；共同覆盖少于原Top3全集时，不能推广为全部席位稳健。'
        '若没有至少一对独立、同基质且有共同覆盖的来源，本表仍只构成输入核查，不构成完成的OAV对照。')


if __name__=='__main__':main()

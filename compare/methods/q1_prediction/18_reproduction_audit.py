"""Trace the small historical/current BASE5 mismatch without changing mainline."""
from protocol import *


def main():
    tr,te=FOLDS[0];keep,_=FE.prune_cols(CHEM,rows=tr)
    a,b,_=FE.build(CHEM,STYLE,tr,te,blocks=('chem','miss'),feat_cols=keep)
    points,_=MC.fit_members(a,b,Y[tr],len(keep),seed_off=0)
    rows=[]
    for k,p in points.items():
        cached=BASE[k+'_pred'][te]
        rows.append(dict(member=k,max_abs_difference=float(np.max(abs(p-cached))),
                         mean_abs_difference=float(np.mean(abs(p-cached)))))
    np.savez_compressed(OUT/'18_reproduction_audit.npz',**points,test_index=te,fold_id=FID,
                        pred=np.mean(list(points.values()),axis=0),r2_dim=C.macro_r2(Y[te],np.mean(list(points.values()),axis=0)))
    report(OUT/'18_reproduction_audit.md','历史缓存复现核查（第1折）',{'members':pd.DataFrame(rows)},
           '固定同一折、同一参数与种子，在当前环境单线程BLAS重拟合，与历史逐成员缓存对照。'
           '本次仅M3（目标PCA）出现明显差异，其余四成员在约1e−15数值精度内一致；历史融合差异不可归因于折或其他成员。'
           '未保存历史PCA状态/运行环境/BLAS状态，不能断言具体库是唯一原因。当前实验固定BLAS单线程用于后续可重复性。')


if __name__=='__main__':main()

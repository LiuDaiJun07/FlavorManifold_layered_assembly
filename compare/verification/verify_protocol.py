"""Numerical and leakage-boundary checks for the new experiment infrastructure."""
import os
import sys
import json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'methods/q1_prediction'))
from protocol import *
from geometry_corrected import H as fixed
from common_stats import wilcoxon,friedman_nemenyi


def main():
    checks={}
    checks['fold_coverage']=bool(np.all(np.bincount(np.concatenate([te for _,te in FOLDS]),minlength=len(Y))==1))
    checks['fold_train_test_disjoint']=all(not set(tr)&set(te) for tr,te in FOLDS)
    checks['cached_folds_match_split']=bool(np.array_equal(FID,BASE['fold_id'])) if NF==10 else True
    if NF==10:
        strata=np.char.add(STYLE,np.where(G.mean(1)>=3,'_h','_l'))
        recreated=np.full(len(Y),-1)
        for f,(_,te) in enumerate(C.split_folds(strata,n_folds=NF)):recreated[te]=f
        checks['original_folds_exact_match']=bool(np.array_equal(FID,recreated))
    checks['wilcoxon_identical']=wilcoxon(np.arange(5),np.arange(5))==1
    z=np.array([[.01,.02],[-.03,.01]])
    checks['corrected_roundtrip']=all(np.allclose(fixed.exp_map0(fixed.log_map0(z,c),c),z,atol=1e-12) for c in [1e-8,.1,1.,10.])
    rng=np.random.default_rng(C.SEED);z=rng.normal(size=(5,2))*.03
    target=np.ones((5,5));np.fill_diagonal(target,0)
    gradient_errors=[]
    for c in [1e-8,.1,1.,10.]:
        loss,grad=fixed.stress_loss_grad(z,target,c)
        zp,zm=z.copy(),z.copy();zp[0,0]+=1e-7;zm[0,0]-=1e-7
        numeric=(fixed.stress_loss_grad(zp,target,c)[0]-fixed.stress_loss_grad(zm,target,c)[0])/2e-7
        gradient_errors.append(abs(numeric-grad[0,0]))
    checks['corrected_gradient']=bool(max(gradient_errors)<1e-5)
    # Inner pipeline must not change trained statistics if validation-only cells change.
    import importlib.util
    spec=importlib.util.spec_from_file_location('baselines',Path(__file__).resolve().parents[1]/'methods/q1_prediction/10_baselines.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    tr,te=FOLDS[0]
    transformer=module.FoldFeatures().fit(CHEM.iloc[tr]);keep=list(transformer.keep_)
    test=CHEM.iloc[te].copy();test.loc[:,keep[0]]=1e20
    transformer.transform(test)
    checks['inner_stats_unchanged_by_validation']=keep==transformer.keep_
    stressed=CHEM.copy();column=keep[0]
    stressed.loc[stressed.index[tr],column]=np.nan
    aa,bb,names=design(tr,te,chem=stressed)
    checks['all_missing_training_column_removed']=column not in names
    checks['stress_features_finite']=bool(np.isfinite(aa).all() and np.isfinite(bb).all())
    if NF!=10 or SCALE!=1:
        p=cv('verify_smoke_E1');checks['smoke_oof_finite']=bool(np.isfinite(p).all())
    result=dict(checks=checks,all_passed=all(checks.values()),gradient_max_error=max(gradient_errors),
                python=sys.version,numpy=np.__version__,pandas=pd.__version__)
    (OUT/'protocol_verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))
    assert all(checks.values())


if __name__=='__main__':main()

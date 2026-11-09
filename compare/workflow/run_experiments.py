"""Resumable experiment driver. Outputs remain entirely within compare/."""
import os
import sys
import time
import json
import hashlib
import subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
TASKS=[
 ('C5','methods/q1_prediction/14_fold_sensitivity.py'),('C9','methods/q2_factor_analysis/05_confound.py'),
 ('C8','methods/q1_prediction/17_grade_external.py'),('C12','methods/q2_factor_analysis/08_graph_sensitivity.py'),
 ('C14','methods/q2_factor_analysis/09_oav_sensitivity.py'),('C15_C16','methods/q3_generation_transfer/04_gen_baselines.py'),
 ('C17','methods/q3_generation_transfer/05_transfer_baselines.py'),('C18','methods/q3_generation_transfer/06_guard_cost.py'),
 ('C19','methods/q3_generation_transfer/07_readout_robust.py'),('C20','methods/q3_generation_transfer/08_path_controls.py'),
 ('C1','methods/q1_prediction/10_baselines.py'),('C2','methods/q1_prediction/11_repeated_cv.py'),
 ('C3','methods/q1_prediction/12_hyper_controls.py'),('C4','methods/q1_prediction/13_fusion_controls.py'),
 ('C6','methods/q1_prediction/15_missing_stress.py'),('C7','methods/q1_prediction/16_hyperparam_sensitivity.py'),
 ('C10','methods/q2_factor_analysis/06_topk_predictive.py'),('C11','methods/q2_factor_analysis/07_method_matrix.py'),
 ('C13','methods/q2_factor_analysis/10_bootstrap_upgrade.py'),('C21','methods/q3_generation_transfer/09_latent_sensitivity.py')]


def hashes():
    return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
            for folder in ['src','Output'] for p in (ROOT/folder).rglob('*') if p.is_file() and '__pycache__' not in str(p)}


def main():
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--only',default='');parser.add_argument('--rerun',action='store_true')
    args=parser.parse_args()
    compare=ROOT/'compare';logs=compare/'provenance/logs';logs.mkdir(parents=True,exist_ok=True)
    (compare/'provenance/manifests').mkdir(parents=True,exist_ok=True)
    (compare/'verification/reports').mkdir(parents=True,exist_ok=True)
    statuspath=compare/'provenance/execution/run_status.json'
    statuspath.parent.mkdir(parents=True,exist_ok=True)
    status=json.loads(statuspath.read_text(encoding='utf-8')) if statuspath.exists() else {}
    before=hashes();(compare/'provenance/manifests/mainline_sha256.json').write_text(json.dumps(before,indent=2),encoding='utf-8')
    env=os.environ.copy()
    for k,v in dict(NJOB='6',TREE_SCALE='1',FOLDS='10',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1',PYTHONWARNINGS='ignore::sklearn.exceptions.ConvergenceWarning').items():env.setdefault(k,v)
    env['PYTHONWARNINGS']='ignore::FutureWarning'
    for name,script in TASKS:
        if args.only and name not in args.only.split(','):continue
        digest=hashlib.sha256((compare/script).read_bytes()).hexdigest()
        if not args.rerun and status.get(name,{}).get('state')=='completed' and status[name].get('script_sha256')==digest:continue
        status[name]=dict(state='running',script=script,started=time.strftime('%Y-%m-%d %H:%M:%S'),script_sha256=digest)
        statuspath.write_text(json.dumps(status,indent=2,ensure_ascii=False),encoding='utf-8')
        print('START',name,flush=True);start=time.time()
        with (logs/(name+'.log')).open('w',encoding='utf-8') as log:
            result=subprocess.run([sys.executable,'-B','-u',str(compare/script)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        status[name].update(state='completed' if result.returncode==0 else 'failed',exit_code=result.returncode,elapsed_seconds=time.time()-start)
        statuspath.write_text(json.dumps(status,indent=2,ensure_ascii=False),encoding='utf-8')
        print('END',name,status[name]['state'],round(time.time()-start,1),flush=True)
    after=hashes();changed=[p for p in set(before)|set(after) if before.get(p)!=after.get(p)]
    (compare/'verification/reports/mainline_integrity.json').write_text(json.dumps(dict(unchanged=not changed,changed=changed),indent=2),encoding='utf-8')
    if changed:raise RuntimeError('Mainline changed: '+str(changed))
    failed=[k for k,v in status.items() if v.get('state')=='failed']
    if failed:raise RuntimeError('Experiments failed; see logs: '+','.join(failed))


if __name__=='__main__':main()

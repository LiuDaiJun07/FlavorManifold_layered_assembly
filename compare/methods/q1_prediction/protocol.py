"""Read-only bridge to original implementation. Never call original scripts."""
import os
import sys
import json
import hashlib
import importlib
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
sys.dont_write_bytecode = True

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'src/q1'))
sys.path.insert(0,str(ROOT/'compare/workflow'))
from artifact_layout import ArtifactStore, table_path
import numpy as np
import pandas as pd
import common as C
from threadpoolctl import threadpool_limits
threadpool_limits(1)
C.OUT = str(ROOT / 'Output/Q1')
C.OUT_Q2 = str(ROOT / 'Output/Q2')
C.OUT_Q3 = str(ROOT / 'Output/Q3')
FE = importlib.import_module('03_features')
MC = importlib.import_module('06b_member_core')
H = importlib.import_module('04_hyperbolic')
MC.NJOB = 1
requested_workers=int(os.getenv('NJOB','6'))
WORKERS=(os.cpu_count() or 1) if requested_workers==-1 else max(1,min(requested_workers,os.cpu_count() or 1))
NF = int(os.getenv('FOLDS', '10'))
SCALE = float(os.getenv('TREE_SCALE', '1'))
TAG = '' if NF == 10 and SCALE == 1 else f'/smoke_f{NF}_t{SCALE:g}'
OUT = ArtifactStore('Q1',TAG.strip('/'))
OUT.mkdir(parents=True, exist_ok=True)
CHEM, SENS = C.load_chem(), C.load_sensory()
Y = SENS[C.TARGET_COLS].to_numpy(float)
STYLE = SENS.tasting_category_fine.astype(str).to_numpy()
BASE = np.load(ROOT / 'Output/Q1/06b_members.npz', allow_pickle=False)
G = BASE['g']
assert np.array_equal(BASE['y'], Y)
assert CHEM.beer.is_unique and SENS.beer.is_unique
assert np.array_equal(CHEM.beer, SENS.beer), 'sample beer order mismatch'
assert np.array_equal(CHEM.tasting_category_fine, SENS.tasting_category_fine)
FID = BASE['fold_id'] if NF == 10 else np.full(len(Y), -1)
if NF != 10:
    for f, (_, te) in enumerate(C.split_folds(STYLE, n_folds=NF)):
        FID[te] = f
FOLDS = [(np.where(FID != f)[0], np.where(FID == f)[0]) for f in np.unique(FID)]
if not (OUT/'folds.npz').exists():
    np.savez_compressed(OUT / 'folds.npz', fold_id=FID, beer=SENS.beer.to_numpy(),
        beer_id=SENS.beer_id.to_numpy(),chem_barcode=CHEM.beer_id.to_numpy(),y=Y)


def parallel(fn, items):
    if os.getenv('COMPARE_BACKEND','loky')=='threading':
        with ThreadPoolExecutor(WORKERS) as pool:
            return list(pool.map(fn, items))
    from joblib import Parallel,delayed,parallel_config
    with parallel_config(backend='loky',n_jobs=WORKERS,inner_max_num_threads=1,
                         temp_folder=str(OUT/'joblib_temp'),max_nbytes=None):
        return Parallel()(delayed(fn)(item) for item in items)


def save_npz(path, **values):
    """Publish complete caches atomically when independent experiment workers overlap."""
    import uuid
    import time
    path=Path(path).resolve()
    assert path.is_relative_to((ROOT/'compare').resolve())
    temporary=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp.npz')
    np.savez_compressed(temporary,**values)
    for attempt in range(20):
        try:
            os.replace(temporary,path)
            return
        except PermissionError:
            if attempt==19:raise
            time.sleep(.02)


def table(df):
    # No optional tabulate dependency.
    def fmt(x):
        return f'{x:.6g}' if isinstance(x, (float, np.floating)) else str(x)
    return '\n'.join(['| ' + ' | '.join(map(str, df.columns)) + ' |',
                      '| ' + ' | '.join(['---'] * len(df.columns)) + ' |'] +
                     ['| ' + ' | '.join(fmt(x) for x in row) + ' |' for row in df.itertuples(index=False, name=None)])


def report(path, title, frames, conclusion):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    chunks = [f'# {title}', f'运行协议：FOLDS={NF}, TREE_SCALE={SCALE}, 当前并行任务上限={WORKERS}（缓存可复用同参数的先前计算）。',
              '检验为探索性：目标维度相关、CV训练集重叠，p值不可当独立重复证据；弱/强维为事后子集。']
    for label, df in frames.items():
        df.to_csv(table_path(path,label), index=False, encoding='utf-8-sig')
        chunks.extend([f'## {label}', table(df)])
    chunks += ['## 结论与限制', conclusion]
    path.write_text('\n\n'.join(chunks), encoding='utf-8')
    print(f'REPORT {path}', flush=True)


def design(tr, te, style=True, chem=None):
    frame=CHEM if chem is None else chem
    keep,_=FE.prune_cols(frame,rows=tr)
    keep=[c for c in keep if frame.iloc[tr][c].notna().any()]
    return FE.build(frame, STYLE, tr, te,feat_cols=keep,
                    blocks=('chem', 'miss', 'style') if style else ('chem', 'miss'))


def fit_one(tr, te, cfg='E1', seed_off=0, chem=None, **params):
    from sklearn.ensemble import ExtraTreesRegressor
    if cfg == 'BASE5':
        keep, _ = FE.prune_cols(CHEM, rows=tr)
        a, b, _ = FE.build(CHEM, STYLE, tr, te, blocks=('chem', 'miss'), feat_cols=keep)
        pts, _ = MC.fit_members(a, b, Y[tr], len(keep), seed_off=seed_off)
        return np.mean(list(pts.values()), axis=0)
    a, b, _ = design(tr, te, style=cfg != 'E0', chem=chem)
    kw = dict(n_estimators=C.trees(300), max_features=.5, min_samples_leaf=1,
              random_state=C.SEED + seed_off, n_jobs=1)
    kw.update(params)
    return ExtraTreesRegressor(**kw).fit(a, Y[tr]).predict(b)


def cv(name, cfg='E1', folds=None, seed=0, fitter=None, **params):
    folds = FOLDS if folds is None else folds
    path = OUT / (name + '.npz')
    fid = np.full(len(Y), -1)
    for f, (_, te) in enumerate(folds):
        fid[te] = f
    if path.exists():
        d = np.load(path)
        assert np.array_equal(d['fold_id'], fid)
        return d['pred']
    def task(item):
        f, (tr, te) = item
        checkpoint = OUT / f'{name}_fold{f}.npz'
        if checkpoint.exists():
            saved = np.load(checkpoint)
            assert np.array_equal(saved['test_index'], te)
            return te, saved['pred']
        pred = fitter(tr, te, f) if fitter else fit_one(tr, te, cfg, seed_off=f + seed, **params)
        assert np.isfinite(pred).all()
        np.savez_compressed(checkpoint,pred=pred,test_index=te,train_index=tr,fold_id=fid,
                            r2_dim=C.macro_r2(Y[te],pred),tree_scale=SCALE)
        print(name, f + 1, '/', len(folds), flush=True)
        return te, pred
    pred = np.full_like(Y, np.nan)
    for te, p in parallel(task, list(enumerate(folds))):
        pred[te] = p
    assert np.isfinite(pred).all()
    np.savez_compressed(path, y=Y, pred=pred, fold_id=fid, r2_dim=C.macro_r2(Y, pred),
                        tree_scale=SCALE, seed_offset=seed)
    return pred


def original(name):
    if NF != 10 or SCALE != 1:
        return cv(name, cfg=name)
    if name == 'BASE5':
        return BASE['base']
    return np.load(ROOT / 'Output/Q1/05_oof.npz')[name + '_pred']


def metrics(name, pred, reference=None, fid=FID):
    from common_stats import wilcoxon, paired
    mac, weak, strong, dim = C.r2_groups(Y, pred)
    row = dict(model=name, macro_r2=mac, weak12_r2=weak, strong5_r2=strong,
               **{'mae_' + k: v for k, v in C.mae_caliber_oof(Y, pred, fid).items()})
    if reference is not None:
        a, b = [], []
        for f in np.unique(fid):
            idx = fid == f
            a.append(C.macro_r2(Y[idx], pred[idx]).mean())
            b.append(C.macro_r2(Y[idx], reference[idx]).mean())
        row.update(p_dim=wilcoxon(dim, C.macro_r2(Y, reference)), **paired(a, b))
    return row

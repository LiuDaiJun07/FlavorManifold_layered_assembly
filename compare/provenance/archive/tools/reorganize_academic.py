"""One-time, content-preserving restructuring of the supplementary experiments."""
import hashlib
import json
import shutil
import zipfile
from pathlib import Path
from artifact_layout import BASE, DOMAINS, artifact_path

PROJECT=BASE.parent
GROUPS={'q1':'q1_prediction','q2':'q2_factor_analysis','q3':'q3_generation_transfer'}


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def checked(path):
    path=path.resolve()
    if not path.is_relative_to(BASE.resolve()):raise RuntimeError('Outside compare: '+str(path))
    return path


def move(source,target):
    source,target=checked(source),checked(target)
    if target.exists():raise RuntimeError('Destination exists: '+str(target))
    target.parent.mkdir(parents=True,exist_ok=True)
    source.replace(target)


def edit(path,pairs):
    path=checked(path);text=path.read_text(encoding='utf-8')
    for old,new in pairs:
        if old not in text:raise RuntimeError('Expected source text missing in '+str(path)+': '+old)
        text=text.replace(old,new)
    path.write_text(text,encoding='utf-8')


def main():
    archive=BASE/'provenance/archive';archive.mkdir(parents=True,exist_ok=True)
    state=json.loads((BASE/'run_status.json').read_text(encoding='utf-8'))
    if any(x['state']!='completed' for x in state.values()):raise RuntimeError('Experiments still running')
    shutil.copyfile(BASE/'run_status.json',archive/'run_status_before_reorganization.json')
    sources=list((BASE/'scripts').rglob('*.py'))+list(BASE.glob('*.py'))
    with zipfile.ZipFile(archive/'implementation_before_reorganization.zip','w',zipfile.ZIP_DEFLATED) as zipped:
        for path in sources:zipped.write(path,path.relative_to(BASE))
    planned=[]
    for area in DOMAINS:
        for source in (BASE/area).rglob('*'):
            if not source.is_file() or '__pycache__' in source.parts:continue
            relative=source.relative_to(BASE/area)
            variant=relative.parts[0] if len(relative.parts)>1 and relative.parts[0].startswith('smoke_') else ''
            planned.append((source,artifact_path(area,source.name,variant),sha(source)))
    destinations=[str(target).lower() for _,target,_ in planned]
    if len(destinations)!=len(set(destinations)):raise RuntimeError('Artifact routing collision')

    edit(BASE/'scripts/q1/protocol.py',[
      ("sys.path.insert(0, str(ROOT / 'src/q1'))","sys.path.insert(0, str(ROOT / 'src/q1'))\nsys.path.insert(0,str(ROOT/'compare/workflow'))\nfrom artifact_layout import ArtifactStore, table_path"),
      ("OUT = ROOT / ('compare/Q1' + TAG)","OUT = ArtifactStore('Q1',TAG.strip('/'))"),
      ("df.to_csv(path.with_name(path.stem + '_' + label + '.csv')","df.to_csv(table_path(path,label)"),
    ])
    for group in ['q2','q3']:
        edit(BASE/f'scripts/{group}/shared.py',[
          ("parents[1]/'q1'","parents[1]/'q1_prediction'"),
          (f"QOUT=ROOT/('compare/{group.upper()}'+TAG)",f"QOUT=ArtifactStore('{group.upper()}',TAG.strip('/'))")])
    edit(BASE/'scripts/q2/06_topk_predictive.py',[("compare/logs","compare/provenance/logs")])
    for q,domain in GROUPS.items():move(BASE/'scripts'/q,BASE/'methods'/domain)

    edit(BASE/'run_all.py',[
      ("ROOT=Path(__file__).resolve().parent.parent","ROOT=Path(__file__).resolve().parents[2]"),
      *[(f'scripts/{q}/',f'methods/{domain}/') for q,domain in GROUPS.items()],
      ("logs=compare/'logs';logs.mkdir(exist_ok=True)","logs=compare/'provenance/logs';logs.mkdir(parents=True,exist_ok=True)\n    (compare/'provenance/manifests').mkdir(parents=True,exist_ok=True)\n    (compare/'verification/reports').mkdir(parents=True,exist_ok=True)"),
      ("statuspath=compare/'run_status.json'","statuspath=compare/'provenance/execution/run_status.json'\n    statuspath.parent.mkdir(parents=True,exist_ok=True)"),
      ("compare/'mainline_sha256.json'","compare/'provenance/manifests/mainline_sha256.json'"),
      ("compare/'mainline_integrity.json'","compare/'verification/reports/mainline_integrity.json'")])
    move(BASE/'run_all.py',BASE/'workflow/run_experiments.py')
    (BASE/'run_all.py').write_text('"""Public entry point for supplementary experiments."""\nfrom pathlib import Path\nimport runpy\nif __name__ == "__main__":\n    runpy.run_path(str(Path(__file__).resolve().parent/"workflow/run_experiments.py"),run_name="__main__")\n',encoding='utf-8')

    edit(BASE/'summarize.py',[
      ("ROOT=Path(__file__).resolve().parent.parent","ROOT=Path(__file__).resolve().parents[2]"),
      ("compare/scripts/q1","compare/methods/q1_prediction"),
      ("from protocol import table","from protocol import table\nfrom artifact_layout import artifact_path,legacy_path"),
      ("compare=ROOT/'compare';rows=[]","compare=ROOT/'compare';rows=[]\n    (compare/'provenance/manifests').mkdir(parents=True,exist_ok=True)"),
      ("exists=(compare/file).exists()","exists=legacy_path(file).exists()"),
      ("(compare/required[key]).exists()","legacy_path(required[key]).exists()"),
      ("report=file)","report=legacy_path(file).relative_to(compare).as_posix())"),
      ("f'[{r[\"report\"]}]({r[\"report\"]})'","f'[{r[\"report\"]}]({Path(r[\"report\"]).relative_to(\"results\").as_posix()})'"),
      ("path=compare/'Q1'/file","path=artifact_path('Q1',file)"),
      ("compare/'Q1/ablation_compare.csv'","legacy_path('Q1/ablation_compare.csv')"),
      ("compare/'Q1/ablation_compare.md'","legacy_path('Q1/ablation_compare.md')"),
      ("(Q1/ablation_compare.csv)","(synthesis/tables/ablation_compare.csv)"),
      ("(Q1/ablation_compare.md)","(synthesis/reports/ablation_compare.md)"),
      ("path=compare/file","path=legacy_path(file)"),
      ("compare/'logs'","compare/'provenance/logs'"),
      ("if 'smoke_' in str(path):","if 'smoke' in path.parts:"),
      ("compare/'prediction_inventory.csv'","compare/'provenance/manifests/prediction_inventory.csv'"),
      ("compare/'mainline_sha256.json'","compare/'provenance/manifests/mainline_sha256.json'"),
      ("compare/'mainline_integrity.json'","compare/'verification/reports/mainline_integrity.json'"),
      ("compare/'RESULTS.md'","compare/'results/RESULTS.md'"),
      ("compare/'completion_inventory.csv'","compare/'results/experiment_registry.csv'"),
      ('详见feasibility.md与各报告','详见../methodology/feasibility.md与各报告'),
    ])
    move(BASE/'summarize.py',BASE/'workflow/summarize.py')
    edit(BASE/'verify_protocol.py',[("parent/'scripts/q1'","parents[1]/'methods/q1_prediction'"),
      ("parent/'scripts/q1/10_baselines.py'","parents[1]/'methods/q1_prediction/10_baselines.py'")])
    move(BASE/'verify_protocol.py',BASE/'verification/verify_protocol.py')
    edit(BASE/'verify_optional_inputs.py',[("ROOT=Path(__file__).resolve().parent.parent","ROOT=Path(__file__).resolve().parents[2]"),
      ('compare/scripts/q2','compare/methods/q2_factor_analysis'),
      ('compare/optional_input_verification.json','compare/verification/reports/optional_input_verification.json')])
    move(BASE/'verify_optional_inputs.py',BASE/'verification/verify_optional_inputs.py')

    for source,target,_ in planned:move(source,target)
    roots={
      'feasibility.md':'methodology/feasibility.md','environment.json':'provenance/environment.json',
      'mainline_sha256.json':'provenance/manifests/mainline_sha256.json',
      'mainline_integrity.json':'verification/reports/mainline_integrity.json',
      'delivery_verification.json':'verification/reports/delivery_verification.json',
      'final_numerical_verification.json':'verification/reports/final_numerical_verification.json',
      'optional_input_verification.json':'verification/reports/optional_input_verification.json',
      'layout_migration.json':'provenance/archive/layout_migration.json',
      'code_sha256.json':'provenance/archive/code_sha256_before_reorganization.json',
      'completion_inventory.csv':'provenance/archive/completion_inventory_before_reorganization.csv',
      'prediction_inventory.csv':'provenance/archive/prediction_inventory_before_reorganization.csv',
      'SESSION_STATE.md':'provenance/archive/SESSION_STATE.md',
      'migrate_layout.py':'provenance/archive/tools/migrate_case_layout.py',
      'RESULTS.md':'results/RESULTS.md',
    }
    for source,target in roots.items():move(BASE/source,BASE/target)
    move(BASE/'logs',BASE/'provenance/logs')
    for area in DOMAINS:
        folder=checked(BASE/area)
        remaining=[p for p in folder.rglob('*') if p.is_file()]
        if remaining:raise RuntimeError('Unmoved files: '+str(remaining))
        # All targets are verified inside compare; remove only empty old directories.
        for directory in sorted(folder.rglob('*'),key=lambda x:len(x.parts),reverse=True):
            if directory.is_dir():checked(directory).rmdir()
        folder.rmdir()
    old_scripts=checked(BASE/'scripts')
    for cache in list(old_scripts.rglob('__pycache__')):
        shutil.rmtree(checked(cache))
    for directory in sorted(old_scripts.rglob('*'),key=lambda x:len(x.parts),reverse=True):
        if directory.is_dir():checked(directory).rmdir()
    old_scripts.rmdir()
    for cache in [BASE/'__pycache__']:
        if cache.exists():shutil.rmtree(checked(cache))

    index=[]
    for source,target,original_hash in planned:
        if sha(target)!=original_hash:raise RuntimeError('Artifact content changed: '+str(target))
        index.append(dict(previous_path=str(source.relative_to(BASE)),path=str(target.relative_to(BASE)),sha256=original_hash))
    manifest=BASE/'provenance/manifests/artifact_reorganization.json'
    manifest.parent.mkdir(parents=True,exist_ok=True)
    manifest.write_text(json.dumps(dict(files=index,all_contents_preserved=True),ensure_ascii=False,indent=2),encoding='utf-8')
    for item in state.values():
        previous=item['script']
        for q,domain in GROUPS.items():item['script']=item['script'].replace('scripts/'+q+'/', 'methods/'+domain+'/')
        item['before_academic_reorganization']=dict(script=previous,sha256=item['script_sha256'])
        item['script_sha256']=sha(BASE/item['script'])
        item['refactor_note']='Paths and artifact taxonomy only; model calculations unchanged.'
    status=BASE/'provenance/execution/run_status.json';status.parent.mkdir(parents=True,exist_ok=True)
    status.write_text(json.dumps(state,ensure_ascii=False,indent=2),encoding='utf-8')
    (BASE/'run_status.json').unlink()
    print(json.dumps(dict(moved_artifacts=len(index),all_contents_preserved=True),ensure_ascii=False))


if __name__=='__main__':main()

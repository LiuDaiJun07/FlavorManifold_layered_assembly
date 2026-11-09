"""One-time Windows case-insensitive layout adaptation; run after all jobs finish."""
import json
import hashlib
import shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent
COMPARE=ROOT/'compare'


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def contained(path):
    resolved=path.resolve()
    if not resolved.is_relative_to(COMPARE.resolve()):raise RuntimeError('Path leaves compare: '+str(path))
    return resolved


def main():
    state=json.loads((COMPARE/'run_status.json').read_text(encoding='utf-8'))
    active=[k for k,v in state.items() if v.get('state')!='completed']
    if active:raise RuntimeError('Finish experiment jobs before layout migration: '+str(active))
    if (COMPARE/'layout_migration.json').exists():
        print('Layout already migrated');return
    before={}
    for q in ['q1','q2','q3']:
        source=contained(COMPARE/q)
        for p in source.rglob('*'):
            if p.is_file() and p.suffix!='.py' and '__pycache__' not in p.parts:
                before[(q.upper(),str(p.relative_to(source)))]=digest(p)
    # Only path-depth adjustments; no model or statistical parameters change.
    replacements={
        COMPARE/'q1/protocol.py':[("parents[2]","parents[3]")],
        COMPARE/'q1/geometry_corrected.py':[("parents[2]","parents[3]")],
        COMPARE/'verify_protocol.py':[("parent/'q1'","parent/'scripts/q1'"),("parent/'q1/10_baselines.py'","parent/'scripts/q1/10_baselines.py'")],
        COMPARE/'verify_optional_inputs.py':[("compare/q2","compare/scripts/q2")],
        COMPARE/'summarize.py':[("compare/q1","compare/scripts/q1")],
        COMPARE/'run_all.py':[("'q1/","'scripts/q1/"),("'q2/","'scripts/q2/"),("'q3/","'scripts/q3/")],
    }
    for p,pairs in replacements.items():
        p=contained(p);text=p.read_text(encoding='utf-8')
        for a,b in pairs:text=text.replace(a,b)
        p.write_text(text,encoding='utf-8')
    for q in ['q1','q2','q3']:
        source=contained(COMPARE/q);destination=contained(COMPARE/'scripts'/q)
        destination.mkdir(parents=True,exist_ok=True)
        for p in source.glob('*.py'):
            target=contained(destination/p.name)
            if target.exists():raise RuntimeError('Destination already exists: '+str(target))
            p.replace(target)
        cache=source/'__pycache__'
        if cache.exists():
            cache=contained(cache)
            assert cache.name=='__pycache__'
            shutil.rmtree(cache)
        # A checked two-step rename works on case-insensitive NTFS as well as Linux.
        interim=contained(COMPARE/('_case_'+q))
        if interim.exists():raise RuntimeError('Temporary rename path exists')
        source.rename(interim)
        interim.rename(contained(COMPARE/q.upper()))
    for (q,relative),sha in before.items():
        if digest(COMPARE/q/relative)!=sha:raise RuntimeError('Output content changed: '+q+'/'+relative)
    for item in state.values():
        old=item.get('script','')
        if old.startswith(('q1/','q2/','q3/')):
            item['executed_path_before_layout_migration']=old
            item['script']='scripts/'+old
    (COMPARE/'run_status.json').write_text(json.dumps(state,ensure_ascii=False,indent=2),encoding='utf-8')
    readme=COMPARE/'README.md'
    readme.write_text(readme.read_text(encoding='utf-8').replace('位于q1/common_stats.py','位于scripts/q1/common_stats.py')+
        '\nWindows大小写适配：脚本在scripts/q1、scripts/q2、scripts/q3；产物在Q1、Q2、Q3，避免q1/Q1目录合并。运行入口仍是compare/run_all.py。\n',encoding='utf-8')
    marker=dict(migrated=True,outputs_unchanged=True,verified_output_files=len(before),
                scripts='compare/scripts/q1,q2,q3',outputs='compare/Q1,Q2,Q3',
                source_path_mapping={f'compare/{q}/':f'compare/scripts/{q}/' for q in ['q1','q2','q3']})
    (COMPARE/'layout_migration.json').write_text(json.dumps(marker,indent=2),encoding='utf-8')
    print(marker)


if __name__=='__main__':main()

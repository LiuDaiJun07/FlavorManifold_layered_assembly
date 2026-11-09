"""Validate artifact relocation, runnable paths, and original project integrity."""
import ast
import hashlib
import json
import re
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
ROOT = BASE.parent
sys.path.insert(0, str(BASE / 'workflow'))
from artifact_layout import DOMAINS, artifact_path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    manifest = json.loads((BASE/'provenance/manifests/artifact_reorganization.json').read_text(encoding='utf-8'))
    changed = [item['path'] for item in manifest['files']
               if not (BASE/item['path']).is_file() or sha(BASE/item['path']) != item['sha256']]
    state = json.loads((BASE/'provenance/execution/run_status.json').read_text(encoding='utf-8'))
    state_ok = all(item['state'] == 'completed' and
                   sha(BASE/item['script']) == item['script_sha256'] for item in state.values())
    baseline = json.loads((BASE/'provenance/manifests/mainline_sha256.json').read_text(encoding='utf-8'))
    mainline_changed = [name for name, digest in baseline.items()
                        if not (ROOT/name).is_file() or sha(ROOT/name) != digest]
    syntax_count = 0
    for folder in ['methods', 'workflow', 'verification']:
        for path in (BASE/folder).rglob('*.py'):
            ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
            syntax_count += 1
    broken = []
    for path in [BASE/'README.md', BASE/'RESULTS.md', *(BASE/'results').rglob('*.md')]:
        for target in re.findall(r'\]\(([^)]+)\)', path.read_text(encoding='utf-8')):
            if '://' in target or target.startswith('#'):
                continue
            if not (path.parent/target.split('#')[0]).exists():
                broken.append(dict(file=str(path.relative_to(BASE)), target=target))
    sys.path.insert(0, str(BASE/'methods/q1_prediction'))
    import protocol as P
    import numpy as np
    cached = artifact_path('Q1', '10_B0.npz', create=False)
    assert cached.exists(), cached
    digest = sha(cached)
    predictions = P.cv('10_B0')
    cached_ok = bool(np.isfinite(predictions).all()) and sha(cached) == digest
    environment = json.loads((BASE/'provenance/environment.json').read_text(encoding='utf-8'))
    input_changed = [name for name, expected in environment['inputs_sha256'].items()
                     if sha(Path(P.C.ATT)/name) != expected]
    checks = dict(artifact_contents_preserved=not changed, completed_execution_paths_valid=state_ok,
                  original_src_output_unchanged=not mainline_changed, original_inputs_unchanged=not input_changed,
                  cached_prediction_reused=cached_ok, markdown_links_valid=not broken,
                  old_flat_directories_absent=all(not (BASE/area).exists() for area in DOMAINS))
    result = dict(checks=checks, all_passed=all(checks.values()), migrated_artifacts=len(manifest['files']),
                  parsed_python_files=syntax_count, changed_artifacts=changed, broken_links=broken,
                  changed_mainline=mainline_changed, changed_inputs=input_changed)
    (BASE/'verification/reports/academic_layout_verification.json').write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))
    assert result['all_passed'], result


if __name__ == '__main__':
    main()

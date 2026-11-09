"""Synthetic unit fixture only; never contributes experimental results."""
import sys
import json
import tempfile
import importlib.util
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'compare/methods/q2_factor_analysis'))
spec=importlib.util.spec_from_file_location('oav_input_check',ROOT/'compare/methods/q2_factor_analysis/09_oav_sensitivity.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
import pandas as pd
import numpy as np


def main():
    checks={}
    with tempfile.TemporaryDirectory(prefix='unit_fixture_',dir=ROOT/'compare') as folder:
        temporary=Path(folder).resolve()
        # Verify scope before TemporaryDirectory's recursive cleanup.
        assert temporary.is_relative_to((ROOT/'compare').resolve())
        module.QOUT=temporary/'units';module.QOUT.mkdir()
        module.CHEM=pd.DataFrame({'fixture_compound':[1.,2.,3.]})
        data=[]
        for source,threshold,unit in [('fixture_A',1.,'mg/L'),('fixture_B',1000.,'μg/L')]:
            data.append(dict(compound='fixture_compound',source_id=source,source_url='https://example.invalid/unit-fixture',
                threshold=threshold,threshold_unit=unit,matrix='fixture_beer',concentration_unit='mg/L',
                concentration_scale='raw',independence_group=source,conversion_verified=True))
        pd.DataFrame(data).to_csv(module.QOUT/'threshold_sources.csv',index=False)
        module.main()
        result=pd.read_csv(module.QOUT/'09_oav_sensitivity_oav.csv')
        pairs=pd.read_csv(module.QOUT/'09_oav_sensitivity_pairs.csv')
        checks['mg_ug_conversion']=bool(np.allclose(result.oav,[2.,2.]))
        checks['equal_sources_jaccard']=bool(pairs.jaccard.iloc[0]==1)
        module.QOUT=temporary/'empty';module.QOUT.mkdir();module.main()
        checks['empty_sources_no_numeric_oav']=not (module.QOUT/'09_oav_sensitivity.npz').exists()
    out=ROOT/'compare/verification/reports/optional_input_verification.json'
    out.write_text(json.dumps(dict(checks=checks,all_passed=all(checks.values())),indent=2),encoding='utf-8')
    assert all(checks.values())
    print(out)


if __name__=='__main__':main()

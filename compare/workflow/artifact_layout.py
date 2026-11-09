"""Academic artifact taxonomy shared by experiment code and migration utilities."""
from pathlib import Path
import re

BASE = Path(__file__).resolve().parents[1]
DOMAINS = {'Q1':'q1_prediction', 'Q2':'q2_factor_analysis', 'Q3':'q3_generation_transfer'}
EXPERIMENTS = {
 'Q1': {'10':'C01_external_baselines','11':'C02_repeated_validation','12':'C03_geometric_ablation',
        '13':'C04_ensemble_fusion','14':'C05_partition_sensitivity','15':'C06_missingness_robustness',
        '16':'C07_hyperparameter_sensitivity','17':'C08_rating_validation','18':'A01_reproducibility_audit'},
 'Q2': {'05':'C09_confounding_adjustment','06':'C10_predictive_sufficiency','07':'C11_method_consensus',
        '08':'C12_network_sensitivity','09':'C14_threshold_sensitivity','10':'C13_selection_stability'},
 'Q3': {'04':'C15_C16_generative_validation','05':'C17_transfer_benchmarks','06':'C18_guard_tradeoffs',
        '07':'C19_readout_robustness','08':'C20_geodesic_controls','09':'C21_latent_sensitivity'},
}


def artifact_path(area, filename, variant='', create=True):
    area=area.upper();name=Path(filename).name
    if name != str(filename):raise ValueError('Expected one artifact filename: '+str(filename))
    domain=DOMAINS[area]
    if name=='threshold_source_candidates.md':
        path=BASE/'methodology/references/threshold_source_candidates.md'
    elif name=='threshold_sources.csv':
        path=BASE/'data/reference/olfactory_thresholds'/name
    elif name=='17_card_mapping.csv':
        path=BASE/'data/reference/sensory_rating_forms'/name
    elif name=='folds.npz':
        path=BASE/'data/protocol/cross_validation'/name
    elif name=='protocol_verification.json' or name=='parallel_backend_verification.json':
        path=BASE/'verification/reports'/name
    elif name=='refresh_manifest.json':
        path=BASE/'provenance/archive'/name
    elif name=='joblib_temp':
        path=BASE/'cache/runtime'/domain
    elif name.startswith('ablation_compare'):
        kind='reports' if Path(name).suffix=='.md' else 'tables'
        path=BASE/'results/synthesis'/kind/name
    else:
        match=re.match(r'(?:fig)?(\d{2})',name)
        if not match and area=='Q1' and re.match(r'(?:E0|E1|BASE5)(?:_fold\d+)?\.npz$',name):
            match=re.match(r'(\d{2})','10')
        if not match:
            if name.startswith('verify_'):
                path=BASE/'cache'/domain/'protocol_smoke'/name
            else:raise ValueError('Unclassified artifact: '+area+'/'+name)
        else:
            number=match[1]
            experiment=EXPERIMENTS[area][number]
            suffix=Path(name).suffix
            if suffix=='.md':path=BASE/'results'/domain/experiment/'reports'/name
            elif suffix=='.png':path=BASE/'results'/domain/experiment/'figures'/name
            elif suffix=='.csv' and name.startswith('17_card_'):
                path=BASE/'data'/domain/experiment/'extracted_rating_forms'/name
            elif suffix=='.csv':path=BASE/'results'/domain/experiment/'tables'/name
            elif suffix=='.json':path=BASE/'data'/domain/experiment/'model_selection'/name
            elif suffix=='.npz':
                if re.search(r'_fold\d+\.npz$',name):
                    config=re.sub(r'_fold\d+\.npz$','',name)
                    path=BASE/'cache'/domain/experiment/'cross_validation'/config/name
                elif '_geom_' in name:
                    config=re.sub(r'_f\d+_','_',Path(name).stem)
                    path=BASE/'cache'/domain/experiment/'embeddings'/config/name
                elif 'inner_members' in name:path=BASE/'cache'/domain/experiment/'nested_validation'/name
                elif 'selection_f' in name:path=BASE/'cache'/domain/experiment/'feature_selection'/name
                elif 'boot_target' in name:path=BASE/'cache'/domain/experiment/'bootstrap_replicates'/name
                elif 'chain_target' in name:path=BASE/'cache'/domain/experiment/'stability_selection'/name
                elif 'perm_fold' in name:path=BASE/'cache'/domain/experiment/'permutation_importance'/name
                elif name=='06_random_triples.npz' and area=='Q2':
                    path=BASE/'data'/domain/experiment/'null_distributions'/name
                else:
                    if area=='Q1':kind='predictions'
                    elif area=='Q2':kind='predictions' if number=='06' else 'statistics'
                    else:kind={'04':'generated_samples','05':'transfer_samples','06':'transfer_samples',
                               '07':'sensory_predictions','08':'interpolated_samples','09':'generated_samples'}[number]
                    path=BASE/'data'/domain/experiment/kind/name
            else:raise ValueError('Unclassified artifact extension: '+name)
    if variant:
        path=BASE/'verification/smoke'/variant.removeprefix('smoke_')/path.relative_to(BASE)
    if create:path.parent.mkdir(parents=True,exist_ok=True)
    return path


def legacy_path(relative, create=True):
    parts=Path(relative).parts
    if len(parts)!=2 or parts[0].upper() not in DOMAINS:raise ValueError(relative)
    return artifact_path(parts[0],parts[1],create=create)


class ArtifactStore:
    """Filename-based routing keeps existing scientific computations unchanged."""
    def __init__(self, area, variant=''):
        self.area=area.upper();self.variant=variant
    def __truediv__(self, name):return artifact_path(self.area,str(name),self.variant)
    def __fspath__(self):return str(BASE/'results'/DOMAINS[self.area])
    def __str__(self):return self.__fspath__()
    def mkdir(self,parents=True,exist_ok=True):Path(self).mkdir(parents=parents,exist_ok=exist_ok)


def table_path(report, label):
    report=Path(report)
    if report.parent.name=='reports' and report.is_relative_to(BASE):
        path=report.parent.parent/'tables'/(report.stem+'_'+label+'.csv')
        path.parent.mkdir(parents=True,exist_ok=True)
        return path
    return report.with_name(report.stem+'_'+label+'.csv')

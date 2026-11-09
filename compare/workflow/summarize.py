"""Build reviewable completion inventory from real on-disk reports and predictions."""
import json
import sys
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'compare/methods/q1_prediction'))
from protocol import table
from artifact_layout import artifact_path,legacy_path

CATALOG=[
('C1','Q1/10_baselines.md','外部基线与嵌套调参'),('C2','Q1/11_repeated_cv.md','重复CV、种子与CD图'),
('C3','Q1/12_hyper_controls.md','几何/曲率/维数/噪声对照'),('C4','Q1/13_fusion_controls.md','融合与成员消融'),
('C5','Q1/14_fold_sensitivity.md','折敏感性'),('C6','Q1/15_missing_stress.md','缺失压力'),
('C7','Q1/16_hyperparam_sensitivity.md','36组超参'),('C8','Q1/17_grade_external.md','z空间正式化；外部评分不足'),
('C9','Q2/05_confound.md','混杂控制'),('C10','Q2/06_topk_predictive.md','嵌套综合Top3与1000随机子集'),
('C11','Q2/07_method_matrix.md','六链矩阵，小森林PI代理'),('C12','Q2/08_graph_sensitivity.md','图扩散敏感性'),
('C13','Q2/10_bootstrap_upgrade.md','条件频率BCa与系数BCa；原数据频率BCa未做'),('C14','Q2/09_oav_sensitivity.md','单位与来源前提不足'),
('C15','Q3/04_gen_baselines.md','生成基线'),('C16','Q3/04_gen_baselines.md','小样本LOO与分层合并'),
('C17','Q3/05_transfer_baselines.md','迁移基线；无配对T1调整'),('C18','Q3/06_guard_cost.md','守护代价'),
('C19','Q3/07_readout_robust.md','读出形式稳健性'),('C20','Q3/08_path_controls.md','路径与同口径应力'),
('C21','Q3/09_latent_sensitivity.md','q/κ网格')]


def main():
    compare=ROOT/'compare';rows=[]
    (compare/'provenance/manifests').mkdir(parents=True,exist_ok=True)
    for key,file,scope in CATALOG:
        exists=legacy_path(file).exists()
        required={'C1':'Q1/10_B3_med_bounded.npz','C2':'Q1/fig11_cd.png',
                  'C3':'Q1/12_hyper_controls_buckets.csv','C4':'Q1/13_inner_select.npz'}
        if key in required:exists=exists and legacy_path(required[key]).exists()
        state='报告已生成' if exists else '未完成'
        if exists and key in ['C8','C14']:state='部分完成/输入前提不足'
        if exists and key=='C13':state='部分完成/频率原数据BCa未做'
        rows.append(dict(experiment=key,state=state,scope=scope,report=legacy_path(file).relative_to(compare).as_posix()))
    display=pd.DataFrame(rows).copy();display['report']=[f'[{r["report"]}]({Path(r["report"]).relative_to("results").as_posix()})' for r in rows]
    report=['# 对照实验实施与结果总览',
        '主线src与Output只读。表中“报告已生成”表示实际产物存在；科学口径调整与输入前提不足另行明示。',table(display)]
    ablations=[]
    original=pd.read_csv(ROOT/'Output/Q1/07_ablation.csv').rename(columns={'config':'model','weak_r2':'weak12_r2','strong_r2':'strong5_r2'})
    original['experiment']='original';original['p_reference']='';original['valid_for_predictive_comparison']=True
    ablations.append(original[['experiment','model','macro_r2','weak12_r2','strong5_r2','p_reference','valid_for_predictive_comparison']])
    for key,file,reference in [('C1','10_baselines_metrics.csv','BASE5'),('C3','12_hyper_controls_metrics.csv','E1'),
        ('C4','13_fusion_controls_metrics.csv','BASE5'),('C6','15_missing_stress_metrics.csv','E1'),('C7','16_hyperparam_sensitivity_metrics.csv','E1')]:
        path=artifact_path('Q1',file)
        if not path.exists():continue
        df=pd.read_csv(path);df['experiment']=key;df['p_reference']=reference
        df['valid_for_predictive_comparison']=df.model!='leaky_select'
        df['role']='sensitivity_scan' if key=='C7' else 'fixed_control'
        df.loc[df.model.str.contains('bounded'),'role']='posthoc_extrapolation_diagnostic'
        df.loc[df.model=='leaky_select','role']='leakage_demonstration_only'
        ablations.append(df[['experiment','model','macro_r2','weak12_r2','strong5_r2','p_reference','p_dim','valid_for_predictive_comparison','role']])
    combined=pd.concat(ablations,ignore_index=True)
    combined.to_csv(legacy_path('Q1/ablation_compare.csv'),index=False,encoding='utf-8-sig')
    (legacy_path('Q1/ablation_compare.md')).write_text('# 原主线与对照消融总表\n\n仅汇总相同固定十折的点估计；leaky_select不可作预测性能。扫描最高点不是选择后的无偏性能。p_reference列说明各行的逐维配对参照，p值仅探索性。\n\n'+table(combined),encoding='utf-8')
    report.append('统一十折消融表：[CSV](synthesis/tables/ablation_compare.csv) / [Markdown](synthesis/reports/ablation_compare.md)。')
    for title,file,columns in [
        ('Q1 外部基线','Q1/10_baselines_metrics.csv',['model','macro_r2','weak12_r2','strong5_r2','p_dim']),
        ('Q1 重复CV','Q1/11_repeated_cv_statistics.csv',None),
        ('Q1 双曲','Q1/12_hyper_controls_metrics.csv',['model','macro_r2','p_dim']),
        ('Q1 折划分','Q1/14_fold_sensitivity_metrics.csv',['scheme','model','macro_r2']),
        ('Q1 融合','Q1/13_fusion_controls_metrics.csv',['model','macro_r2','p_dim']),
        ('Q1 缺失压力','Q1/15_missing_stress_metrics.csv',['model','macro_r2']),
        ('Q2 混杂','Q2/05_confound_summary.csv',None),
        ('Q2 Top3预测','Q2/06_topk_predictive_metrics.csv',['model','macro_r2','weak12_r2','strong5_r2']),
        ('Q2 六链一致性','Q2/07_method_matrix_jaccard.csv',None),
        ('Q3 生成','Q3/04_gen_baselines_metrics.csv',['method','mmd2','energy','ks_dim_median','nearest_ratio','fake_classifier_accuracy']),
        ('Q3 迁移','Q3/05_transfer_baselines_metrics.csv',['method','state','rmse_portrait','ks_nondef','mmd2','defect_increases']),
        ('Q3 同口径应力','Q3/08_path_controls_common_stress.csv',None)]:
        path=legacy_path(file)
        if path.exists():
            df=pd.read_csv(path);report.extend(['## '+title,table(df[columns] if columns else df)])
    report+=['## 尚未满足原设计的部分',
        'C8：3.xlsx是12张空白评分模板，缺真实独立评分，无法计算外部Kappa。C14：缺经过核验的浓度单位与两套独立阈值，未伪造OAV结果。C13：已有1000次各比例子采样、条件频率BCa/Wilson蒙特卡洛区间和系数BCa；选择频率的原数据BCa需要嵌套重抽样，未声称完成。',
        'C17的T1因无样本配对，改为位置尺度仿射。C11用设计允许的小森林PI代理，不能称为BASE5的重要性。全部p值仅探索性；不显著不证明等效。详见../methodology/feasibility.md与各报告。']
    warning_count=sum(path.read_text(encoding='utf-8',errors='replace').count('ConvergenceWarning')
                      for path in (compare/'provenance/logs').glob('C10*.log'))
    if warning_count:report+=['## 数值限制',f'C10保存的运行日志有{warning_count}次Lasso候选/筛选收敛警告；预测缓存均有限，但并不表示全部候选优化器已精确收敛。详见C10报告及日志，相关排序只作探索性结果。']
    inventory=[];invalid=[]
    for path in compare.rglob('*.npz'):
        if 'smoke' in path.parts:continue
        with np.load(path,allow_pickle=False) as d:
            if 'pred' in d and d['pred'].size:
                finite=bool(np.isfinite(d['pred']).all())
                inventory.append(dict(file=str(path.relative_to(compare)),shape=str(d['pred'].shape),finite=finite))
                if not finite:invalid.append(str(path))
    pd.DataFrame(inventory).to_csv(compare/'provenance/manifests/prediction_inventory.csv',index=False,encoding='utf-8-sig')
    digest=compare/'provenance/manifests/mainline_sha256.json'
    if digest.exists():
        old=json.loads(digest.read_text(encoding='utf-8'));changed=[]
        for file,sha in old.items():
            path=ROOT/file
            if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest()!=sha:changed.append(file)
        integrity=dict(unchanged=not changed,changed=changed)
        (compare/'verification/reports/mainline_integrity.json').write_text(json.dumps(integrity,indent=2),encoding='utf-8')
        report+=['## 验证',f'主线SHA256内容核验：{integrity}。预测缓存有限性核验：{len(inventory)}份，异常{len(invalid)}份。']
    (compare/'results/RESULTS.md').write_text('\n\n'.join(report),encoding='utf-8')
    pd.DataFrame(rows).to_csv(compare/'results/experiment_registry.csv',index=False,encoding='utf-8-sig')
    assert not invalid
    print(compare/'results/RESULTS.md')


if __name__=='__main__':main()

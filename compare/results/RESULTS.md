# 对照实验实施与结果总览

主线src与Output只读。表中“报告已生成”表示实际产物存在；科学口径调整与输入前提不足另行明示。

| experiment | state | scope | report |
| --- | --- | --- | --- |
| C1 | 报告已生成 | 外部基线与嵌套调参 | [results/q1_prediction/C01_external_baselines/reports/10_baselines.md](q1_prediction/C01_external_baselines/reports/10_baselines.md) |
| C2 | 报告已生成 | 重复CV、种子与CD图 | [results/q1_prediction/C02_repeated_validation/reports/11_repeated_cv.md](q1_prediction/C02_repeated_validation/reports/11_repeated_cv.md) |
| C3 | 报告已生成 | 几何/曲率/维数/噪声对照 | [results/q1_prediction/C03_geometric_ablation/reports/12_hyper_controls.md](q1_prediction/C03_geometric_ablation/reports/12_hyper_controls.md) |
| C4 | 报告已生成 | 融合与成员消融 | [results/q1_prediction/C04_ensemble_fusion/reports/13_fusion_controls.md](q1_prediction/C04_ensemble_fusion/reports/13_fusion_controls.md) |
| C5 | 报告已生成 | 折敏感性 | [results/q1_prediction/C05_partition_sensitivity/reports/14_fold_sensitivity.md](q1_prediction/C05_partition_sensitivity/reports/14_fold_sensitivity.md) |
| C6 | 报告已生成 | 缺失压力 | [results/q1_prediction/C06_missingness_robustness/reports/15_missing_stress.md](q1_prediction/C06_missingness_robustness/reports/15_missing_stress.md) |
| C7 | 报告已生成 | 36组超参 | [results/q1_prediction/C07_hyperparameter_sensitivity/reports/16_hyperparam_sensitivity.md](q1_prediction/C07_hyperparameter_sensitivity/reports/16_hyperparam_sensitivity.md) |
| C8 | 部分完成/输入前提不足 | z空间正式化；外部评分不足 | [results/q1_prediction/C08_rating_validation/reports/17_grade_external.md](q1_prediction/C08_rating_validation/reports/17_grade_external.md) |
| C9 | 报告已生成 | 混杂控制 | [results/q2_factor_analysis/C09_confounding_adjustment/reports/05_confound.md](q2_factor_analysis/C09_confounding_adjustment/reports/05_confound.md) |
| C10 | 报告已生成 | 嵌套综合Top3与1000随机子集 | [results/q2_factor_analysis/C10_predictive_sufficiency/reports/06_topk_predictive.md](q2_factor_analysis/C10_predictive_sufficiency/reports/06_topk_predictive.md) |
| C11 | 报告已生成 | 六链矩阵，小森林PI代理 | [results/q2_factor_analysis/C11_method_consensus/reports/07_method_matrix.md](q2_factor_analysis/C11_method_consensus/reports/07_method_matrix.md) |
| C12 | 报告已生成 | 图扩散敏感性 | [results/q2_factor_analysis/C12_network_sensitivity/reports/08_graph_sensitivity.md](q2_factor_analysis/C12_network_sensitivity/reports/08_graph_sensitivity.md) |
| C13 | 部分完成/频率原数据BCa未做 | 条件频率BCa与系数BCa；原数据频率BCa未做 | [results/q2_factor_analysis/C13_selection_stability/reports/10_bootstrap_upgrade.md](q2_factor_analysis/C13_selection_stability/reports/10_bootstrap_upgrade.md) |
| C14 | 部分完成/输入前提不足 | 单位与来源前提不足 | [results/q2_factor_analysis/C14_threshold_sensitivity/reports/09_oav_sensitivity.md](q2_factor_analysis/C14_threshold_sensitivity/reports/09_oav_sensitivity.md) |
| C15 | 报告已生成 | 生成基线 | [results/q3_generation_transfer/C15_C16_generative_validation/reports/04_gen_baselines.md](q3_generation_transfer/C15_C16_generative_validation/reports/04_gen_baselines.md) |
| C16 | 报告已生成 | 小样本LOO与分层合并 | [results/q3_generation_transfer/C15_C16_generative_validation/reports/04_gen_baselines.md](q3_generation_transfer/C15_C16_generative_validation/reports/04_gen_baselines.md) |
| C17 | 报告已生成 | 迁移基线；无配对T1调整 | [results/q3_generation_transfer/C17_transfer_benchmarks/reports/05_transfer_baselines.md](q3_generation_transfer/C17_transfer_benchmarks/reports/05_transfer_baselines.md) |
| C18 | 报告已生成 | 守护代价 | [results/q3_generation_transfer/C18_guard_tradeoffs/reports/06_guard_cost.md](q3_generation_transfer/C18_guard_tradeoffs/reports/06_guard_cost.md) |
| C19 | 报告已生成 | 读出形式稳健性 | [results/q3_generation_transfer/C19_readout_robustness/reports/07_readout_robust.md](q3_generation_transfer/C19_readout_robustness/reports/07_readout_robust.md) |
| C20 | 报告已生成 | 路径与同口径应力 | [results/q3_generation_transfer/C20_geodesic_controls/reports/08_path_controls.md](q3_generation_transfer/C20_geodesic_controls/reports/08_path_controls.md) |
| C21 | 报告已生成 | q/κ网格 | [results/q3_generation_transfer/C21_latent_sensitivity/reports/09_latent_sensitivity.md](q3_generation_transfer/C21_latent_sensitivity/reports/09_latent_sensitivity.md) |

统一十折消融表：[CSV](synthesis/tables/ablation_compare.csv) / [Markdown](synthesis/reports/ablation_compare.md)。

## Q1 外部基线

| model | macro_r2 | weak12_r2 | strong5_r2 | p_dim |
| --- | --- | --- | --- | --- |
| E0 | 0.248227 | 0.0502978 | 0.649975 | 4.52151e-06 |
| E1 | 0.254999 | 0.0548745 | 0.668137 | 0.0990859 |
| BASE5 | 0.262041 | 0.0766585 | 0.669131 | 1 |
| B0 | -0.00694863 | -0.00773997 | -0.00572112 | 1.77636e-15 |
| B1 | 0.15235 | -0.022785 | 0.537261 | 2.4869e-14 |
| B2 | -3.28367 | -0.850804 | -7.25205 | 1.77636e-14 |
| B3 | -3.64171 | -0.0310131 | -36.9932 | 4.49418e-13 |
| B4 | -0.346557 | -0.141197 | -1.44982 | 1.77636e-15 |
| B5_5 | 0.0984633 | -0.0582851 | 0.531018 | 1.77636e-15 |
| B5_15 | 0.150911 | 0.0150982 | 0.490459 | 1.77636e-15 |
| B5_25 | 0.136594 | 0.0215906 | 0.410509 | 1.77636e-15 |
| B6 | 0.190188 | 0.00117663 | 0.657173 | 1.28717e-10 |
| B7 | 0.253471 | 0.0780828 | 0.693056 | 0.19096 |
| B2_med | -3.00967 | -0.717515 | -6.45791 | 3.37508e-14 |
| B3_med | -2.77317 | 0.0308042 | -27.6615 | 2.02322e-10 |
| B2_bounded | 0.160384 | 0.00812028 | 0.466341 | 1.5449e-11 |
| B3_bounded | 0.186405 | 0.0049942 | 0.584246 | 4.21885e-12 |
| B4_bounded | 0.0849939 | -0.0272926 | 0.413854 | 1.24345e-14 |
| B2_med_bounded | 0.174687 | 0.0406043 | 0.498759 | 3.13456e-10 |
| B3_med_bounded | 0.19138 | 0.0313015 | 0.586075 | 4.79229e-10 |

## Q1 重复CV

| delta_mean | delta_std | wilcoxon_p | t_p | p_dim_mean_repeat | seed_range | E1_mean | E1_std | BASE5_mean | BASE5_std | E1_pooled_repeat_mean | E1_pooled_repeat_std | BASE5_pooled_repeat_mean | BASE5_pooled_repeat_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.00958541 | 0.0120254 | 0.00671387 | 0.00803498 | 0.00235681 | 0.0012999 | 0.237989 | 0.0292587 | 0.247575 | 0.0279494 | 0.253416 | 0.00569784 | 0.26288 | 0.00153009 |

## Q1 双曲

| model | macro_r2 | p_dim |
| --- | --- | --- |
| E1 | 0.254999 | 1 |
| euclidean | 0.215025 | 8.8427e-12 |
| c1e-08 | 0.213709 | 2.9944e-11 |
| c0.1 | 0.214813 | 1.17275e-11 |
| c1 | 0.212177 | 1.13687e-12 |
| c10 | 0.241771 | 3.38221e-09 |
| corrected_c1e-08 | 0.212996 | 1.56319e-13 |
| corrected_c0.1 | 0.213967 | 2.63434e-12 |
| corrected_c1 | 0.212177 | 1.13687e-12 |
| corrected_c10 | 0.192069 | 3.00204e-13 |
| d2_dist | 0.21801 | 4.91163e-12 |
| d2_typ | 0.255767 | 0.572086 |
| d2_tan | 0.249771 | 8.23756e-05 |
| d4_dist | 0.215025 | 5.86198e-14 |
| d4_typ | 0.254755 | 0.901018 |
| d4_tan | 0.249067 | 5.63962e-05 |
| d8_dist | 0.208592 | 1.89893e-12 |
| d8_typ | 0.257018 | 0.127993 |
| d8_tan | 0.246808 | 7.23524e-08 |
| perm0 | 0.253395 | 0.066332 |
| perm1 | 0.251814 | 0.00858774 |
| perm2 | 0.252591 | 0.0970938 |
| perm3 | 0.250892 | 0.0011907 |
| perm4 | 0.249155 | 2.08107e-05 |
| perm5 | 0.252783 | 0.066332 |
| perm6 | 0.252046 | 0.132961 |
| perm7 | 0.250735 | 0.000529488 |
| perm8 | 0.249432 | 0.000469247 |
| perm9 | 0.248564 | 3.58616e-06 |
| perm10 | 0.249394 | 9.89861e-06 |
| perm11 | 0.253144 | 0.184449 |
| perm12 | 0.250809 | 0.00279936 |
| perm13 | 0.24906 | 0.000185858 |
| perm14 | 0.250563 | 0.0125682 |
| perm15 | 0.248957 | 4.64697e-05 |
| perm16 | 0.249463 | 0.000754934 |
| perm17 | 0.249188 | 2.08107e-05 |
| perm18 | 0.252279 | 0.0932041 |
| perm19 | 0.25037 | 0.000273864 |

## Q1 折划分

| scheme | model | macro_r2 |
| --- | --- | --- |
| current | E1 | 0.254999 |
| current | BASE5 | 0.262041 |
| style | E1 | 0.258625 |
| style | BASE5 | 0.266056 |
| random | E1 | 0.254999 |
| random | BASE5 | 0.262634 |

## Q1 融合

| model | macro_r2 | p_dim |
| --- | --- | --- |
| equal | 0.262041 | 1 |
| drop_M1 | 0.262296 | 0.766733 |
| drop_M2 | 0.262416 | 0.825816 |
| drop_M3 | 0.263263 | 0.20796 |
| drop_M4 | 0.257855 | 0.000914937 |
| drop_M5 | 0.258961 | 0.0136869 |
| inverse_error | 0.26219 | 0.673634 |
| inner_select | 0.249511 | 0.0108796 |
| ridge_stack | 0.260691 | 0.484303 |
| leaky_select | 0.272108 | 2.55529e-05 |
| lsf_def | 0.259378 | 0.218662 |
| zits | 0.262041 | 1 |

## Q1 缺失压力

| model | macro_r2 |
| --- | --- |
| sentinel | 0.254999 |
| median_0_0 | 0.256099 |
| knn_0_0 | 0.252316 |
| zero_0_0 | 0.255016 |
| drop_0_0 | 0.256943 |
| sentinel_0.05_0 | 0.223153 |
| sentinel_0.05_1 | 0.238829 |
| sentinel_0.1_0 | 0.201078 |
| sentinel_0.1_1 | 0.205642 |
| sentinel_0.2_0 | 0.169409 |
| sentinel_0.2_1 | 0.176034 |

## Q2 混杂

| method | jaccard |
| --- | --- |
| K1_style_demean | 0.196 |
| K2_partial_spearman | 0.328 |
| K3_random_effects | 0.184 |

## Q2 Top3预测

| model | macro_r2 | weak12_r2 | strong5_r2 |
| --- | --- | --- | --- |
| top_ridge | 0.0918711 | 0.0157618 | 0.230177 |
| all_ridge | -1906.43 | -2056.99 | -591.977 |
| top_ET | -0.064249 | -0.23247 | 0.287231 |
| all_ET | 0.244485 | 0.0463643 | 0.657038 |

## Q2 六链一致性

| spearman | lasso | diffusion | permutation | RF | stability | method |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.160952 | 0.168095 | 0.157619 | 0.317302 | 0.149286 | spearman |
| 0.160952 | 1 | 0.966667 | 0.119365 | 0.137698 | 0.689524 | lasso |
| 0.168095 | 0.966667 | 1 | 0.122937 | 0.14127 | 0.684762 | diffusion |
| 0.157619 | 0.119365 | 0.122937 | 1 | 0.181984 | 0.111587 | permutation |
| 0.317302 | 0.137698 | 0.14127 | 0.181984 | 1 | 0.125952 | RF |
| 0.149286 | 0.689524 | 0.684762 | 0.111587 | 0.125952 | 1 | stability |

## Q3 生成

| method | mmd2 | energy | ks_dim_median | nearest_ratio | fake_classifier_accuracy |
| --- | --- | --- | --- | --- | --- |
| MAIN | 0.00237941 | 0.120363 | 0.056 | 1.06796 | 0.440632 |
| G0 | 0.0112113 | 0.356519 | 0.264 | 1.12436 | 0.753767 |
| G1 | 0.0130295 | 0.364053 | 0.264 | 1.07372 | 0.670936 |
| G2 | 0.00101533 | 0.0512688 | 0.044 | 0.00013178 | 0.48108 |
| G3 | 0.0141867 | 0.425749 | 0.276 | 1.17108 | 0.786012 |

## Q3 迁移

| method | state | rmse_portrait | ks_nondef | mmd2 | defect_increases |
| --- | --- | --- | --- | --- | --- |
| T0 | raw | 0.0381628 | 0.673845 | 0.132201 | 41 |
| T0 | guarded | 0.0397228 | 0.673845 | 0.136791 | 0 |
| T1 | raw | 0.0410036 | 0.338649 | 0.0296197 | 26 |
| T1 | guarded | 0.0408702 | 0.338649 | 0.0333992 | 0 |
| T2 | raw | 0.0362235 | 0.616899 | 0.12245 | 40 |
| T2 | guarded | 0.0375455 | 0.616899 | 0.128174 | 0 |
| T3 | raw | 0.0612244 | 0.377976 | 0.0462379 | 36 |
| T3 | guarded | 0.0636639 | 0.377976 | 0.0505103 | 0 |

## Q3 同口径应力

| hyperbolic | euclidean | ratio |
| --- | --- | --- |
| 0.234428 | 0.223292 | 0.952499 |

## 尚未满足原设计的部分

C8：3.xlsx是12张空白评分模板，缺真实独立评分，无法计算外部Kappa。C14：缺经过核验的浓度单位与两套独立阈值，未伪造OAV结果。C13：已有1000次各比例子采样、条件频率BCa/Wilson蒙特卡洛区间和系数BCa；选择频率的原数据BCa需要嵌套重抽样，未声称完成。

C17的T1因无样本配对，改为位置尺度仿射。C11用设计允许的小森林PI代理，不能称为BASE5的重要性。全部p值仅探索性；不显著不证明等效。详见../methodology/feasibility.md与各报告。

## 数值限制

C10保存的运行日志有84次Lasso候选/筛选收敛警告；预测缓存均有限，但并不表示全部候选优化器已精确收敛。详见C10报告及日志，相关排序只作探索性结果。

## 验证

主线SHA256内容核验：{'unchanged': True, 'changed': []}。预测缓存有限性核验：1179份，异常0份。
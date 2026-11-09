# 原主线与对照消融总表

仅汇总相同固定十折的点估计；leaky_select不可作预测性能。扫描最高点不是选择后的无偏性能。p_reference列说明各行的逐维配对参照，p值仅探索性。

| experiment | model | macro_r2 | weak12_r2 | strong5_r2 | p_reference | valid_for_predictive_comparison | p_dim | role |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| original | E0 | 0.248227 | 0.0502978 | 0.649975 |  | True | nan | nan |
| original | E1 | 0.254999 | 0.0548745 | 0.668137 |  | True | nan | nan |
| original | E2 | 0.212177 | 0.0355711 | 0.554708 |  | True | nan | nan |
| original | E5 | 0.213744 | 0.0438603 | 0.558384 |  | True | nan | nan |
| original | E3 | 0.253189 | 0.0689643 | 0.656684 |  | True | nan | nan |
| original | E4 | 0.241598 | 0.0404757 | 0.662073 |  | True | nan | nan |
| original | BASE5 | 0.262041 | 0.0766585 | 0.669131 |  | True | nan | nan |
| original | LSF_sel | 0.254376 | 0.06854 | 0.648693 |  | True | nan | nan |
| original | LSF_def | 0.259378 | 0.0741453 | 0.668698 |  | True | nan | nan |
| original | ZITS | 0.262041 | 0.0766585 | 0.669131 |  | True | nan | nan |
| original | FINAL | 0.262041 | 0.0766585 | 0.669131 |  | True | nan | nan |
| C1 | E0 | 0.248227 | 0.0502978 | 0.649975 | BASE5 | True | 4.52151e-06 | fixed_control |
| C1 | E1 | 0.254999 | 0.0548745 | 0.668137 | BASE5 | True | 0.0990859 | fixed_control |
| C1 | BASE5 | 0.262041 | 0.0766585 | 0.669131 | BASE5 | True | 1 | fixed_control |
| C1 | B0 | -0.00694863 | -0.00773997 | -0.00572112 | BASE5 | True | 1.77636e-15 | fixed_control |
| C1 | B1 | 0.15235 | -0.022785 | 0.537261 | BASE5 | True | 2.4869e-14 | fixed_control |
| C1 | B2 | -3.28367 | -0.850804 | -7.25205 | BASE5 | True | 1.77636e-14 | fixed_control |
| C1 | B3 | -3.64171 | -0.0310131 | -36.9932 | BASE5 | True | 4.49418e-13 | fixed_control |
| C1 | B4 | -0.346557 | -0.141197 | -1.44982 | BASE5 | True | 1.77636e-15 | fixed_control |
| C1 | B5_5 | 0.0984633 | -0.0582851 | 0.531018 | BASE5 | True | 1.77636e-15 | fixed_control |
| C1 | B5_15 | 0.150911 | 0.0150982 | 0.490459 | BASE5 | True | 1.77636e-15 | fixed_control |
| C1 | B5_25 | 0.136594 | 0.0215906 | 0.410509 | BASE5 | True | 1.77636e-15 | fixed_control |
| C1 | B6 | 0.190188 | 0.00117663 | 0.657173 | BASE5 | True | 1.28717e-10 | fixed_control |
| C1 | B7 | 0.253471 | 0.0780828 | 0.693056 | BASE5 | True | 0.19096 | fixed_control |
| C1 | B2_med | -3.00967 | -0.717515 | -6.45791 | BASE5 | True | 3.37508e-14 | fixed_control |
| C1 | B3_med | -2.77317 | 0.0308042 | -27.6615 | BASE5 | True | 2.02322e-10 | fixed_control |
| C1 | B2_bounded | 0.160384 | 0.00812028 | 0.466341 | BASE5 | True | 1.5449e-11 | posthoc_extrapolation_diagnostic |
| C1 | B3_bounded | 0.186405 | 0.0049942 | 0.584246 | BASE5 | True | 4.21885e-12 | posthoc_extrapolation_diagnostic |
| C1 | B4_bounded | 0.0849939 | -0.0272926 | 0.413854 | BASE5 | True | 1.24345e-14 | posthoc_extrapolation_diagnostic |
| C1 | B2_med_bounded | 0.174687 | 0.0406043 | 0.498759 | BASE5 | True | 3.13456e-10 | posthoc_extrapolation_diagnostic |
| C1 | B3_med_bounded | 0.19138 | 0.0313015 | 0.586075 | BASE5 | True | 4.79229e-10 | posthoc_extrapolation_diagnostic |
| C3 | E1 | 0.254999 | 0.0548745 | 0.668137 | E1 | True | 1 | fixed_control |
| C3 | euclidean | 0.215025 | 0.0413276 | 0.552942 | E1 | True | 8.8427e-12 | fixed_control |
| C3 | c1e-08 | 0.213709 | 0.0441978 | 0.558416 | E1 | True | 2.9944e-11 | fixed_control |
| C3 | c0.1 | 0.214813 | 0.0441522 | 0.559105 | E1 | True | 1.17275e-11 | fixed_control |
| C3 | c1 | 0.212177 | 0.0355711 | 0.554708 | E1 | True | 1.13687e-12 | fixed_control |
| C3 | c10 | 0.241771 | 0.0387564 | 0.655267 | E1 | True | 3.38221e-09 | fixed_control |
| C3 | corrected_c1e-08 | 0.212996 | 0.0361702 | 0.565904 | E1 | True | 1.56319e-13 | fixed_control |
| C3 | corrected_c0.1 | 0.213967 | 0.0374856 | 0.562925 | E1 | True | 2.63434e-12 | fixed_control |
| C3 | corrected_c1 | 0.212177 | 0.0355711 | 0.554708 | E1 | True | 1.13687e-12 | fixed_control |
| C3 | corrected_c10 | 0.192069 | 0.0270117 | 0.494888 | E1 | True | 3.00204e-13 | fixed_control |
| C3 | d2_dist | 0.21801 | 0.0373705 | 0.566132 | E1 | True | 4.91163e-12 | fixed_control |
| C3 | d2_typ | 0.255767 | 0.054501 | 0.667471 | E1 | True | 0.572086 | fixed_control |
| C3 | d2_tan | 0.249771 | 0.0534737 | 0.660972 | E1 | True | 8.23756e-05 | fixed_control |
| C3 | d4_dist | 0.215025 | 0.039642 | 0.558819 | E1 | True | 5.86198e-14 | fixed_control |
| C3 | d4_typ | 0.254755 | 0.0531552 | 0.666427 | E1 | True | 0.901018 | fixed_control |
| C3 | d4_tan | 0.249067 | 0.0535797 | 0.651008 | E1 | True | 5.63962e-05 | fixed_control |
| C3 | d8_dist | 0.208592 | 0.0378742 | 0.533315 | E1 | True | 1.89893e-12 | fixed_control |
| C3 | d8_typ | 0.257018 | 0.0556474 | 0.666226 | E1 | True | 0.127993 | fixed_control |
| C3 | d8_tan | 0.246808 | 0.0487836 | 0.655902 | E1 | True | 7.23524e-08 | fixed_control |
| C3 | perm0 | 0.253395 | 0.0546147 | 0.661159 | E1 | True | 0.066332 | fixed_control |
| C3 | perm1 | 0.251814 | 0.0529741 | 0.66461 | E1 | True | 0.00858774 | fixed_control |
| C3 | perm2 | 0.252591 | 0.0545446 | 0.664481 | E1 | True | 0.0970938 | fixed_control |
| C3 | perm3 | 0.250892 | 0.0532843 | 0.664103 | E1 | True | 0.0011907 | fixed_control |
| C3 | perm4 | 0.249155 | 0.0508557 | 0.660952 | E1 | True | 2.08107e-05 | fixed_control |
| C3 | perm5 | 0.252783 | 0.0557446 | 0.666294 | E1 | True | 0.066332 | fixed_control |
| C3 | perm6 | 0.252046 | 0.0485436 | 0.665849 | E1 | True | 0.132961 | fixed_control |
| C3 | perm7 | 0.250735 | 0.0554589 | 0.661687 | E1 | True | 0.000529488 | fixed_control |
| C3 | perm8 | 0.249432 | 0.0500971 | 0.661956 | E1 | True | 0.000469247 | fixed_control |
| C3 | perm9 | 0.248564 | 0.0474437 | 0.665008 | E1 | True | 3.58616e-06 | fixed_control |
| C3 | perm10 | 0.249394 | 0.0535289 | 0.661661 | E1 | True | 9.89861e-06 | fixed_control |
| C3 | perm11 | 0.253144 | 0.0560622 | 0.660331 | E1 | True | 0.184449 | fixed_control |
| C3 | perm12 | 0.250809 | 0.0534403 | 0.660807 | E1 | True | 0.00279936 | fixed_control |
| C3 | perm13 | 0.24906 | 0.0553292 | 0.657013 | E1 | True | 0.000185858 | fixed_control |
| C3 | perm14 | 0.250563 | 0.0526812 | 0.663968 | E1 | True | 0.0125682 | fixed_control |
| C3 | perm15 | 0.248957 | 0.0554667 | 0.660008 | E1 | True | 4.64697e-05 | fixed_control |
| C3 | perm16 | 0.249463 | 0.0537493 | 0.661423 | E1 | True | 0.000754934 | fixed_control |
| C3 | perm17 | 0.249188 | 0.0479586 | 0.664424 | E1 | True | 2.08107e-05 | fixed_control |
| C3 | perm18 | 0.252279 | 0.0489729 | 0.664187 | E1 | True | 0.0932041 | fixed_control |
| C3 | perm19 | 0.25037 | 0.0530071 | 0.661121 | E1 | True | 0.000273864 | fixed_control |
| C4 | equal | 0.262041 | 0.0766585 | 0.669131 | BASE5 | True | 1 | fixed_control |
| C4 | drop_M1 | 0.262296 | 0.0793583 | 0.672186 | BASE5 | True | 0.766733 | fixed_control |
| C4 | drop_M2 | 0.262416 | 0.0796254 | 0.670973 | BASE5 | True | 0.825816 | fixed_control |
| C4 | drop_M3 | 0.263263 | 0.0800032 | 0.672595 | BASE5 | True | 0.20796 | fixed_control |
| C4 | drop_M4 | 0.257855 | 0.0698438 | 0.660961 | BASE5 | True | 0.000914937 | fixed_control |
| C4 | drop_M5 | 0.258961 | 0.0674851 | 0.665999 | BASE5 | True | 0.0136869 | fixed_control |
| C4 | inverse_error | 0.26219 | 0.0768836 | 0.669562 | BASE5 | True | 0.673634 | fixed_control |
| C4 | inner_select | 0.249511 | 0.0660123 | 0.66697 | BASE5 | True | 0.0108796 | fixed_control |
| C4 | ridge_stack | 0.260691 | 0.0606492 | 0.688634 | BASE5 | True | 0.484303 | fixed_control |
| C4 | leaky_select | 0.272108 | 0.0908169 | 0.681985 | BASE5 | False | 2.55529e-05 | leakage_demonstration_only |
| C4 | lsf_def | 0.259378 | 0.0741453 | 0.668698 | BASE5 | True | 0.218662 | fixed_control |
| C4 | zits | 0.262041 | 0.0766585 | 0.669131 | BASE5 | True | 1 | fixed_control |
| C6 | sentinel | 0.254999 | 0.0548745 | 0.668137 | E1 | True | 1 | fixed_control |
| C6 | median_0_0 | 0.256099 | 0.0557782 | 0.671152 | E1 | True | 0.277988 | fixed_control |
| C6 | knn_0_0 | 0.252316 | 0.0508096 | 0.66672 | E1 | True | 0.0399683 | fixed_control |
| C6 | zero_0_0 | 0.255016 | 0.0528448 | 0.671165 | E1 | True | 0.946606 | fixed_control |
| C6 | drop_0_0 | 0.256943 | 0.0549562 | 0.673188 | E1 | True | 0.160017 | fixed_control |
| C6 | sentinel_0.05_0 | 0.223153 | 0.0549193 | 0.630759 | E1 | True | 9.74576e-08 | fixed_control |
| C6 | sentinel_0.05_1 | 0.238829 | 0.0603351 | 0.639336 | E1 | True | 0.000221142 | fixed_control |
| C6 | sentinel_0.1_0 | 0.201078 | 0.03839 | 0.609719 | E1 | True | 4.31486e-10 | fixed_control |
| C6 | sentinel_0.1_1 | 0.205642 | 0.0432488 | 0.611035 | E1 | True | 9.78513e-10 | fixed_control |
| C6 | sentinel_0.2_0 | 0.169409 | 0.00724525 | 0.56601 | E1 | True | 4.44089e-14 | fixed_control |
| C6 | sentinel_0.2_1 | 0.176034 | 0.0151811 | 0.581282 | E1 | True | 1.95399e-13 | fixed_control |
| C7 | 16_mf0.3_leaf1_n100 | 0.244682 | 0.0456778 | 0.664371 | E1 | True | 7.54686e-09 | sensitivity_scan |
| C7 | 16_mf0.3_leaf1_n300 | 0.252622 | 0.0547944 | 0.660287 | E1 | True | 0.0483095 | sensitivity_scan |
| C7 | 16_mf0.3_leaf1_n800 | 0.253149 | 0.0583961 | 0.659005 | E1 | True | 0.237353 | sensitivity_scan |
| C7 | 16_mf0.3_leaf2_n100 | 0.249599 | 0.0624471 | 0.651615 | E1 | True | 0.0105666 | sensitivity_scan |
| C7 | 16_mf0.3_leaf2_n300 | 0.25202 | 0.0646127 | 0.651984 | E1 | True | 0.0876008 | sensitivity_scan |
| C7 | 16_mf0.3_leaf2_n800 | 0.252838 | 0.0639308 | 0.655426 | E1 | True | 0.174988 | sensitivity_scan |
| C7 | 16_mf0.3_leaf5_n100 | 0.228953 | 0.0456986 | 0.625345 | E1 | True | 7.54686e-09 | sensitivity_scan |
| C7 | 16_mf0.3_leaf5_n300 | 0.231424 | 0.0494651 | 0.626527 | E1 | True | 4.2364e-08 | sensitivity_scan |
| C7 | 16_mf0.3_leaf5_n800 | 0.231496 | 0.049277 | 0.626754 | E1 | True | 1.15832e-08 | sensitivity_scan |
| C7 | 16_mf0.5_leaf1_n100 | 0.251701 | 0.0557453 | 0.663273 | E1 | True | 0.0157434 | sensitivity_scan |
| C7 | 16_mf0.5_leaf1_n300 | 0.254999 | 0.0548745 | 0.668137 | E1 | True | 1 | sensitivity_scan |
| C7 | 16_mf0.5_leaf1_n800 | 0.256441 | 0.0571153 | 0.669866 | E1 | True | 0.0913058 | sensitivity_scan |
| C7 | 16_mf0.5_leaf2_n100 | 0.252269 | 0.0553091 | 0.661659 | E1 | True | 0.162932 | sensitivity_scan |
| C7 | 16_mf0.5_leaf2_n300 | 0.254036 | 0.0588203 | 0.662584 | E1 | True | 0.362243 | sensitivity_scan |
| C7 | 16_mf0.5_leaf2_n800 | 0.25542 | 0.0608597 | 0.664445 | E1 | True | 0.803527 | sensitivity_scan |
| C7 | 16_mf0.5_leaf5_n100 | 0.236124 | 0.0450476 | 0.640713 | E1 | True | 1.26018e-08 | sensitivity_scan |
| C7 | 16_mf0.5_leaf5_n300 | 0.237183 | 0.0450479 | 0.641897 | E1 | True | 6.22072e-08 | sensitivity_scan |
| C7 | 16_mf0.5_leaf5_n800 | 0.236967 | 0.0471115 | 0.641665 | E1 | True | 5.76496e-08 | sensitivity_scan |
| C7 | 16_mf0.7_leaf1_n100 | 0.24752 | 0.039858 | 0.661718 | E1 | True | 0.00153956 | sensitivity_scan |
| C7 | 16_mf0.7_leaf1_n300 | 0.256065 | 0.0504348 | 0.664091 | E1 | True | 0.923781 | sensitivity_scan |
| C7 | 16_mf0.7_leaf1_n800 | 0.255777 | 0.0513103 | 0.665861 | E1 | True | 0.855748 | sensitivity_scan |
| C7 | 16_mf0.7_leaf2_n100 | 0.252156 | 0.0545197 | 0.663398 | E1 | True | 0.0932041 | sensitivity_scan |
| C7 | 16_mf0.7_leaf2_n300 | 0.254446 | 0.0543744 | 0.666804 | E1 | True | 0.759433 | sensitivity_scan |
| C7 | 16_mf0.7_leaf2_n800 | 0.255987 | 0.0569121 | 0.666554 | E1 | True | 0.687682 | sensitivity_scan |
| C7 | 16_mf0.7_leaf5_n100 | 0.235463 | 0.0465077 | 0.636031 | E1 | True | 1.26018e-08 | sensitivity_scan |
| C7 | 16_mf0.7_leaf5_n300 | 0.237254 | 0.0463422 | 0.640908 | E1 | True | 3.62327e-08 | sensitivity_scan |
| C7 | 16_mf0.7_leaf5_n800 | 0.238592 | 0.0468462 | 0.643172 | E1 | True | 1.50792e-07 | sensitivity_scan |
| C7 | 16_mf1.0_leaf1_n100 | 0.247713 | 0.03544 | 0.657291 | E1 | True | 0.00652521 | sensitivity_scan |
| C7 | 16_mf1.0_leaf1_n300 | 0.251266 | 0.039508 | 0.661478 | E1 | True | 0.0471943 | sensitivity_scan |
| C7 | 16_mf1.0_leaf1_n800 | 0.254249 | 0.0449486 | 0.665467 | E1 | True | 0.352175 | sensitivity_scan |
| C7 | 16_mf1.0_leaf2_n100 | 0.248494 | 0.0477372 | 0.655395 | E1 | True | 0.0054056 | sensitivity_scan |
| C7 | 16_mf1.0_leaf2_n300 | 0.254702 | 0.0528108 | 0.664883 | E1 | True | 0.578629 | sensitivity_scan |
| C7 | 16_mf1.0_leaf2_n800 | 0.25576 | 0.053639 | 0.663751 | E1 | True | 0.969474 | sensitivity_scan |
| C7 | 16_mf1.0_leaf5_n100 | 0.237189 | 0.0415853 | 0.639735 | E1 | True | 5.76496e-08 | sensitivity_scan |
| C7 | 16_mf1.0_leaf5_n300 | 0.237938 | 0.0419624 | 0.646966 | E1 | True | 5.34062e-08 | sensitivity_scan |
| C7 | 16_mf1.0_leaf5_n800 | 0.238366 | 0.0427432 | 0.648049 | E1 | True | 6.22072e-08 | sensitivity_scan |
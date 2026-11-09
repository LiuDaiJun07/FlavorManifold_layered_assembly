# C20 路径任务对照与应力复核

运行协议：FOLDS=10, TREE_SCALE=1.0, 当前并行任务上限=6（缓存可复用同参数的先前计算）。

检验为探索性：目标维度相关、CV训练集重叠，p值不可当独立重复证据；弱/强维为事后子集。

## metrics

| method | t | nearest_distance | guard_fixes | mmd2 | energy | mmd_p | energy_p |
| --- | --- | --- | --- | --- | --- | --- | --- |
| euclidean | 0.25 | 5.78038 | 35 | 0.0591643 | 2.36019 | 0.0598802 | 0.0698603 |
| euclidean | 0.5 | 6.46245 | 40 | 0.0479849 | 2.018 | 0.199601 | 0.201597 |
| euclidean | 0.75 | 7.03175 | 40 | 0.0414651 | 1.82409 | 0.423154 | 0.403194 |
| main_origin_tangent | 0.25 | 5.33752 | 28 | 0.0684317 | 2.63966 | 0.0319361 | 0.0319361 |
| main_origin_tangent | 0.5 | 5.83645 | 36 | 0.0576571 | 2.30941 | 0.0638723 | 0.0818363 |
| main_origin_tangent | 0.75 | 6.48975 | 39 | 0.0478084 | 2.0107 | 0.201597 | 0.207585 |
| true_geodesic | 0.25 | 5.3253 | 28 | 0.0688223 | 2.65156 | 0.0319361 | 0.0319361 |
| true_geodesic | 0.5 | 5.8174 | 36 | 0.0580863 | 2.32245 | 0.0618762 | 0.0778443 |
| true_geodesic | 0.75 | 6.48052 | 39 | 0.047927 | 2.01437 | 0.201597 | 0.203593 |

## weights

| method | t | style | weight |
| --- | --- | --- | --- |
| euclidean | 0.25 | Amber | 0.0369582 |
| euclidean | 0.25 | Blond | 0.235599 |
| euclidean | 0.25 | Brett/cofermented | 0.0969431 |
| euclidean | 0.25 | Brown | 0.00118251 |
| euclidean | 0.25 | Brut | 5.1191e-05 |
| euclidean | 0.25 | Christmas | 1.60223e-05 |
| euclidean | 0.25 | Dubbel | 0.0046212 |
| euclidean | 0.25 | Faro | 9.08164e-05 |
| euclidean | 0.25 | Flanders old brown | 0.0131343 |
| euclidean | 0.25 | Fruitbeer | 0.000250621 |
| euclidean | 0.25 | Hoppy | 0.00875215 |
| euclidean | 0.25 | Kriek | 0.000174989 |
| euclidean | 0.25 | Lambic | 0.0191213 |
| euclidean | 0.25 | Low/No alcohol | 0.00758403 |
| euclidean | 0.25 | Pils/Lager | 0.262212 |
| euclidean | 0.25 | Saison | 0.0482762 |
| euclidean | 0.25 | Scotch | 5.90744e-06 |
| euclidean | 0.25 | Stout/Porter | 4.15885e-05 |
| euclidean | 0.25 | Strong ale | 7.18326e-05 |
| euclidean | 0.25 | Tripel | 0.0184373 |
| euclidean | 0.25 | West Flanders ale | 0.0052471 |
| euclidean | 0.25 | Wheat | 0.241228 |
| euclidean | 0.5 | Amber | 0.119208 |
| euclidean | 0.5 | Blond | 0.277518 |
| euclidean | 0.5 | Brett/cofermented | 0.104676 |
| euclidean | 0.5 | Brown | 0.0132651 |
| euclidean | 0.5 | Brut | 0.000169667 |
| euclidean | 0.5 | Christmas | 0.000170049 |
| euclidean | 0.5 | Dubbel | 0.022987 |
| euclidean | 0.5 | Faro | 7.34673e-05 |
| euclidean | 0.5 | Flanders old brown | 0.00213628 |
| euclidean | 0.5 | Fruitbeer | 7.8786e-05 |
| euclidean | 0.5 | Hoppy | 0.0464418 |
| euclidean | 0.5 | Kriek | 3.81147e-05 |
| euclidean | 0.5 | Lambic | 0.00249805 |
| euclidean | 0.5 | Low/No alcohol | 0.00469427 |
| euclidean | 0.5 | Pils/Lager | 0.100104 |
| euclidean | 0.5 | Saison | 0.0966223 |
| euclidean | 0.5 | Scotch | 5.31129e-05 |
| euclidean | 0.5 | Stout/Porter | 0.000633322 |
| euclidean | 0.5 | Strong ale | 0.00120463 |
| euclidean | 0.5 | Tripel | 0.100104 |
| euclidean | 0.5 | West Flanders ale | 0.000866768 |
| euclidean | 0.5 | Wheat | 0.106457 |
| euclidean | 0.75 | Amber | 0.175363 |
| euclidean | 0.75 | Blond | 0.14909 |
| euclidean | 0.75 | Brett/cofermented | 0.0515483 |
| euclidean | 0.75 | Brown | 0.0678661 |
| euclidean | 0.75 | Brut | 0.000256472 |
| euclidean | 0.75 | Christmas | 0.000823118 |
| euclidean | 0.75 | Dubbel | 0.0521494 |
| euclidean | 0.75 | Faro | 2.71058e-05 |
| euclidean | 0.75 | Flanders old brown | 0.000158471 |
| euclidean | 0.75 | Fruitbeer | 1.12959e-05 |
| euclidean | 0.75 | Hoppy | 0.112394 |
| euclidean | 0.75 | Kriek | 3.7863e-06 |
| euclidean | 0.75 | Lambic | 0.000148842 |
| euclidean | 0.75 | Low/No alcohol | 0.00132518 |
| euclidean | 0.75 | Pils/Lager | 0.0174298 |
| euclidean | 0.75 | Saison | 0.0881985 |
| euclidean | 0.75 | Scotch | 0.000217791 |
| euclidean | 0.75 | Stout/Porter | 0.00439861 |
| euclidean | 0.75 | Strong ale | 0.0092135 |
| euclidean | 0.75 | Tripel | 0.247885 |
| euclidean | 0.75 | West Flanders ale | 6.53019e-05 |
| euclidean | 0.75 | Wheat | 0.0214268 |
| main_origin_tangent | 0.25 | Amber | 0.0311785 |
| main_origin_tangent | 0.25 | Blond | 0.162073 |
| main_origin_tangent | 0.25 | Brett/cofermented | 0.0340332 |
| main_origin_tangent | 0.25 | Brown | 0.000714179 |
| main_origin_tangent | 0.25 | Brut | 0.000109167 |
| main_origin_tangent | 0.25 | Christmas | 2.90426e-06 |
| main_origin_tangent | 0.25 | Dubbel | 0.0112819 |
| main_origin_tangent | 0.25 | Faro | 0.00102857 |
| main_origin_tangent | 0.25 | Flanders old brown | 0.00745974 |
| main_origin_tangent | 0.25 | Fruitbeer | 0.000484465 |
| main_origin_tangent | 0.25 | Hoppy | 0.00171394 |
| main_origin_tangent | 0.25 | Kriek | 0.000337749 |
| main_origin_tangent | 0.25 | Lambic | 0.0466718 |
| main_origin_tangent | 0.25 | Low/No alcohol | 0.0899844 |
| main_origin_tangent | 0.25 | Pils/Lager | 0.328602 |
| main_origin_tangent | 0.25 | Saison | 0.00557537 |
| main_origin_tangent | 0.25 | Scotch | 5.30259e-05 |
| main_origin_tangent | 0.25 | Stout/Porter | 1.20022e-05 |
| main_origin_tangent | 0.25 | Strong ale | 0.000157042 |
| main_origin_tangent | 0.25 | Tripel | 0.0901975 |
| main_origin_tangent | 0.25 | West Flanders ale | 0.0142896 |
| main_origin_tangent | 0.25 | Wheat | 0.17404 |
| main_origin_tangent | 0.5 | Amber | 0.0811053 |
| main_origin_tangent | 0.5 | Blond | 0.190753 |
| main_origin_tangent | 0.5 | Brett/cofermented | 0.0509263 |
| main_origin_tangent | 0.5 | Brown | 0.00447472 |
| main_origin_tangent | 0.5 | Brut | 0.000336126 |
| main_origin_tangent | 0.5 | Christmas | 2.45277e-05 |
| main_origin_tangent | 0.5 | Dubbel | 0.0354748 |
| main_origin_tangent | 0.5 | Faro | 0.00156197 |
| main_origin_tangent | 0.5 | Flanders old brown | 0.00222033 |
| main_origin_tangent | 0.5 | Fruitbeer | 0.000444889 |
| main_origin_tangent | 0.5 | Hoppy | 0.00755212 |
| main_origin_tangent | 0.5 | Kriek | 0.000188771 |
| main_origin_tangent | 0.5 | Lambic | 0.0148629 |
| main_origin_tangent | 0.5 | Low/No alcohol | 0.086468 |
| main_origin_tangent | 0.5 | Pils/Lager | 0.198678 |
| main_origin_tangent | 0.5 | Saison | 0.0157678 |
| main_origin_tangent | 0.5 | Scotch | 0.000278382 |
| main_origin_tangent | 0.5 | Stout/Porter | 0.000122287 |
| main_origin_tangent | 0.5 | Strong ale | 0.00104329 |
| main_origin_tangent | 0.5 | Tripel | 0.198715 |
| main_origin_tangent | 0.5 | West Flanders ale | 0.00667562 |
| main_origin_tangent | 0.5 | Wheat | 0.102326 |
| main_origin_tangent | 0.75 | Amber | 0.143811 |
| main_origin_tangent | 0.75 | Blond | 0.152451 |
| main_origin_tangent | 0.75 | Brett/cofermented | 0.0502452 |
| main_origin_tangent | 0.75 | Brown | 0.019595 |
| main_origin_tangent | 0.75 | Brut | 0.000699639 |
| main_origin_tangent | 0.75 | Christmas | 0.000142874 |
| main_origin_tangent | 0.75 | Dubbel | 0.0782986 |
| main_origin_tangent | 0.75 | Faro | 0.00160516 |
| main_origin_tangent | 0.75 | Flanders old brown | 0.000459956 |
| main_origin_tangent | 0.75 | Fruitbeer | 0.00027321 |
| main_origin_tangent | 0.75 | Hoppy | 0.0223005 |
| main_origin_tangent | 0.75 | Kriek | 7.05388e-05 |
| main_origin_tangent | 0.75 | Lambic | 0.00332438 |
| main_origin_tangent | 0.75 | Low/No alcohol | 0.0581344 |
| main_origin_tangent | 0.75 | Pils/Lager | 0.0838384 |
| main_origin_tangent | 0.75 | Saison | 0.0293504 |
| main_origin_tangent | 0.75 | Scotch | 0.0010058 |
| main_origin_tangent | 0.75 | Stout/Porter | 0.000875235 |
| main_origin_tangent | 0.75 | Strong ale | 0.00485991 |
| main_origin_tangent | 0.75 | Tripel | 0.30552 |
| main_origin_tangent | 0.75 | West Flanders ale | 0.00216147 |
| main_origin_tangent | 0.75 | Wheat | 0.0409775 |
| true_geodesic | 0.25 | Amber | 0.0306464 |
| true_geodesic | 0.25 | Blond | 0.15882 |
| true_geodesic | 0.25 | Brett/cofermented | 0.0328184 |
| true_geodesic | 0.25 | Brown | 0.000719312 |
| true_geodesic | 0.25 | Brut | 0.00011729 |
| true_geodesic | 0.25 | Christmas | 3.07489e-06 |
| true_geodesic | 0.25 | Dubbel | 0.0116232 |
| true_geodesic | 0.25 | Faro | 0.00110478 |
| true_geodesic | 0.25 | Flanders old brown | 0.00783707 |
| true_geodesic | 0.25 | Fruitbeer | 0.000524663 |
| true_geodesic | 0.25 | Hoppy | 0.00166788 |
| true_geodesic | 0.25 | Kriek | 0.000366233 |
| true_geodesic | 0.25 | Lambic | 0.04803 |
| true_geodesic | 0.25 | Low/No alcohol | 0.0935644 |
| true_geodesic | 0.25 | Pils/Lager | 0.330063 |
| true_geodesic | 0.25 | Saison | 0.0053749 |
| true_geodesic | 0.25 | Scotch | 5.62268e-05 |
| true_geodesic | 0.25 | Stout/Porter | 1.23347e-05 |
| true_geodesic | 0.25 | Strong ale | 0.000162529 |
| true_geodesic | 0.25 | Tripel | 0.0906173 |
| true_geodesic | 0.25 | West Flanders ale | 0.0150736 |
| true_geodesic | 0.25 | Wheat | 0.170797 |
| true_geodesic | 0.5 | Amber | 0.0794286 |
| true_geodesic | 0.5 | Blond | 0.1862 |
| true_geodesic | 0.5 | Brett/cofermented | 0.0486387 |
| true_geodesic | 0.5 | Brown | 0.00452857 |
| true_geodesic | 0.5 | Brut | 0.000371531 |
| true_geodesic | 0.5 | Christmas | 2.66052e-05 |
| true_geodesic | 0.5 | Dubbel | 0.0370345 |
| true_geodesic | 0.5 | Faro | 0.00172386 |
| true_geodesic | 0.5 | Flanders old brown | 0.00237505 |
| true_geodesic | 0.5 | Fruitbeer | 0.000495992 |
| true_geodesic | 0.5 | Hoppy | 0.00729111 |
| true_geodesic | 0.5 | Kriek | 0.000210611 |
| true_geodesic | 0.5 | Lambic | 0.0154795 |
| true_geodesic | 0.5 | Low/No alcohol | 0.0913354 |
| true_geodesic | 0.5 | Pils/Lager | 0.200479 |
| true_geodesic | 0.5 | Saison | 0.0150387 |
| true_geodesic | 0.5 | Scotch | 0.00030244 |
| true_geodesic | 0.5 | Stout/Porter | 0.000127276 |
| true_geodesic | 0.5 | Strong ale | 0.00109633 |
| true_geodesic | 0.5 | Tripel | 0.200479 |
| true_geodesic | 0.5 | West Flanders ale | 0.00718185 |
| true_geodesic | 0.5 | Wheat | 0.100154 |
| true_geodesic | 0.75 | Amber | 0.141575 |
| true_geodesic | 0.75 | Blond | 0.149808 |
| true_geodesic | 0.75 | Brett/cofermented | 0.0485635 |
| true_geodesic | 0.75 | Brown | 0.0197749 |
| true_geodesic | 0.75 | Brut | 0.000755349 |
| true_geodesic | 0.75 | Christmas | 0.000152193 |
| true_geodesic | 0.75 | Dubbel | 0.0809303 |
| true_geodesic | 0.75 | Faro | 0.00172946 |
| true_geodesic | 0.75 | Flanders old brown | 0.000483534 |
| true_geodesic | 0.75 | Fruitbeer | 0.000296403 |
| true_geodesic | 0.75 | Hoppy | 0.021705 |
| true_geodesic | 0.75 | Kriek | 7.65185e-05 |
| true_geodesic | 0.75 | Lambic | 0.00342776 |
| true_geodesic | 0.75 | Low/No alcohol | 0.0605923 |
| true_geodesic | 0.75 | Pils/Lager | 0.0844685 |
| true_geodesic | 0.75 | Saison | 0.0283165 |
| true_geodesic | 0.75 | Scotch | 0.00107225 |
| true_geodesic | 0.75 | Stout/Porter | 0.0009028 |
| true_geodesic | 0.75 | Strong ale | 0.00505004 |
| true_geodesic | 0.75 | Tripel | 0.307667 |
| true_geodesic | 0.75 | West Flanders ale | 0.0022825 |
| true_geodesic | 0.75 | Wheat | 0.0403702 |

## common_stress

| hyperbolic | euclidean | ratio |
| --- | --- | --- |
| 0.234428 | 0.223292 | 0.952499 |

## calibration_stress

| calibration | hyperbolic | euclidean | ratio | max_distance_change |
| --- | --- | --- | --- | --- |
| unit | 0.234426 | 0.223359 | 0.952788 | 0.00469729 |
| sd | 0.234428 | 0.223292 | 0.952499 | 4.44089e-16 |
| minmax | 0.234428 | 0.223292 | 0.952499 | 5.55112e-16 |

## 结论与限制

同口径stress-1：双曲=0.234428，欧氏=0.223292，欧氏/双曲=0.9525。原应力优势不再成立。现行路径是原点切空间线性插值，并非一般两点间测地线，另补真正Mobius测地线。两空间先归一平均成对距离，再共享双曲侧带宽，保证带宽数值可比。两拟合器优化目标不同，此为现行实现的比较，并非两类几何的最优性能界。三口径应力另表并列；路径任务主表沿用现行sd坐标。最近邻与MMD可能给出不同排序，当前结果不能支持双曲路径全面优于欧氏路径。中间态最近邻包含源训练点且以全部真实样本为参照，是描述性真实感，不能证明存在真实中间风格。
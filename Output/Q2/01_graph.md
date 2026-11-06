# Q2 步骤一：化合物共表达图

2.csv 用可读化学名命名，1.csv 用字段名命名，按 8.xlsx 的 chemical_name 到 field_name_R 建立映射，231 个化合物全部对上，无重名。
相关阵 231x231，对角均值 1.000000（自相关为 1），对称性最大偏差 0.00e+00

## 1 图的基本形态

两种建图口径的对照（n = 231 个化合物，全部两两相关）：

| 口径 | 边数 | 连通分量 | 最大分量 | 平均度 | 社区数 | 模块度 Q |
| --- | --- | --- | --- | --- | --- | --- |
| \|rho\| >= 0.3 | 3918 | 5 | 227 | 33.9 | 8 | 0.266 |
| \|rho\| >= 0.35 | 2582 | 8 | 224 | 22.4 | 12 | 0.285 |
| \|rho\| >= 0.4 | 1660 | 15 | 217 | 14.4 | 21 | 0.426 |
| \|rho\| >= 0.45 | 1038 | 28 | 202 | 9.0 | 35 | 0.510 |
| \|rho\| >= 0.5 | 645 | 52 | 175 | 5.6 | 59 | 0.604 |
| \|rho\| >= 0.6 | 266 | 114 | 99 | 2.3 | 120 | 0.746 |
| k 近邻并集 k=8（主口径） | 1358 | 1 | 231 | 11.8 | 7 | 0.597 |

采用 k=8 的近邻并集作为主图：全图连通，模块度 0.597 明显高于卡阈值口径，社区边界更干净。
平均度 11.76，孤立节点 0 个，图密度 0.0511，平均聚类系数 0.3672
贪心模块度划分得 7 个社区，模块度 Q = 0.5970

## 2 社区构成与化学来源

| 社区 | 节点数 | 主导来源 | 示例化合物 |
| --- | --- | --- | --- |
| C0 | 52 | Hops(15), Yeast(13), Malt(8) | CS2, E.caryophyllene, SO2.mg.L., X.E..alpha.ionone, X.E..beta.ionone |
| C1 | 47 | Yeast(18), Hops(17), Spices(6) | E.methyl.isoeugenol, E.nerolidol, X14.hydroxy.alpha.humulene, X2.3.dihydro.farnesol, X2.ethyl.1.hexanol |
| C2 | 34 | Yeast(11), Hops(8), Others(5) | X.E..2.nonenal, X.E..beta.damascenone, X.E..beta.farnesene, X.E.E..2.4.decadienal, X1.propanol |
| C3 | 28 | Yeast(18), Malt(5), Spices(1) | DES, DMS, E.cinnamaldehyde, H2S, MeSH |
| C4 | 24 | Yeast(14), Hops(6), Malt(2) | GFS, X.Z..beta.farnesene, X1.2.dihydro.8.hydroxy..2E..linalool, X2.butyloctanoate, X2.methylbutyl.octanoate |
| C5 | 23 | Hops(15), Spices(3), Malt(2) | E.ocimenone, X.E..beta.ocimene, X1.alpha.terpineol, X2.methyl.benzofuran, alpha.phellandrene |
| C6 | 23 | Hops(15), Yeast(3), Others(3) | CO2.PSI., X1.10.di.epi.cubenol, X2.undecanol, X2.undecanone, X6.methyl.5.hepten.2.one |

## 3 度中心性前 20 的化合物（共表达枢纽）

| 化合物 | 化学名 | 来源 | 度中心性 | 介数 | 社区 |
| --- | --- | --- | --- | --- | --- |
| ethyl.phenylacetate | Ethyl phenylacetate | Yeast | 0.1130 | 0.0381 | C0 |
| esters_sum | Total ester compounds | nan | 0.1130 | 0.0204 | C3 |
| ortho.cresol | O-cresol | Yeast | 0.1087 | 0.0259 | C0 |
| nerol.oxide | Nerol oxide | Hops | 0.1087 | 0.0281 | C0 |
| X1.10.di.epi.cubenol | Epicubenol | Hops | 0.1043 | 0.0220 | C6 |
| ethanol..v.v. | Ethanol | Yeast | 0.1000 | 0.0282 | C3 |
| X3.furanmethanol | 3-furanmethanol | Malt | 0.1000 | 0.0293 | C0 |
| X3.methyl.2.hexanol | 3-methyl-2-hexanol | Yeast | 0.0913 | 0.0294 | C1 |
| X4.methyl.1.pentanol | 4-methyl-1-pentanol | Yeast | 0.0913 | 0.0215 | C1 |
| pentyl.octanoate | Pentyl octanoate | Yeast | 0.0913 | 0.0121 | C4 |
| X2.methylbutyl.octanoate | 2-methylbutyl octanoate | Yeast | 0.0913 | 0.0207 | C4 |
| prenyl.isobutyrate | Prenyl isobutyrate | Yeast | 0.0870 | 0.0089 | C0 |
| isoborneol | Isoborneol | Others | 0.0870 | 0.0187 | C5 |
| terpinen.4.ol | Terpinen-4-ol | Spices | 0.0870 | 0.0101 | C5 |
| isopentyl.hexanoate | Isopentyl hexanoate | Yeast | 0.0870 | 0.0124 | C4 |
| alpha.calacorene | α-calacorene | Hops | 0.0870 | 0.0176 | C0 |
| alpha.muurolol | α-muurolol | Hops | 0.0870 | 0.0175 | C6 |
| kcalperc | Caloric value | Others | 0.0826 | 0.0171 | C3 |
| ethyl.tetradecanoate | Ethyl tetradecanoate | Yeast | 0.0826 | 0.0110 | C4 |
| ethyl_acetate | Ethyl acetate | Yeast | 0.0783 | 0.0092 | C3 |

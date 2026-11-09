# C9 混杂控制

运行协议：FOLDS=10, TREE_SCALE=1.0, 并行任务=6。

检验为探索性：目标维度相关、CV训练集重叠，p值不可当独立重复证据；弱/强维为事后子集。

## summary

| method | jaccard |
| --- | --- |
| K1_style_demean | 0.196 |
| K2_partial_spearman | 0.328 |
| K3_random_effects | 0.184 |

## seats

| method | dim | rank | original | adjusted | replaced | raw_rho | adjusted_rho | I2 | valid_styles |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| K1_style_demean | A_malt_all | 1 | acetaldehyde | furfural | True | 0.259801 | 0.253534 | 0 | 16 |
| K1_style_demean | A_malt_all | 2 | methylpyrazine | ethyl_octanoate | True | 0.105317 | -0.222947 | 0.353368 | 16 |
| K1_style_demean | A_malt_all | 3 | beta.phellandrene | ethyl..9..decenoate | True | -0.0872888 | -0.1855 | 0.0815271 | 16 |
| K1_style_demean | A_malt_grain | 1 | terpinen.4.ol | terpinen.4.ol | False | -0.264539 | -0.206776 | 0.324579 | 16 |
| K1_style_demean | A_malt_grain | 2 | beta.phellandrene | DMS | True | -0.155374 | -0.19954 | 0 | 16 |
| K1_style_demean | A_malt_grain | 3 | isoborneol | X1.alpha.terpineol | True | -0.199838 | -0.196667 | 0 | 16 |
| K1_style_demean | A_malt_bread | 1 | terpinen.4.ol | X4.ethyl.guaiacol | True | -0.207802 | -0.239206 | 0 | 16 |
| K1_style_demean | A_malt_bread | 2 | ethyl.tiglate | X4.ethyl.phenol | True | -0.216364 | -0.205721 | 0.0641761 | 16 |
| K1_style_demean | A_malt_bread | 3 | acetaldehyde | neryl.formate | True | -0.225005 | -0.20115 | 0 | 16 |
| K1_style_demean | A_malt_cara | 1 | acetaldehyde | furfural | True | 0.308843 | 0.231373 | 0 | 16 |
| K1_style_demean | A_malt_cara | 2 | X2.ethyl.1.hexanol | ethyl_octanoate | True | 0.0807241 | -0.205036 | 0.255221 | 16 |
| K1_style_demean | A_malt_cara | 3 | furfural | X2.methyl.benzofuran | True | -0.300742 | -0.184017 | 0.248927 | 16 |
| K1_style_demean | A_malt_burn | 1 | isobutanol | X6.methyl.5.hepten.2.one | True | 0.0695949 | 0.170751 | 0.133202 | 11 |
| K1_style_demean | A_malt_burn | 2 | DES | y.methyl.1.octanol | True | 0.225018 | 0.168485 | 0 | 15 |
| K1_style_demean | A_malt_burn | 3 | y.methyl.1.octanol | dihydroisocaryophyllene.epoxide | True | -0.0102934 | -0.160614 | 0.217927 | 16 |
| K1_style_demean | A_hops_all | 1 | ethyl_octanoate | ethyl_octanoate | False | 0.463745 | 0.252988 | 0.201967 | 16 |
| K1_style_demean | A_hops_all | 2 | X.E..2.nonenal | ethyl_isovalerate | True | 0.0844452 | 0.242835 | 0.453288 | 15 |
| K1_style_demean | A_hops_all | 3 | ethyl_decanoate | phenethyl_acetate | True | -0.0582964 | -0.238188 | 0.343412 | 16 |
| K1_style_demean | A_hops_citrus | 1 | citronellol | isoborneol | True | 0.227881 | 0.28716 | 0.35462 | 16 |
| K1_style_demean | A_hops_citrus | 2 | humulene.epoxide.II | citronellol | True | 0.369736 | 0.281151 | 0 | 16 |
| K1_style_demean | A_hops_citrus | 3 | linalool | X1.alpha.terpineol | True | 0.221026 | 0.239103 | 0.239501 | 16 |
| K1_style_demean | A_hops_tropical | 1 | DMS | beta.phellandrene | True | 0.230655 | 0.240906 | 0.133729 | 16 |
| K1_style_demean | A_hops_tropical | 2 | X3.furanmethanol | citronellol | True | 0.255184 | 0.221494 | 0 | 16 |
| K1_style_demean | A_hops_tropical | 3 | methyl.geranate | hexyl.acetate | True | 0.22116 | 0.213671 | 0.13285 | 16 |
| K1_style_demean | A_hops_noble | 1 | ethyl_octanoate | ethyl_octanoate | False | 0.481829 | 0.309112 | 0 | 16 |
| K1_style_demean | A_hops_noble | 2 | ethyl_decanoate | ethyl_decanoate | False | 0.469279 | 0.272253 | 0 | 16 |
| K1_style_demean | A_hops_noble | 3 | X.E..2.nonenal | alpha.calacorene | True | 0.26302 | 0.210421 | 0 | 16 |
| K1_style_demean | A_hops_woody | 1 | trans.dihydro.alpha.terpinyl.acetate | alpha.calacorene | True | 0.272512 | 0.306335 | 0.244241 | 16 |
| K1_style_demean | A_hops_woody | 2 | alpha.calacorene | neryl.propanoate | True | 0.264465 | 0.280361 | 0.00862325 | 16 |
| K1_style_demean | A_hops_woody | 3 | ethyl.tetradecanoate | ethyl..E..linalool | True | 0.15616 | 0.258617 | 0.191254 | 16 |
| K1_style_demean | A_esters_all | 1 | n.hexanol | isopentyl_acetate | True | 0.292396 | 0.36941 | 0.256784 | 16 |
| K1_style_demean | A_esters_all | 2 | acetaldehyde | acetaldehyde | False | 0.35303 | 0.310683 | 0 | 16 |
| K1_style_demean | A_esters_all | 3 | ethyl_acetate | ethyl_acetate | False | 0.321283 | 0.237646 | 0.358996 | 16 |
| K1_style_demean | A_esters_ethac | 1 | ethyl_acetate | X.Z..beta.farnesene | True | 0.217257 | 0.189196 | 0 | 16 |
| K1_style_demean | A_esters_ethac | 2 | acetaldehyde | isoamyl_alcohol | True | 0.330677 | 0.182987 | 0.343184 | 16 |
| K1_style_demean | A_esters_ethac | 3 | isoamyl_alcohol | Z.Z.geranyl.linalool | True | 0.285575 | 0.180551 | 0 | 16 |
| K1_style_demean | A_esters_isoaa | 1 | n.hexanol | isopentyl_acetate | True | 0.282815 | 0.30001 | 0 | 16 |
| K1_style_demean | A_esters_isoaa | 2 | isopentyl_acetate | n.hexanol | True | 0.33993 | 0.237109 | 0 | 16 |
| K1_style_demean | A_esters_isoaa | 3 | isoamyl_alcohol | isoborneol | True | -0.113321 | -0.209892 | 0.0119179 | 16 |
| K1_style_demean | A_esters_flower | 1 | beta.phellandrene | X.E..dihydro.apofarnesal | True | -0.162619 | -0.204424 | 0 | 16 |
| K1_style_demean | A_esters_flower | 2 | isobutyl_acetate | citronellol | True | 0.112461 | 0.187604 | 0 | 16 |
| K1_style_demean | A_esters_flower | 3 | X1.alpha.terpineol | beta.phellandrene | True | 0.195764 | 0.187294 | 0.405655 | 16 |
| K1_style_demean | A_esters_fruity | 1 | beta.phellandrene | isopentyl_acetate | True | -0.0508517 | 0.203421 | 0.260885 | 16 |
| K1_style_demean | A_esters_fruity | 2 | eugenol | X1.10.di.epi.cubenol | True | -0.170111 | -0.18647 | 0 | 16 |
| K1_style_demean | A_esters_fruity | 3 | benzaldehyde | ethyl.hydrocinnamate | True | -0.132303 | -0.177014 | 0 | 16 |
| K1_style_demean | F_malt_all | 1 | acetaldehyde | ethyl..E..linalool | True | -0.300028 | -0.198347 | 0.241113 | 16 |
| K1_style_demean | F_malt_all | 2 | MeSH | geraniol | True | -0.00456358 | -0.189408 | 0 | 16 |
| K1_style_demean | F_malt_all | 3 | ethyl.butyrate | terpinen.4.ol | True | -0.300323 | -0.170778 | 0.0612052 | 16 |
| K1_style_demean | F_malt_grain | 1 | isoborneol | terpinen.4.ol.acetate | True | -0.25595 | -0.219538 | 0.211091 | 16 |
| K1_style_demean | F_malt_grain | 2 | terpinen.4.ol.acetate | isoborneol | True | -0.275786 | -0.217755 | 0.119458 | 16 |
| K1_style_demean | F_malt_grain | 3 | X.E..beta.ionone | geraniol | True | -0.134199 | -0.202502 | 0 | 16 |
| K1_style_demean | F_malt_bread | 1 | acetaldehyde | X4.ethyl.phenol | True | -0.284564 | -0.206502 | 0.280704 | 16 |
| K1_style_demean | F_malt_bread | 2 | ethyl.butyrate | terpinen.4.ol | True | -0.315758 | -0.200978 | 0 | 16 |
| K1_style_demean | F_malt_bread | 3 | isopentyl_acetate | beta.phellandrene | True | -0.287074 | -0.200131 | 0 | 16 |
| K1_style_demean | F_malt_cara | 1 | acetaldehyde | pentyl.octanoate | True | 0.00886478 | -0.167544 | 0.554793 | 16 |
| K1_style_demean | F_malt_cara | 2 | ethyl.butyrate | X1.2.dihydro.8.hydroxy..2E..linalool | True | 0.00266831 | -0.166059 | 0.186131 | 16 |
| K1_style_demean | F_malt_cara | 3 | n.hexanol | ethyl.hexadecanoate | True | 0.0752194 | -0.16539 | 0.144753 | 16 |
| K1_style_demean | F_malt_burn | 1 | MeSH | phenethyl_acetate | True | 0.098662 | -0.203156 | 0.280834 | 16 |
| K1_style_demean | F_malt_burn | 2 | ethyl_decanoate | X.Z..dihydro.apofarnesal | True | -0.178087 | -0.194105 | 0 | 16 |
| K1_style_demean | F_malt_burn | 3 | acetaldehyde | dihydroisocaryophyllene.epoxide | True | 0.0481161 | -0.181223 | 0.0384348 | 16 |
| K1_style_demean | F_hops_all | 1 | ethyl_octanoate | ethyl_octanoate | False | 0.500767 | 0.345338 | 0 | 16 |
| K1_style_demean | F_hops_all | 2 | X.E..2.nonenal | phenethyl_acetate | True | -0.0332703 | -0.320524 | 0.293495 | 16 |
| K1_style_demean | F_hops_all | 3 | ethyl_decanoate | ethyl_isovalerate | True | 0.117607 | 0.3163 | 0 | 15 |
| K1_style_demean | F_hops_citrus | 1 | humulene.epoxide.II | citronellol | True | 0.232269 | 0.21635 | 0 | 16 |
| K1_style_demean | F_hops_citrus | 2 | furfural | isoborneol | True | 0.157648 | 0.184403 | 0 | 16 |
| K1_style_demean | F_hops_citrus | 3 | ethyl.furoate | X1.2.dihydro.8.hydroxy..2E..linalool | True | 0.235325 | 0.172985 | 0.176888 | 16 |
| K1_style_demean | F_hops_tropical | 1 | geraniol | hexyl.acetate | True | 0.280912 | 0.254371 | 0 | 16 |
| K1_style_demean | F_hops_tropical | 2 | hexyl.acetate | terpinolene | True | 0.253406 | 0.195913 | 0.0596466 | 16 |
| K1_style_demean | F_hops_tropical | 3 | DMS | alpha.phellandrene | True | 0.226902 | 0.187691 | 0 | 16 |
| K1_style_demean | F_hops_noble | 1 | ethyl_octanoate | ethyl_octanoate | False | 0.505088 | 0.362923 | 0.250505 | 16 |
| K1_style_demean | F_hops_noble | 2 | ethyl_decanoate | ethyl_decanoate | False | 0.457765 | 0.285995 | 0 | 16 |
| K1_style_demean | F_hops_noble | 3 | X.E..2.nonenal | octyl_acetate | True | -0.258664 | -0.27393 | 0.374746 | 16 |
| K1_style_demean | F_hops_woody | 1 | ethyl.tetradecanoate | ethyl.tetradecanoate | False | 0.346064 | 0.231667 | 0.115577 | 16 |
| K1_style_demean | F_hops_woody | 2 | ethyl_decanoate | ethyl.hexadecanoate | True | 0.319175 | 0.197224 | 0.399407 | 16 |
| K1_style_demean | F_hops_woody | 3 | humulene.epoxide.II | ethyl_2.methyl_butyrate | True | 0.155931 | 0.194349 | 0 | 15 |
| K1_style_demean | F_esters_all | 1 | acetaldehyde | isopentyl_acetate | True | 0.310518 | 0.335827 | 0.488596 | 16 |
| K1_style_demean | F_esters_all | 2 | ethyl_acetate | acetaldehyde | True | 0.403872 | 0.267329 | 0 | 16 |
| K1_style_demean | F_esters_all | 3 | n.hexanol | phenethyl_alcohol | True | 0.215752 | 0.228728 | 0 | 16 |
| K1_style_demean | F_esters_ethac | 1 | ethyl_acetate | ethyl_acetate | False | 0.503097 | 0.270497 | 0 | 16 |
| K1_style_demean | F_esters_ethac | 2 | isoamyl_alcohol | Z.Z.geranyl.linalool | True | 0.341737 | 0.211276 | 0 | 16 |
| K1_style_demean | F_esters_ethac | 3 | acetaldehyde | acetaldehyde | False | 0.385441 | 0.201669 | 0.356161 | 16 |
| K1_style_demean | F_esters_isoaa | 1 | n.hexanol | isopentyl_acetate | True | 0.27363 | 0.296954 | 0.2856 | 16 |
| K1_style_demean | F_esters_isoaa | 2 | isopentyl_acetate | ethyl.phenylacetate | True | -0.0709657 | -0.225765 | 0 | 16 |
| K1_style_demean | F_esters_isoaa | 3 | acetaldehyde | isoborneol | True | -0.110965 | -0.213895 | 0 | 16 |
| K1_style_demean | F_esters_flower | 1 | beta.phellandrene | ethyl_2.methyl_butyrate | True | -0.0674929 | -0.238098 | 0.233509 | 15 |
| K1_style_demean | F_esters_flower | 2 | isopentyl.butanoate | ethyl.hydrocinnamate | True | -0.0766318 | -0.205221 | 0.368631 | 16 |
| K1_style_demean | F_esters_flower | 3 | terpinolene | citronellol | True | 0.0747542 | 0.181378 | 0.179979 | 16 |
| K1_style_demean | F_esters_fruity | 1 | DES | X2.ethyl.3.methylpyrazine | True | -0.0238787 | -0.212243 | 0.325214 | 16 |
| K1_style_demean | F_esters_fruity | 2 | humulene.epoxide.II | ethyl_2.methyl_butyrate | True | 0.00232327 | -0.169319 | 0 | 15 |
| K1_style_demean | F_esters_fruity | 3 | beta.phellandrene | n.octanol | True | -0.0856801 | -0.168821 | 0.278095 | 16 |
| K1_style_demean | acidity | 1 | prenyl.isobutyrate | eugenol | True | 0.218769 | 0.249436 | 0.0926713 | 16 |
| K1_style_demean | acidity | 2 | ethyl.phenylacetate | ethyl.hexadecanoate | True | 0.0200492 | 0.245413 | 0 | 16 |
| K1_style_demean | acidity | 3 | perillyl.acetate | ethyl.pentadecanoate | True | 0.00419718 | 0.24032 | 0 | 16 |
| K1_style_demean | bitternes | 1 | ethyl_octanoate | ethyl_octanoate | False | 0.592674 | 0.4655 | 0.364789 | 16 |
| K1_style_demean | bitternes | 2 | ethyl_decanoate | ethyl_decanoate | False | 0.558887 | 0.411505 | 0.339878 | 16 |
| K1_style_demean | bitternes | 3 | MeSH | phenethyl_acetate | True | 0.0497745 | -0.302655 | 0.447601 | 16 |
| K1_style_demean | sweetness | 1 | ethyl_decanoate | ethyl_decanoate | False | -0.391162 | -0.317304 | 0.105541 | 16 |
| K1_style_demean | sweetness | 2 | pentyl.octanoate | X2.methylbutyl.octanoate | True | -0.329487 | -0.312961 | 0.528372 | 16 |
| K1_style_demean | sweetness | 3 | X1.2.dihydro.8.hydroxy..2E..linalool | ethyl_octanoate | True | -0.294699 | -0.304426 | 0.139151 | 16 |
| K1_style_demean | X4vg | 1 | X4VG | X4VG | False | 0.472618 | 0.428243 | 0 | 15 |
| K1_style_demean | X4vg | 2 | delta.terpinyl.acetate | delta.terpinyl.acetate | False | 0.399224 | 0.223046 | 0 | 16 |
| K1_style_demean | X4vg | 3 | neryl.formate | isopentyl.butanoate | True | -0.157331 | -0.190236 | 0 | 16 |
| K1_style_demean | diacetyl | 1 | eugenol | ethyl_octanoate | True | -0.22731 | -0.399177 | 0 | 16 |
| K1_style_demean | diacetyl | 2 | ethyl_decanoate | ethyl_decanoate | False | -0.282542 | -0.365858 | 0.259164 | 16 |
| K1_style_demean | diacetyl | 3 | alpha.calacorene | pentyl.octanoate | True | -0.228155 | -0.258104 | 0 | 16 |
| K1_style_demean | dms | 1 | amyl.acetate | X.Z..dihydro.apofarnesal | True | 0.154057 | 0.186557 | 0 | 16 |
| K1_style_demean | dms | 2 | terpinen.4.ol | phenethyl_alcohol | True | -0.172824 | -0.180437 | 0 | 16 |
| K1_style_demean | dms | 3 | E.nerolidol | X9.epi..E..caryophyllene | True | 0.15524 | 0.169479 | 0.528051 | 16 |
| K1_style_demean | metallic | 1 | ethyl_isovalerate | isopentyl.butanoate | True | -0.228142 | -0.189451 | 0 | 16 |
| K1_style_demean | metallic | 2 | X6.methyl.5.hepten.2.one | gamma.eudesmol | True | -0.111372 | -0.161978 | 0 | 16 |
| K1_style_demean | metallic | 3 | alpha.calacorene | alpha.calacorene | False | -0.232906 | -0.161535 | 0.133661 | 16 |
| K1_style_demean | stale_hops | 1 | X2.undecanol | gamma.eudesmol | True | 0.246862 | 0.214271 | 0.592715 | 16 |
| K1_style_demean | stale_hops | 2 | alloaromadendrene.epoxide | X.E..beta.ocimene | True | 0.13375 | 0.207669 | 0.359519 | 16 |
| K1_style_demean | stale_hops | 3 | neryl.formate | beta.phellandrene | True | 0.149217 | 0.195674 | 0 | 16 |
| K1_style_demean | t2n | 1 | beta.phellandrene | vinyl.caprylate | True | 0.175324 | 0.173073 | 0 | 16 |
| K1_style_demean | t2n | 2 | terpinen.4.ol | E.cinnamaldehyde | True | 0.148822 | 0.171456 | 0 | 16 |
| K1_style_demean | t2n | 3 | geraniol | alpha.phellandrene | True | -0.186344 | -0.164479 | 0.0184736 | 16 |
| K1_style_demean | orange | 1 | camphor | X1.alpha.terpineol | True | 0.212532 | 0.415834 | 0.207216 | 16 |
| K1_style_demean | orange | 2 | limonene | linalool | True | 0.178953 | 0.361637 | 0.540561 | 16 |
| K1_style_demean | orange | 3 | beta.phellandrene | camphor | True | 0.367483 | 0.312175 | 0 | 16 |
| K1_style_demean | coriander | 1 | X2.methyl.benzofuran | geraniol | True | 0.141147 | 0.314348 | 0.500112 | 16 |
| K1_style_demean | coriander | 2 | E.ocimenone | linalool | True | 0.177351 | 0.288078 | 0.482578 | 16 |
| K1_style_demean | coriander | 3 | terpinen.4.ol | x.methyl.1.octanol | True | -0.0200353 | -0.272607 | 0.325454 | 16 |
| K1_style_demean | clove | 1 | limonene | limonene | False | 0.342317 | 0.337821 | 0.275059 | 14 |
| K1_style_demean | clove | 2 | eucalyptol | x.methyl.1.octanol | True | -0.14207 | -0.232033 | 0.6502 | 16 |
| K1_style_demean | clove | 3 | camphor | linalool | True | 0.132623 | 0.222649 | 0.449081 | 16 |
| K1_style_demean | lactic | 1 | perillyl.acetate | isopentyl_acetate | True | -0.409885 | -0.211501 | 0.262857 | 16 |
| K1_style_demean | lactic | 2 | prenyl.isobutyrate | DES | True | -0.297826 | -0.189441 | 0.170362 | 16 |
| K1_style_demean | lactic | 3 | ethyl.furoate | tau.cadinol | True | -0.130101 | -0.18473 | 0 | 16 |
| K1_style_demean | acetic | 1 | prenyl.isobutyrate | DES | True | -0.279142 | -0.268729 | 0.421213 | 16 |
| K1_style_demean | acetic | 2 | ethyl_isovalerate | eugenol | True | 0.227685 | 0.215754 | 0.636281 | 16 |
| K1_style_demean | acetic | 3 | ethyl.phenylacetate | isopentyl_acetate | True | -0.308238 | -0.214298 | 0.41776 | 16 |
| K1_style_demean | barnyard | 1 | X2.undecanol | gamma.eudesmol | True | 0.264066 | 0.226399 | 0.53551 | 16 |
| K1_style_demean | barnyard | 2 | neryl.formate | X.E..beta.ocimene | True | 0.128273 | 0.210085 | 0.349673 | 16 |
| K1_style_demean | barnyard | 3 | alloaromadendrene.epoxide | propyl.acetate | True | -0.0900172 | -0.197694 | 0 | 16 |
| K1_style_demean | alcohol | 1 | ethyl_octanoate | ethyl_octanoate | False | 0.619371 | 0.408984 | 0 | 16 |
| K1_style_demean | alcohol | 2 | ethyl_hexanoate | ethyl_acetate | True | 0.578353 | 0.393067 | 0 | 16 |
| K1_style_demean | alcohol | 3 | acetaldehyde | ethyl_decanoate | True | 0.479907 | 0.345637 | 0 | 16 |
| K1_style_demean | aftertaste | 1 | ethyl_octanoate | ethyl_octanoate | False | 0.506179 | 0.35432 | 0.456754 | 16 |
| K1_style_demean | aftertaste | 2 | X.E..2.nonenal | ethyl_isovalerate | True | 0.439277 | 0.310414 | 0 | 15 |
| K1_style_demean | aftertaste | 3 | ethyl_decanoate | ethyl_decanoate | False | 0.48915 | 0.296681 | 0.272102 | 16 |
| K1_style_demean | body | 1 | ethyl_hexanoate | X3.methyl.2.hexanol | True | -0.546826 | -0.299089 | 0.299183 | 16 |
| K1_style_demean | body | 2 | X3.methyl.2.hexanol | ethyl_octanoate | True | 0.530738 | 0.261988 | 0 | 16 |
| K1_style_demean | body | 3 | ethyl_octanoate | n.octanol | True | -0.367875 | -0.260542 | 0 | 16 |
| K1_style_demean | co2 | 1 | ethyl_octanoate | ethyl_octanoate | False | 0.552687 | 0.333867 | 0.428504 | 16 |
| K1_style_demean | co2 | 2 | ethyl_hexanoate | ethyl_hexanoate | False | 0.467309 | 0.261741 | 0.525013 | 16 |
| K1_style_demean | co2 | 3 | ethyl_decanoate | octyl_acetate | True | -0.16414 | -0.239841 | 0 | 16 |
| K1_style_demean | overall | 1 | ethyl_acetate | ethyl_acetate | False | 0.324999 | 0.199455 | 0.191729 | 16 |
| K1_style_demean | overall | 2 | ethyl_octanoate | ethyl_hexanoate | True | 0.262999 | 0.15291 | 0 | 16 |
| K1_style_demean | overall | 3 | isopentyl_acetate | ethyl_octanoate | True | 0.285956 | 0.150317 | 0.111827 | 16 |
| K2_partial_spearman | A_malt_all | 1 | acetaldehyde | beta.phellandrene | True | -0.322861 | -0.304388 | 0 | 16 |
| K2_partial_spearman | A_malt_all | 2 | methylpyrazine | terpinen.4.ol | True | -0.313791 | -0.288198 | 0 | 16 |
| K2_partial_spearman | A_malt_all | 3 | beta.phellandrene | furfural | True | 0.259801 | 0.269875 | 0 | 16 |
| K2_partial_spearman | A_malt_grain | 1 | terpinen.4.ol | terpinen.4.ol | False | -0.264539 | -0.248785 | 0.324579 | 16 |
| K2_partial_spearman | A_malt_grain | 2 | beta.phellandrene | beta.phellandrene | False | -0.248036 | -0.231601 | 0.455983 | 16 |
| K2_partial_spearman | A_malt_grain | 3 | isoborneol | isoborneol | False | -0.224753 | -0.210996 | 0.37518 | 16 |
| K2_partial_spearman | A_malt_bread | 1 | terpinen.4.ol | neryl.formate | True | -0.225005 | -0.22119 | 0 | 16 |
| K2_partial_spearman | A_malt_bread | 2 | ethyl.tiglate | terpinen.4.ol.acetate | True | -0.209155 | -0.219544 | 0 | 16 |
| K2_partial_spearman | A_malt_bread | 3 | acetaldehyde | X4.ethyl.guaiacol | True | -0.207802 | -0.218273 | 0 | 16 |
| K2_partial_spearman | A_malt_cara | 1 | acetaldehyde | furfural | True | 0.308843 | 0.278075 | 0 | 16 |
| K2_partial_spearman | A_malt_cara | 2 | X2.ethyl.1.hexanol | eugenol | True | -0.0319448 | -0.256476 | 0.0427243 | 16 |
| K2_partial_spearman | A_malt_cara | 3 | furfural | X2.ethyl.1.hexanol | True | -0.317247 | -0.253185 | 0 | 16 |
| K2_partial_spearman | A_malt_burn | 1 | isobutanol | alpha.eudesmol | True | 0.173462 | 0.179536 | 0 | 16 |
| K2_partial_spearman | A_malt_burn | 2 | DES | X2.methyltetrahydrofuran.3.one | True | -0.0913515 | -0.177993 | 0 | 16 |
| K2_partial_spearman | A_malt_burn | 3 | y.methyl.1.octanol | y.methyl.1.octanol | False | 0.225018 | 0.177443 | 0 | 15 |
| K2_partial_spearman | A_hops_all | 1 | ethyl_octanoate | linalool | True | 0.412072 | 0.357121 | 0.0618418 | 16 |
| K2_partial_spearman | A_hops_all | 2 | X.E..2.nonenal | X.E..2.nonenal | False | 0.431972 | 0.315066 | 0 | 16 |
| K2_partial_spearman | A_hops_all | 3 | ethyl_decanoate | ethyl_octanoate | True | 0.463745 | 0.303118 | 0.201967 | 16 |
| K2_partial_spearman | A_hops_citrus | 1 | citronellol | citronellol | False | 0.369736 | 0.313705 | 0 | 16 |
| K2_partial_spearman | A_hops_citrus | 2 | humulene.epoxide.II | ethyl.tetradecanoate | True | 0.282507 | 0.27975 | 0.297465 | 16 |
| K2_partial_spearman | A_hops_citrus | 3 | linalool | linalool | False | 0.326743 | 0.274993 | 0 | 16 |
| K2_partial_spearman | A_hops_tropical | 1 | DMS | beta.phellandrene | True | 0.230655 | 0.25433 | 0.133729 | 16 |
| K2_partial_spearman | A_hops_tropical | 2 | X3.furanmethanol | DMS | True | 0.314181 | 0.249199 | 0.0431023 | 16 |
| K2_partial_spearman | A_hops_tropical | 3 | methyl.geranate | hexyl.acetate | True | 0.22116 | 0.218456 | 0.13285 | 16 |
| K2_partial_spearman | A_hops_noble | 1 | ethyl_octanoate | ethyl_decanoate | True | 0.469279 | 0.37245 | 0 | 16 |
| K2_partial_spearman | A_hops_noble | 2 | ethyl_decanoate | ethyl_octanoate | True | 0.481829 | 0.344053 | 0 | 16 |
| K2_partial_spearman | A_hops_noble | 3 | X.E..2.nonenal | X1.2.dihydro.8.hydroxy..2E..linalool | True | 0.363128 | 0.317377 | 0.335127 | 16 |
| K2_partial_spearman | A_hops_woody | 1 | trans.dihydro.alpha.terpinyl.acetate | X.E..ortho.methoxy.cinnamaldehyde | True | 0.241428 | 0.300074 | 0 | 16 |
| K2_partial_spearman | A_hops_woody | 2 | alpha.calacorene | neryl.propanoate | True | 0.264465 | 0.287355 | 0.00862325 | 16 |
| K2_partial_spearman | A_hops_woody | 3 | ethyl.tetradecanoate | alpha.calacorene | True | 0.272512 | 0.274372 | 0.244241 | 16 |
| K2_partial_spearman | A_esters_all | 1 | n.hexanol | n.hexanol | False | 0.356852 | 0.311253 | 0.125181 | 16 |
| K2_partial_spearman | A_esters_all | 2 | acetaldehyde | acetaldehyde | False | 0.35303 | 0.233427 | 0 | 16 |
| K2_partial_spearman | A_esters_all | 3 | ethyl_acetate | diethyl.succinate | True | -0.052173 | -0.221886 | 0 | 16 |
| K2_partial_spearman | A_esters_ethac | 1 | ethyl_acetate | X2.methyl.1.butanol | True | 0.270958 | 0.240975 | 0.366435 | 16 |
| K2_partial_spearman | A_esters_ethac | 2 | acetaldehyde | X2.3.dihydro.farnesol | True | 0.264435 | 0.227238 | 0.36502 | 16 |
| K2_partial_spearman | A_esters_ethac | 3 | isoamyl_alcohol | X2E.6Z.farnesal | True | 0.259623 | 0.22667 | 0.343495 | 16 |
| K2_partial_spearman | A_esters_isoaa | 1 | n.hexanol | n.hexanol | False | 0.33993 | 0.350657 | 0 | 16 |
| K2_partial_spearman | A_esters_isoaa | 2 | isopentyl_acetate | isopentyl_acetate | False | 0.282815 | 0.2613 | 0 | 16 |
| K2_partial_spearman | A_esters_isoaa | 3 | isoamyl_alcohol | diethyl.succinate | True | -0.13699 | -0.227167 | 0 | 16 |
| K2_partial_spearman | A_esters_flower | 1 | beta.phellandrene | beta.phellandrene | False | 0.195764 | 0.220547 | 0.405655 | 16 |
| K2_partial_spearman | A_esters_flower | 2 | isobutyl_acetate | terpinolene | True | 0.160124 | 0.215275 | 0 | 16 |
| K2_partial_spearman | A_esters_flower | 3 | X1.alpha.terpineol | geraniol | True | 0.167656 | 0.190691 | 0.426991 | 16 |
| K2_partial_spearman | A_esters_fruity | 1 | beta.phellandrene | beta.phellandrene | False | 0.231943 | 0.249766 | 0.180943 | 16 |
| K2_partial_spearman | A_esters_fruity | 2 | eugenol | X2.6.dimethyl.pyrazine | True | -0.107987 | -0.176143 | 0.411459 | 16 |
| K2_partial_spearman | A_esters_fruity | 3 | benzaldehyde | benzaldehyde | False | 0.183205 | 0.173609 | 0 | 16 |
| K2_partial_spearman | F_malt_all | 1 | acetaldehyde | terpinen.4.ol.acetate | True | -0.199853 | -0.283192 | 0 | 16 |
| K2_partial_spearman | F_malt_all | 2 | MeSH | terpinen.4.ol | True | -0.300323 | -0.261863 | 0.0612052 | 16 |
| K2_partial_spearman | F_malt_all | 3 | ethyl.butyrate | eugenol | True | -0.0553153 | -0.253323 | 0.29999 | 16 |
| K2_partial_spearman | F_malt_grain | 1 | isoborneol | isoborneol | False | -0.275786 | -0.256842 | 0.119458 | 16 |
| K2_partial_spearman | F_malt_grain | 2 | terpinen.4.ol.acetate | terpinen.4.ol | True | -0.245022 | -0.249886 | 0 | 16 |
| K2_partial_spearman | F_malt_grain | 3 | X.E..beta.ionone | X.E..2.nonenal | True | -0.212208 | -0.247818 | 0.252194 | 16 |
| K2_partial_spearman | F_malt_bread | 1 | acetaldehyde | terpinen.4.ol | True | -0.315758 | -0.273985 | 0 | 16 |
| K2_partial_spearman | F_malt_bread | 2 | ethyl.butyrate | terpinen.4.ol.acetate | True | -0.247311 | -0.266371 | 0 | 16 |
| K2_partial_spearman | F_malt_bread | 3 | isopentyl_acetate | beta.phellandrene | True | -0.287074 | -0.261253 | 0 | 16 |
| K2_partial_spearman | F_malt_cara | 1 | acetaldehyde | furfural | True | 0.241456 | 0.23165 | 0.560177 | 16 |
| K2_partial_spearman | F_malt_cara | 2 | ethyl.butyrate | beta.cyclohomocitral | True | 0.186609 | 0.226601 | 0.525834 | 16 |
| K2_partial_spearman | F_malt_cara | 3 | n.hexanol | octyl_acetate | True | 0.214778 | 0.212805 | 0.546898 | 16 |
| K2_partial_spearman | F_malt_burn | 1 | MeSH | methylpyrazine | True | 0.244636 | 0.151823 | 0 | 16 |
| K2_partial_spearman | F_malt_burn | 2 | ethyl_decanoate | isopropyl.tetradecanoate | True | 0.0585854 | 0.123171 | 0.252213 | 16 |
| K2_partial_spearman | F_malt_burn | 3 | acetaldehyde | furfural | True | 0.0535909 | 0.122942 | 0 | 16 |
| K2_partial_spearman | F_hops_all | 1 | ethyl_octanoate | linalool | True | 0.370991 | 0.319996 | 0.343774 | 16 |
| K2_partial_spearman | F_hops_all | 2 | X.E..2.nonenal | X1.octen.3.ol | True | 0.195219 | 0.314117 | 0.272611 | 16 |
| K2_partial_spearman | F_hops_all | 3 | ethyl_decanoate | X2.undecanol | True | 0.378525 | 0.309459 | 0.0209194 | 16 |
| K2_partial_spearman | F_hops_citrus | 1 | humulene.epoxide.II | trans.dihydro.alpha.terpinyl.acetate | True | 0.244049 | 0.253804 | 0.291516 | 16 |
| K2_partial_spearman | F_hops_citrus | 2 | furfural | nerol.oxide | True | 0.264948 | 0.24811 | 0.175732 | 16 |
| K2_partial_spearman | F_hops_citrus | 3 | ethyl.furoate | geosmin | True | 0.257457 | 0.237503 | 0 | 15 |
| K2_partial_spearman | F_hops_tropical | 1 | geraniol | hexyl.acetate | True | 0.280912 | 0.279118 | 0 | 16 |
| K2_partial_spearman | F_hops_tropical | 2 | hexyl.acetate | terpinolene | True | 0.253406 | 0.267561 | 0.0596466 | 16 |
| K2_partial_spearman | F_hops_tropical | 3 | DMS | geraniol | True | 0.284825 | 0.260247 | 0.151515 | 16 |
| K2_partial_spearman | F_hops_noble | 1 | ethyl_octanoate | ethyl_decanoate | True | 0.457765 | 0.303454 | 0 | 16 |
| K2_partial_spearman | F_hops_noble | 2 | ethyl_decanoate | ethyl_octanoate | True | 0.505088 | 0.294683 | 0.250505 | 16 |
| K2_partial_spearman | F_hops_noble | 3 | X.E..2.nonenal | X2.undecanol | True | 0.33008 | 0.245918 | 0 | 16 |
| K2_partial_spearman | F_hops_woody | 1 | ethyl.tetradecanoate | ethyl.tetradecanoate | False | 0.346064 | 0.285364 | 0.115577 | 16 |
| K2_partial_spearman | F_hops_woody | 2 | ethyl_decanoate | ethyl.hexadecanoate | True | 0.319175 | 0.282656 | 0.399407 | 16 |
| K2_partial_spearman | F_hops_woody | 3 | humulene.epoxide.II | X2.undecanol | True | 0.313974 | 0.252054 | 0.519288 | 16 |
| K2_partial_spearman | F_esters_all | 1 | acetaldehyde | n.hexanol | True | 0.376395 | 0.298549 | 0.346821 | 16 |
| K2_partial_spearman | F_esters_all | 2 | ethyl_acetate | diethyl.succinate | True | -0.0249719 | -0.248116 | 0 | 16 |
| K2_partial_spearman | F_esters_all | 3 | n.hexanol | acetaldehyde | True | 0.403872 | 0.213162 | 0 | 16 |
| K2_partial_spearman | F_esters_ethac | 1 | ethyl_acetate | ethyl_acetate | False | 0.503097 | 0.222966 | 0 | 16 |
| K2_partial_spearman | F_esters_ethac | 2 | isoamyl_alcohol | X2E.6Z.farnesal | True | 0.248748 | 0.194604 | 0.483464 | 16 |
| K2_partial_spearman | F_esters_ethac | 3 | acetaldehyde | E.nerolidol | True | 0.300555 | 0.190763 | 0 | 16 |
| K2_partial_spearman | F_esters_isoaa | 1 | n.hexanol | n.hexanol | False | 0.305394 | 0.279812 | 0.311562 | 16 |
| K2_partial_spearman | F_esters_isoaa | 2 | isopentyl_acetate | diethyl.succinate | True | -0.148069 | -0.275701 | 0.0804842 | 16 |
| K2_partial_spearman | F_esters_isoaa | 3 | acetaldehyde | isopentyl_acetate | True | 0.27363 | 0.221482 | 0.2856 | 16 |
| K2_partial_spearman | F_esters_flower | 1 | beta.phellandrene | beta.phellandrene | False | 0.233631 | 0.253453 | 0.0432652 | 16 |
| K2_partial_spearman | F_esters_flower | 2 | isopentyl.butanoate | terpinolene | True | 0.176798 | 0.225999 | 0.257296 | 16 |
| K2_partial_spearman | F_esters_flower | 3 | terpinolene | X14.hydroxy.alpha.humulene | True | -0.16901 | -0.199219 | 0 | 16 |
| K2_partial_spearman | F_esters_fruity | 1 | DES | DES | False | -0.291595 | -0.279817 | 0.43233 | 16 |
| K2_partial_spearman | F_esters_fruity | 2 | humulene.epoxide.II | beta.phellandrene | True | 0.228069 | 0.247799 | 0.513778 | 16 |
| K2_partial_spearman | F_esters_fruity | 3 | beta.phellandrene | MeSH | True | -0.209065 | -0.209232 | 0 | 16 |
| K2_partial_spearman | acidity | 1 | prenyl.isobutyrate | ethyl.phenylacetate | True | 0.429678 | 0.298147 | 0 | 16 |
| K2_partial_spearman | acidity | 2 | ethyl.phenylacetate | X4.ethyl.guaiacol | True | 0.399826 | 0.287445 | 0 | 16 |
| K2_partial_spearman | acidity | 3 | perillyl.acetate | ethyl_isovalerate | True | 0.346809 | 0.286534 | 0.465146 | 15 |
| K2_partial_spearman | bitternes | 1 | ethyl_octanoate | ethyl_decanoate | True | 0.558887 | 0.417159 | 0.339878 | 16 |
| K2_partial_spearman | bitternes | 2 | ethyl_decanoate | ethyl_octanoate | True | 0.592674 | 0.394638 | 0.364789 | 16 |
| K2_partial_spearman | bitternes | 3 | MeSH | pentyl.octanoate | True | 0.418887 | 0.355993 | 0.389907 | 16 |
| K2_partial_spearman | sweetness | 1 | ethyl_decanoate | ethyl_decanoate | False | -0.391162 | -0.426273 | 0.105541 | 16 |
| K2_partial_spearman | sweetness | 2 | pentyl.octanoate | pentyl.octanoate | False | -0.362518 | -0.358129 | 0.6193 | 16 |
| K2_partial_spearman | sweetness | 3 | X1.2.dihydro.8.hydroxy..2E..linalool | ethyl_octanoate | True | -0.294699 | -0.345992 | 0.139151 | 16 |
| K2_partial_spearman | X4vg | 1 | X4VG | X4VG | False | 0.472618 | 0.477582 | 0 | 15 |
| K2_partial_spearman | X4vg | 2 | delta.terpinyl.acetate | delta.terpinyl.acetate | False | 0.399224 | 0.383315 | 0 | 16 |
| K2_partial_spearman | X4vg | 3 | neryl.formate | ethyl_isovalerate | True | -0.364252 | -0.339548 | 0 | 15 |
| K2_partial_spearman | diacetyl | 1 | eugenol | eugenol | False | -0.323188 | -0.386274 | 0.207141 | 16 |
| K2_partial_spearman | diacetyl | 2 | ethyl_decanoate | ethyl_decanoate | False | -0.282542 | -0.308824 | 0.259164 | 16 |
| K2_partial_spearman | diacetyl | 3 | alpha.calacorene | ethyl_octanoate | True | -0.22731 | -0.293283 | 0 | 16 |
| K2_partial_spearman | dms | 1 | amyl.acetate | terpinen.4.ol | True | -0.195542 | -0.239956 | 0 | 16 |
| K2_partial_spearman | dms | 2 | terpinen.4.ol | beta.phellandrene | True | -0.167851 | -0.216333 | 0 | 16 |
| K2_partial_spearman | dms | 3 | E.nerolidol | E.nerolidol | False | 0.194389 | 0.212477 | 0.238488 | 16 |
| K2_partial_spearman | metallic | 1 | ethyl_isovalerate | ethyl_isovalerate | False | -0.250358 | -0.29437 | 0 | 15 |
| K2_partial_spearman | metallic | 2 | X6.methyl.5.hepten.2.one | X6.methyl.5.hepten.2.one | False | -0.237235 | -0.237595 | 0.0651975 | 11 |
| K2_partial_spearman | metallic | 3 | alpha.calacorene | isopentyl.butanoate | True | -0.228142 | -0.227109 | 0 | 16 |
| K2_partial_spearman | stale_hops | 1 | X2.undecanol | X2.undecanol | False | 0.299674 | 0.310336 | 0 | 16 |
| K2_partial_spearman | stale_hops | 2 | alloaromadendrene.epoxide | alloaromadendrene.epoxide | False | 0.277627 | 0.290792 | 0 | 16 |
| K2_partial_spearman | stale_hops | 3 | neryl.formate | X1.10.di.epi.cubenol | True | 0.234931 | 0.247724 | 0 | 16 |
| K2_partial_spearman | t2n | 1 | beta.phellandrene | terpinen.4.ol | True | -0.203817 | -0.261627 | 0 | 16 |
| K2_partial_spearman | t2n | 2 | terpinen.4.ol | beta.phellandrene | True | -0.214008 | -0.256428 | 0 | 16 |
| K2_partial_spearman | t2n | 3 | geraniol | alpha.phellandrene | True | -0.186344 | -0.218954 | 0.0184736 | 16 |
| K2_partial_spearman | orange | 1 | camphor | camphor | False | 0.367483 | 0.339655 | 0 | 16 |
| K2_partial_spearman | orange | 2 | limonene | limonene | False | 0.296048 | 0.248413 | 0.00564125 | 14 |
| K2_partial_spearman | orange | 3 | beta.phellandrene | beta.phellandrene | False | 0.288663 | 0.245074 | 0 | 16 |
| K2_partial_spearman | coriander | 1 | X2.methyl.benzofuran | cis.beta.terpineol | True | 0.212356 | 0.237364 | 0 | 16 |
| K2_partial_spearman | coriander | 2 | E.ocimenone | alpha.phellandrene | True | 0.223202 | 0.235606 | 0 | 16 |
| K2_partial_spearman | coriander | 3 | terpinen.4.ol | terpinen.4.ol | False | 0.23473 | 0.228175 | 0.127936 | 16 |
| K2_partial_spearman | clove | 1 | limonene | limonene | False | 0.342317 | 0.323725 | 0.275059 | 14 |
| K2_partial_spearman | clove | 2 | eucalyptol | X4VG | True | 0.170276 | 0.25608 | 0.118621 | 15 |
| K2_partial_spearman | clove | 3 | camphor | eucalyptol | True | 0.236407 | 0.225451 | 0 | 16 |
| K2_partial_spearman | lactic | 1 | perillyl.acetate | isopentyl_acetate | True | -0.409885 | -0.370274 | 0.262857 | 16 |
| K2_partial_spearman | lactic | 2 | prenyl.isobutyrate | phenethyl_acetate | True | -0.289545 | -0.3128 | 0.474692 | 16 |
| K2_partial_spearman | lactic | 3 | ethyl.furoate | alpha.humulene | True | 0.186085 | 0.258805 | 0.160563 | 16 |
| K2_partial_spearman | acetic | 1 | prenyl.isobutyrate | isopentyl_acetate | True | -0.308238 | -0.323885 | 0.41776 | 16 |
| K2_partial_spearman | acetic | 2 | ethyl_isovalerate | ethyl_isovalerate | False | 0.390716 | 0.264238 | 0.557416 | 15 |
| K2_partial_spearman | acetic | 3 | ethyl.phenylacetate | phenethyl_acetate | True | -0.230087 | -0.262493 | 0.51741 | 16 |
| K2_partial_spearman | barnyard | 1 | X2.undecanol | X2.undecanol | False | 0.284709 | 0.300468 | 0 | 16 |
| K2_partial_spearman | barnyard | 2 | neryl.formate | alloaromadendrene.epoxide | True | 0.266234 | 0.277003 | 0 | 16 |
| K2_partial_spearman | barnyard | 3 | alloaromadendrene.epoxide | gamma.eudesmol | True | 0.264066 | 0.245692 | 0.53551 | 16 |
| K2_partial_spearman | alcohol | 1 | ethyl_octanoate | ethyl.hexadecanoate | True | 0.370065 | 0.216898 | 0.301527 | 16 |
| K2_partial_spearman | alcohol | 2 | ethyl_hexanoate | isobutanol | True | 0.551923 | 0.183886 | 0 | 16 |
| K2_partial_spearman | alcohol | 3 | acetaldehyde | n.hexanol | True | 0.528476 | 0.181746 | 0.188792 | 16 |
| K2_partial_spearman | aftertaste | 1 | ethyl_octanoate | ethyl_isovalerate | True | 0.439277 | 0.443597 | 0 | 15 |
| K2_partial_spearman | aftertaste | 2 | X.E..2.nonenal | ethyl_2.methyl_butyrate | True | 0.364467 | 0.395552 | 0.441772 | 15 |
| K2_partial_spearman | aftertaste | 3 | ethyl_decanoate | phenethyl_acetate | True | -0.133032 | -0.393348 | 0.249217 | 16 |
| K2_partial_spearman | body | 1 | ethyl_hexanoate | propylene.glycol | True | -0.114681 | -0.209148 | 0.267925 | 16 |
| K2_partial_spearman | body | 2 | X3.methyl.2.hexanol | isobutanol | True | 0.530346 | 0.205766 | 0.216693 | 16 |
| K2_partial_spearman | body | 3 | ethyl_octanoate | linalool | True | 0.20815 | 0.186506 | 0.36995 | 16 |
| K2_partial_spearman | co2 | 1 | ethyl_octanoate | ethyl_octanoate | False | 0.552687 | 0.347909 | 0.428504 | 16 |
| K2_partial_spearman | co2 | 2 | ethyl_hexanoate | ethyl.hexadecanoate | True | 0.354359 | 0.276475 | 0 | 16 |
| K2_partial_spearman | co2 | 3 | ethyl_decanoate | ethyl_decanoate | False | 0.428564 | 0.272826 | 0.499941 | 16 |
| K2_partial_spearman | overall | 1 | ethyl_acetate | pentyl.octanoate | True | 0.231396 | 0.204708 | 0.444277 | 16 |
| K2_partial_spearman | overall | 2 | ethyl_octanoate | ethyl_acetate | True | 0.324999 | 0.193871 | 0.191729 | 16 |
| K2_partial_spearman | overall | 3 | isopentyl_acetate | X2.butyloctanoate | True | 0.223482 | 0.192038 | 0.320732 | 16 |
| K3_random_effects | A_malt_all | 1 | acetaldehyde | furfural | True | 0.259801 | 0.238893 | 0 | 16 |
| K3_random_effects | A_malt_all | 2 | methylpyrazine | n.hexanol | True | 0.205048 | -0.216028 | 0.204117 | 16 |
| K3_random_effects | A_malt_all | 3 | beta.phellandrene | ethyl.hex..2E..enoate | True | -0.157693 | -0.202594 | 0 | 16 |
| K3_random_effects | A_malt_grain | 1 | terpinen.4.ol | DMS | True | -0.155374 | -0.274571 | 0 | 16 |
| K3_random_effects | A_malt_grain | 2 | beta.phellandrene | X1.alpha.terpineol | True | -0.199838 | -0.26896 | 0 | 16 |
| K3_random_effects | A_malt_grain | 3 | isoborneol | terpinen.4.ol | True | -0.264539 | -0.260024 | 0.324579 | 16 |
| K3_random_effects | A_malt_bread | 1 | terpinen.4.ol | dodecanal | True | -0.095801 | -0.190404 | 0 | 16 |
| K3_random_effects | A_malt_bread | 2 | ethyl.tiglate | terpinen.4.ol | True | -0.250225 | -0.18528 | 0 | 16 |
| K3_random_effects | A_malt_bread | 3 | acetaldehyde | neryl.formate | True | -0.225005 | -0.183858 | 0 | 16 |
| K3_random_effects | A_malt_cara | 1 | acetaldehyde | furfural | True | 0.308843 | 0.237392 | 0 | 16 |
| K3_random_effects | A_malt_cara | 2 | X2.ethyl.1.hexanol | X2.ethyl.1.hexanol | False | -0.317247 | -0.20161 | 0 | 16 |
| K3_random_effects | A_malt_cara | 3 | furfural | cis.beta.terpineol | True | -0.0775122 | -0.19458 | 0 | 16 |
| K3_random_effects | A_malt_burn | 1 | isobutanol | X2.acetylfuran | True | 0.122086 | 0.24439 | 0.184894 | 16 |
| K3_random_effects | A_malt_burn | 2 | DES | X1.10.di.epi.cubenol | True | 0.0950438 | 0.241933 | 0 | 16 |
| K3_random_effects | A_malt_burn | 3 | y.methyl.1.octanol | terpinen.4.ol.acetate | True | 0.0859758 | 0.236957 | 0 | 16 |
| K3_random_effects | A_hops_all | 1 | ethyl_octanoate | linalool | True | 0.412072 | 0.326072 | 0.0618418 | 16 |
| K3_random_effects | A_hops_all | 2 | X.E..2.nonenal | ethyl_octanoate | True | 0.463745 | 0.299537 | 0.201967 | 16 |
| K3_random_effects | A_hops_all | 3 | ethyl_decanoate | ethyl.hexadecanoate | True | 0.242299 | 0.298603 | 0.464761 | 16 |
| K3_random_effects | A_hops_citrus | 1 | citronellol | isoborneol | True | 0.227881 | 0.278997 | 0.35462 | 16 |
| K3_random_effects | A_hops_citrus | 2 | humulene.epoxide.II | Z.Z.geranyl.linalool | True | 0.152506 | 0.252746 | 0.394111 | 16 |
| K3_random_effects | A_hops_citrus | 3 | linalool | X.E.E..2.4.decadienal | True | 0.245558 | 0.244106 | 0 | 16 |
| K3_random_effects | A_hops_tropical | 1 | DMS | beta.phellandrene | True | 0.230655 | 0.29549 | 0.133729 | 16 |
| K3_random_effects | A_hops_tropical | 2 | X3.furanmethanol | X.E..beta.ocimene | True | 0.200566 | 0.21512 | 0 | 16 |
| K3_random_effects | A_hops_tropical | 3 | methyl.geranate | terpinolene | True | 0.214897 | 0.206757 | 0 | 16 |
| K3_random_effects | A_hops_noble | 1 | ethyl_octanoate | ethyl_octanoate | False | 0.481829 | 0.355476 | 0 | 16 |
| K3_random_effects | A_hops_noble | 2 | ethyl_decanoate | ethyl_decanoate | False | 0.469279 | 0.278748 | 0 | 16 |
| K3_random_effects | A_hops_noble | 3 | X.E..2.nonenal | DMS | True | 0.329177 | 0.272872 | 0.0851316 | 16 |
| K3_random_effects | A_hops_woody | 1 | trans.dihydro.alpha.terpinyl.acetate | X1.octen.3.ol | True | 0.211473 | 0.314108 | 0.484944 | 16 |
| K3_random_effects | A_hops_woody | 2 | alpha.calacorene | alpha.calacorene | False | 0.272512 | 0.302457 | 0.244241 | 16 |
| K3_random_effects | A_hops_woody | 3 | ethyl.tetradecanoate | neryl.propanoate | True | 0.264465 | 0.298772 | 0.00862325 | 16 |
| K3_random_effects | A_esters_all | 1 | n.hexanol | isopentyl_acetate | True | 0.292396 | 0.352933 | 0.256784 | 16 |
| K3_random_effects | A_esters_all | 2 | acetaldehyde | acetaldehyde | False | 0.35303 | 0.319656 | 0 | 16 |
| K3_random_effects | A_esters_all | 3 | ethyl_acetate | n.hexanol | True | 0.356852 | 0.271207 | 0.125181 | 16 |
| K3_random_effects | A_esters_ethac | 1 | ethyl_acetate | isobutyl_acetate | True | 0.309747 | 0.263478 | 0.498938 | 16 |
| K3_random_effects | A_esters_ethac | 2 | acetaldehyde | isopentyl_acetate | True | 0.261508 | 0.240193 | 0.307533 | 16 |
| K3_random_effects | A_esters_ethac | 3 | isoamyl_alcohol | X.Z..beta.farnesene | True | 0.217257 | 0.233995 | 0 | 16 |
| K3_random_effects | A_esters_isoaa | 1 | n.hexanol | isopentyl_acetate | True | 0.282815 | 0.286452 | 0 | 16 |
| K3_random_effects | A_esters_isoaa | 2 | isopentyl_acetate | n.hexanol | True | 0.33993 | 0.286374 | 0 | 16 |
| K3_random_effects | A_esters_isoaa | 3 | isoamyl_alcohol | tau.cadinol | True | -0.20927 | -0.246686 | 0 | 16 |
| K3_random_effects | A_esters_flower | 1 | beta.phellandrene | terpinen.4.ol | True | 0.145468 | 0.242475 | 0.396399 | 16 |
| K3_random_effects | A_esters_flower | 2 | isobutyl_acetate | isoborneol | True | 0.144297 | 0.204473 | 0 | 16 |
| K3_random_effects | A_esters_flower | 3 | X1.alpha.terpineol | X3.methyl.2.butenal | True | 0.140705 | 0.202458 | 0.608519 | 16 |
| K3_random_effects | A_esters_fruity | 1 | beta.phellandrene | X1.octen.3.ol | True | -0.00760757 | -0.218818 | 0.457247 | 16 |
| K3_random_effects | A_esters_fruity | 2 | eugenol | methyl.hexadecanoate | True | 0.0829264 | 0.20776 | 0.0552886 | 16 |
| K3_random_effects | A_esters_fruity | 3 | benzaldehyde | nerol | True | -0.037419 | -0.194846 | 0 | 16 |
| K3_random_effects | F_malt_all | 1 | acetaldehyde | X.E.E..2.4.decadienal | True | -0.0406663 | -0.243194 | 0.22339 | 16 |
| K3_random_effects | F_malt_all | 2 | MeSH | X1.alpha.terpineol | True | -0.11089 | -0.233814 | 0 | 16 |
| K3_random_effects | F_malt_all | 3 | ethyl.butyrate | ethyl..E..linalool | True | -0.300028 | -0.2171 | 0.241113 | 16 |
| K3_random_effects | F_malt_grain | 1 | isoborneol | X.E..2.nonenal | True | -0.212208 | -0.295632 | 0.252194 | 16 |
| K3_random_effects | F_malt_grain | 2 | terpinen.4.ol.acetate | cis.dihydro.beta.terpineol | True | -0.159754 | -0.25564 | 0.512528 | 9 |
| K3_random_effects | F_malt_grain | 3 | X.E..beta.ionone | isoborneol | True | -0.275786 | -0.221263 | 0.119458 | 16 |
| K3_random_effects | F_malt_bread | 1 | acetaldehyde | terpinen.4.ol | True | -0.315758 | -0.250229 | 0 | 16 |
| K3_random_effects | F_malt_bread | 2 | ethyl.butyrate | ethyl_isovalerate | True | -0.0582543 | -0.246753 | 0.607315 | 15 |
| K3_random_effects | F_malt_bread | 3 | isopentyl_acetate | beta.phellandrene | True | -0.287074 | -0.230408 | 0 | 16 |
| K3_random_effects | F_malt_cara | 1 | acetaldehyde | isobutyl_acetate | True | 0.223546 | -0.182834 | 0 | 16 |
| K3_random_effects | F_malt_cara | 2 | ethyl.butyrate | X2.furanmethanol | True | -0.0950639 | 0.181838 | 0.357528 | 16 |
| K3_random_effects | F_malt_cara | 3 | n.hexanol | pentyl.octanoate | True | 0.00886478 | -0.178824 | 0.554793 | 16 |
| K3_random_effects | F_malt_burn | 1 | MeSH | DES | True | 0.294187 | 0.535488 | 0.94285 | 16 |
| K3_random_effects | F_malt_burn | 2 | ethyl_decanoate | geraniol | True | 0.0300554 | -0.246034 | 0.312384 | 16 |
| K3_random_effects | F_malt_burn | 3 | acetaldehyde | beta.phellandrene | True | -0.15085 | -0.215613 | 0.0655519 | 16 |
| K3_random_effects | F_hops_all | 1 | ethyl_octanoate | ethyl_isovalerate | True | 0.117607 | 0.388534 | 0 | 15 |
| K3_random_effects | F_hops_all | 2 | X.E..2.nonenal | ethyl_octanoate | True | 0.500767 | 0.354147 | 0 | 16 |
| K3_random_effects | F_hops_all | 3 | ethyl_decanoate | octyl_acetate | True | -0.284514 | -0.350612 | 0 | 16 |
| K3_random_effects | F_hops_citrus | 1 | humulene.epoxide.II | phenylacetaldehyde | True | -0.0200789 | -0.250195 | 0 | 16 |
| K3_random_effects | F_hops_citrus | 2 | furfural | meta.cymenene | True | -0.0884819 | -0.232643 | 0.579916 | 15 |
| K3_random_effects | F_hops_citrus | 3 | ethyl.furoate | isoborneol | True | 0.157648 | 0.204002 | 0 | 16 |
| K3_random_effects | F_hops_tropical | 1 | geraniol | isopentyl.hexanoate | True | 0.104558 | 0.451394 | 0.945557 | 16 |
| K3_random_effects | F_hops_tropical | 2 | hexyl.acetate | isoborneol | True | 0.123672 | 0.258224 | 0.523469 | 16 |
| K3_random_effects | F_hops_tropical | 3 | DMS | geraniol | True | 0.284825 | 0.238906 | 0.151515 | 16 |
| K3_random_effects | F_hops_noble | 1 | ethyl_octanoate | ethyl_octanoate | False | 0.505088 | 0.449281 | 0.250505 | 16 |
| K3_random_effects | F_hops_noble | 2 | ethyl_decanoate | octyl_acetate | True | -0.258664 | -0.353377 | 0.374746 | 16 |
| K3_random_effects | F_hops_noble | 3 | X.E..2.nonenal | ethyl_decanoate | True | 0.457765 | 0.326094 | 0 | 16 |
| K3_random_effects | F_hops_woody | 1 | ethyl.tetradecanoate | ethyl.hexadecanoate | True | 0.319175 | 0.319184 | 0.399407 | 16 |
| K3_random_effects | F_hops_woody | 2 | ethyl_decanoate | ethyl.tetradecanoate | True | 0.346064 | 0.30554 | 0.115577 | 16 |
| K3_random_effects | F_hops_woody | 3 | humulene.epoxide.II | Z.Z.geranyl.linalool | True | 0.203203 | 0.260023 | 0.176104 | 16 |
| K3_random_effects | F_esters_all | 1 | acetaldehyde | isopentyl_acetate | True | 0.310518 | 0.344299 | 0.488596 | 16 |
| K3_random_effects | F_esters_all | 2 | ethyl_acetate | phenethyl_alcohol | True | 0.215752 | 0.272977 | 0 | 16 |
| K3_random_effects | F_esters_all | 3 | n.hexanol | n.hexanol | False | 0.376395 | 0.272643 | 0.346821 | 16 |
| K3_random_effects | F_esters_ethac | 1 | ethyl_acetate | ethyl_acetate | False | 0.503097 | 0.266435 | 0 | 16 |
| K3_random_effects | F_esters_ethac | 2 | isoamyl_alcohol | E.nerolidol | True | 0.300555 | 0.232958 | 0 | 16 |
| K3_random_effects | F_esters_ethac | 3 | acetaldehyde | Z.Z.geranyl.linalool | True | 0.341737 | 0.231345 | 0 | 16 |
| K3_random_effects | F_esters_isoaa | 1 | n.hexanol | isopentyl_acetate | True | 0.27363 | 0.269005 | 0.2856 | 16 |
| K3_random_effects | F_esters_isoaa | 2 | isopentyl_acetate | n.hexanol | True | 0.305394 | 0.227191 | 0.311562 | 16 |
| K3_random_effects | F_esters_isoaa | 3 | acetaldehyde | isoborneol | True | -0.110965 | -0.222421 | 0 | 16 |
| K3_random_effects | F_esters_flower | 1 | beta.phellandrene | terpinolene | True | 0.176798 | 0.253715 | 0.257296 | 16 |
| K3_random_effects | F_esters_flower | 2 | isopentyl.butanoate | ethyl_2.methyl_butyrate | True | -0.0674929 | -0.240072 | 0.233509 | 15 |
| K3_random_effects | F_esters_flower | 3 | terpinolene | camphor | True | 0.0963722 | 0.210741 | 0 | 16 |
| K3_random_effects | F_esters_fruity | 1 | DES | X2.ethyl.3.methylpyrazine | True | -0.0238787 | -0.225395 | 0.325214 | 16 |
| K3_random_effects | F_esters_fruity | 2 | humulene.epoxide.II | ethyl_2.methyl_butyrate | True | 0.00232327 | -0.205368 | 0 | 15 |
| K3_random_effects | F_esters_fruity | 3 | beta.phellandrene | n.octanol | True | -0.0856801 | -0.191252 | 0.278095 | 16 |
| K3_random_effects | acidity | 1 | prenyl.isobutyrate | eugenol | True | 0.218769 | 0.287296 | 0.0926713 | 16 |
| K3_random_effects | acidity | 2 | ethyl.phenylacetate | X.E..beta.damascenone | True | 0.187422 | 0.274031 | 0.168269 | 16 |
| K3_random_effects | acidity | 3 | perillyl.acetate | ethyl.tetradecanoate | True | 0.0331127 | 0.245467 | 0 | 16 |
| K3_random_effects | bitternes | 1 | ethyl_octanoate | ethyl_octanoate | False | 0.592674 | 0.500951 | 0.364789 | 16 |
| K3_random_effects | bitternes | 2 | ethyl_decanoate | ethyl_decanoate | False | 0.558887 | 0.444241 | 0.339878 | 16 |
| K3_random_effects | bitternes | 3 | MeSH | ethyl_2.methyl_butyrate | True | 0.0514829 | 0.352459 | 0.67032 | 15 |
| K3_random_effects | sweetness | 1 | ethyl_decanoate | ethyl_decanoate | False | -0.391162 | -0.329787 | 0.105541 | 16 |
| K3_random_effects | sweetness | 2 | pentyl.octanoate | ethyl_octanoate | True | -0.294699 | -0.30232 | 0.139151 | 16 |
| K3_random_effects | sweetness | 3 | X1.2.dihydro.8.hydroxy..2E..linalool | X2.methylbutyl.octanoate | True | -0.329487 | -0.285411 | 0.528372 | 16 |
| K3_random_effects | X4vg | 1 | X4VG | X4VG | False | 0.472618 | 0.428297 | 0 | 15 |
| K3_random_effects | X4vg | 2 | delta.terpinyl.acetate | delta.terpinyl.acetate | False | 0.399224 | 0.289662 | 0 | 16 |
| K3_random_effects | X4vg | 3 | neryl.formate | X2.methyltetrahydrofuran.3.one | True | 0.121682 | 0.25831 | 0.427888 | 16 |
| K3_random_effects | diacetyl | 1 | eugenol | ethyl_decanoate | True | -0.282542 | -0.440131 | 0.259164 | 16 |
| K3_random_effects | diacetyl | 2 | ethyl_decanoate | ethyl_octanoate | True | -0.22731 | -0.396109 | 0 | 16 |
| K3_random_effects | diacetyl | 3 | alpha.calacorene | pentyl.octanoate | True | -0.228155 | -0.297825 | 0 | 16 |
| K3_random_effects | dms | 1 | amyl.acetate | ethyl.octadecanoate | True | 0.069758 | 0.257343 | 0.620282 | 16 |
| K3_random_effects | dms | 2 | terpinen.4.ol | X9.epi..E..caryophyllene | True | 0.15524 | 0.246163 | 0.528051 | 16 |
| K3_random_effects | dms | 3 | E.nerolidol | X4.methyl.1.pentanol | True | 0.127766 | 0.241342 | 0.419112 | 16 |
| K3_random_effects | metallic | 1 | ethyl_isovalerate | neryl.isobutanoate | True | -0.104121 | -0.233437 | 0 | 16 |
| K3_random_effects | metallic | 2 | X6.methyl.5.hepten.2.one | delta.terpinyl.acetate | True | 0.186719 | 0.209568 | 0.391865 | 16 |
| K3_random_effects | metallic | 3 | alpha.calacorene | gamma.eudesmol | True | -0.111372 | -0.20472 | 0 | 16 |
| K3_random_effects | stale_hops | 1 | X2.undecanol | X.E..ortho.methoxy.cinnamaldehyde | True | 0.248252 | 0.539461 | 0.941475 | 16 |
| K3_random_effects | stale_hops | 2 | alloaromadendrene.epoxide | CS2 | True | -0.0562545 | 0.448011 | 0.953799 | 14 |
| K3_random_effects | stale_hops | 3 | neryl.formate | beta.humulene | True | -0.0547927 | 0.402438 | 0.947406 | 15 |
| K3_random_effects | t2n | 1 | beta.phellandrene | vinyl.caprylate | True | 0.175324 | 0.242274 | 0 | 16 |
| K3_random_effects | t2n | 2 | terpinen.4.ol | alpha.phellandrene | True | -0.186344 | -0.21016 | 0.0184736 | 16 |
| K3_random_effects | t2n | 3 | geraniol | X6.methyl.5.hepten.2.one | True | -0.172676 | -0.194334 | 0 | 11 |
| K3_random_effects | orange | 1 | camphor | camphor | False | 0.367483 | 0.403619 | 0 | 16 |
| K3_random_effects | orange | 2 | limonene | beta.phellandrene | True | 0.288663 | 0.353226 | 0 | 16 |
| K3_random_effects | orange | 3 | beta.phellandrene | ethyl..E..linalool | True | 0.271519 | 0.3333 | 0.244437 | 16 |
| K3_random_effects | coriander | 1 | X2.methyl.benzofuran | cis.beta.terpineol | True | 0.212356 | 0.264506 | 0 | 16 |
| K3_random_effects | coriander | 2 | E.ocimenone | hexyl.acetate | True | 0.207794 | 0.258321 | 0.217006 | 16 |
| K3_random_effects | coriander | 3 | terpinen.4.ol | isopentyl.formate | True | 0.161166 | 0.24026 | 0.424632 | 16 |
| K3_random_effects | clove | 1 | limonene | limonene | False | 0.342317 | 0.247209 | 0.275059 | 14 |
| K3_random_effects | clove | 2 | eucalyptol | cis.dihydro.beta.terpineol | True | -0.189193 | -0.229927 | 0 | 9 |
| K3_random_effects | clove | 3 | camphor | eucalyptol | True | 0.236407 | 0.226798 | 0 | 16 |
| K3_random_effects | lactic | 1 | perillyl.acetate | eucalyptol | True | 0.0931878 | 0.396605 | 0.944482 | 16 |
| K3_random_effects | lactic | 2 | prenyl.isobutyrate | isopentyl_acetate | True | -0.409885 | -0.237251 | 0.262857 | 16 |
| K3_random_effects | lactic | 3 | ethyl.furoate | X2Z.6Z.farnesol | True | -0.00751559 | -0.228242 | 0.470954 | 16 |
| K3_random_effects | acetic | 1 | prenyl.isobutyrate | beta.humulene | True | -0.041627 | 0.526943 | 0.946489 | 15 |
| K3_random_effects | acetic | 2 | ethyl_isovalerate | CS2 | True | 0.12923 | 0.519743 | 0.949877 | 14 |
| K3_random_effects | acetic | 3 | ethyl.phenylacetate | X.E..ortho.methoxy.cinnamaldehyde | True | 0.220519 | 0.331506 | 0.944402 | 16 |
| K3_random_effects | barnyard | 1 | X2.undecanol | X.E..ortho.methoxy.cinnamaldehyde | True | 0.223806 | 0.494206 | 0.941625 | 16 |
| K3_random_effects | barnyard | 2 | neryl.formate | CS2 | True | -0.056444 | 0.435751 | 0.953577 | 14 |
| K3_random_effects | barnyard | 3 | alloaromadendrene.epoxide | beta.humulene | True | -0.0860883 | 0.367541 | 0.94711 | 15 |
| K3_random_effects | alcohol | 1 | ethyl_octanoate | ethyl_octanoate | False | 0.619371 | 0.413183 | 0 | 16 |
| K3_random_effects | alcohol | 2 | ethyl_hexanoate | ethyl_acetate | True | 0.578353 | 0.386957 | 0 | 16 |
| K3_random_effects | alcohol | 3 | acetaldehyde | ethyl_decanoate | True | 0.479907 | 0.343368 | 0 | 16 |
| K3_random_effects | aftertaste | 1 | ethyl_octanoate | ethyl_isovalerate | True | 0.439277 | 0.402883 | 0 | 15 |
| K3_random_effects | aftertaste | 2 | X.E..2.nonenal | ethyl_octanoate | True | 0.506179 | 0.37573 | 0.456754 | 16 |
| K3_random_effects | aftertaste | 3 | ethyl_decanoate | eugenol | True | 0.294196 | 0.319099 | 0.254506 | 16 |
| K3_random_effects | body | 1 | ethyl_hexanoate | X2Z.6E.farnesol | True | 0.0599927 | -0.328328 | 0.609468 | 16 |
| K3_random_effects | body | 2 | X3.methyl.2.hexanol | X3.methyl.2.hexanol | False | -0.546826 | -0.294745 | 0.299183 | 16 |
| K3_random_effects | body | 3 | ethyl_octanoate | ethyl_hexanoate | True | 0.554674 | 0.272689 | 0.0127917 | 16 |
| K3_random_effects | co2 | 1 | ethyl_octanoate | ethyl_octanoate | False | 0.552687 | 0.345472 | 0.428504 | 16 |
| K3_random_effects | co2 | 2 | ethyl_hexanoate | ethyl_hexanoate | False | 0.467309 | 0.302201 | 0.525013 | 16 |
| K3_random_effects | co2 | 3 | ethyl_decanoate | octyl_acetate | True | -0.16414 | -0.286403 | 0 | 16 |
| K3_random_effects | overall | 1 | ethyl_acetate | ethyl_acetate | False | 0.324999 | 0.291123 | 0.191729 | 16 |
| K3_random_effects | overall | 2 | ethyl_octanoate | linalool.isovalerate | True | -0.140501 | -0.234267 | 0.205841 | 16 |
| K3_random_effects | overall | 3 | isopentyl_acetate | X.E.E..2.4.decadienal | True | 0.160674 | 0.201191 | 0 | 16 |

## dimensions

| method | dim | jaccard |
| --- | --- | --- |
| K1_style_demean | A_malt_all | 0 |
| K1_style_demean | A_malt_grain | 0.2 |
| K1_style_demean | A_malt_bread | 0 |
| K1_style_demean | A_malt_cara | 0.2 |
| K1_style_demean | A_malt_burn | 0.2 |
| K1_style_demean | A_hops_all | 0.2 |
| K1_style_demean | A_hops_citrus | 0.2 |
| K1_style_demean | A_hops_tropical | 0 |
| K1_style_demean | A_hops_noble | 0.5 |
| K1_style_demean | A_hops_woody | 0.2 |
| K1_style_demean | A_esters_all | 0.5 |
| K1_style_demean | A_esters_ethac | 0.2 |
| K1_style_demean | A_esters_isoaa | 0.5 |
| K1_style_demean | A_esters_flower | 0.2 |
| K1_style_demean | A_esters_fruity | 0 |
| K1_style_demean | F_malt_all | 0 |
| K1_style_demean | F_malt_grain | 0.5 |
| K1_style_demean | F_malt_bread | 0 |
| K1_style_demean | F_malt_cara | 0 |
| K1_style_demean | F_malt_burn | 0 |
| K1_style_demean | F_hops_all | 0.2 |
| K1_style_demean | F_hops_citrus | 0 |
| K1_style_demean | F_hops_tropical | 0.2 |
| K1_style_demean | F_hops_noble | 0.5 |
| K1_style_demean | F_hops_woody | 0.2 |
| K1_style_demean | F_esters_all | 0.2 |
| K1_style_demean | F_esters_ethac | 0.5 |
| K1_style_demean | F_esters_isoaa | 0.2 |
| K1_style_demean | F_esters_flower | 0 |
| K1_style_demean | F_esters_fruity | 0 |
| K1_style_demean | acidity | 0 |
| K1_style_demean | bitternes | 0.5 |
| K1_style_demean | sweetness | 0.2 |
| K1_style_demean | X4vg | 0.5 |
| K1_style_demean | diacetyl | 0.2 |
| K1_style_demean | dms | 0 |
| K1_style_demean | metallic | 0.2 |
| K1_style_demean | stale_hops | 0 |
| K1_style_demean | t2n | 0 |
| K1_style_demean | orange | 0.2 |
| K1_style_demean | coriander | 0 |
| K1_style_demean | clove | 0.2 |
| K1_style_demean | lactic | 0 |
| K1_style_demean | acetic | 0 |
| K1_style_demean | barnyard | 0 |
| K1_style_demean | alcohol | 0.2 |
| K1_style_demean | aftertaste | 0.5 |
| K1_style_demean | body | 0.5 |
| K1_style_demean | co2 | 0.5 |
| K1_style_demean | overall | 0.5 |
| K2_partial_spearman | A_malt_all | 0.2 |
| K2_partial_spearman | A_malt_grain | 1 |
| K2_partial_spearman | A_malt_bread | 0 |
| K2_partial_spearman | A_malt_cara | 0.5 |
| K2_partial_spearman | A_malt_burn | 0.2 |
| K2_partial_spearman | A_hops_all | 0.5 |
| K2_partial_spearman | A_hops_citrus | 0.5 |
| K2_partial_spearman | A_hops_tropical | 0.2 |
| K2_partial_spearman | A_hops_noble | 0.5 |
| K2_partial_spearman | A_hops_woody | 0.2 |
| K2_partial_spearman | A_esters_all | 0.5 |
| K2_partial_spearman | A_esters_ethac | 0 |
| K2_partial_spearman | A_esters_isoaa | 0.5 |
| K2_partial_spearman | A_esters_flower | 0.2 |
| K2_partial_spearman | A_esters_fruity | 0.5 |
| K2_partial_spearman | F_malt_all | 0 |
| K2_partial_spearman | F_malt_grain | 0.2 |
| K2_partial_spearman | F_malt_bread | 0 |
| K2_partial_spearman | F_malt_cara | 0 |
| K2_partial_spearman | F_malt_burn | 0 |
| K2_partial_spearman | F_hops_all | 0 |
| K2_partial_spearman | F_hops_citrus | 0 |
| K2_partial_spearman | F_hops_tropical | 0.5 |
| K2_partial_spearman | F_hops_noble | 0.5 |
| K2_partial_spearman | F_hops_woody | 0.2 |
| K2_partial_spearman | F_esters_all | 0.5 |
| K2_partial_spearman | F_esters_ethac | 0.2 |
| K2_partial_spearman | F_esters_isoaa | 0.5 |
| K2_partial_spearman | F_esters_flower | 0.5 |
| K2_partial_spearman | F_esters_fruity | 0.5 |
| K2_partial_spearman | acidity | 0.2 |
| K2_partial_spearman | bitternes | 0.5 |
| K2_partial_spearman | sweetness | 0.5 |
| K2_partial_spearman | X4vg | 0.5 |
| K2_partial_spearman | diacetyl | 0.5 |
| K2_partial_spearman | dms | 0.5 |
| K2_partial_spearman | metallic | 0.5 |
| K2_partial_spearman | stale_hops | 0.5 |
| K2_partial_spearman | t2n | 0.5 |
| K2_partial_spearman | orange | 1 |
| K2_partial_spearman | coriander | 0.2 |
| K2_partial_spearman | clove | 0.5 |
| K2_partial_spearman | lactic | 0 |
| K2_partial_spearman | acetic | 0.2 |
| K2_partial_spearman | barnyard | 0.5 |
| K2_partial_spearman | alcohol | 0 |
| K2_partial_spearman | aftertaste | 0 |
| K2_partial_spearman | body | 0 |
| K2_partial_spearman | co2 | 0.5 |
| K2_partial_spearman | overall | 0.2 |
| K3_random_effects | A_malt_all | 0 |
| K3_random_effects | A_malt_grain | 0.2 |
| K3_random_effects | A_malt_bread | 0.2 |
| K3_random_effects | A_malt_cara | 0.5 |
| K3_random_effects | A_malt_burn | 0 |
| K3_random_effects | A_hops_all | 0.2 |
| K3_random_effects | A_hops_citrus | 0 |
| K3_random_effects | A_hops_tropical | 0 |
| K3_random_effects | A_hops_noble | 0.5 |
| K3_random_effects | A_hops_woody | 0.2 |
| K3_random_effects | A_esters_all | 0.5 |
| K3_random_effects | A_esters_ethac | 0 |
| K3_random_effects | A_esters_isoaa | 0.5 |
| K3_random_effects | A_esters_flower | 0 |
| K3_random_effects | A_esters_fruity | 0 |
| K3_random_effects | F_malt_all | 0 |
| K3_random_effects | F_malt_grain | 0.2 |
| K3_random_effects | F_malt_bread | 0 |
| K3_random_effects | F_malt_cara | 0 |
| K3_random_effects | F_malt_burn | 0 |
| K3_random_effects | F_hops_all | 0.2 |
| K3_random_effects | F_hops_citrus | 0 |
| K3_random_effects | F_hops_tropical | 0.2 |
| K3_random_effects | F_hops_noble | 0.5 |
| K3_random_effects | F_hops_woody | 0.2 |
| K3_random_effects | F_esters_all | 0.2 |
| K3_random_effects | F_esters_ethac | 0.2 |
| K3_random_effects | F_esters_isoaa | 0.5 |
| K3_random_effects | F_esters_flower | 0.2 |
| K3_random_effects | F_esters_fruity | 0 |
| K3_random_effects | acidity | 0 |
| K3_random_effects | bitternes | 0.5 |
| K3_random_effects | sweetness | 0.2 |
| K3_random_effects | X4vg | 0.5 |
| K3_random_effects | diacetyl | 0.2 |
| K3_random_effects | dms | 0 |
| K3_random_effects | metallic | 0 |
| K3_random_effects | stale_hops | 0 |
| K3_random_effects | t2n | 0 |
| K3_random_effects | orange | 0.5 |
| K3_random_effects | coriander | 0 |
| K3_random_effects | clove | 0.5 |
| K3_random_effects | lactic | 0 |
| K3_random_effects | acetic | 0 |
| K3_random_effects | barnyard | 0 |
| K3_random_effects | alcohol | 0.2 |
| K3_random_effects | aftertaste | 0.2 |
| K3_random_effects | body | 0.5 |
| K3_random_effects | co2 | 0.5 |
| K3_random_effects | overall | 0.2 |

## 结论与限制

原始对照采用同一候选池的单变量Top3，避免把综合排序变化混入去混杂效应。K1为原值风格内去均值后Spearman；K2为秩残差偏相关；K3为样本≥8风格的Fisher-z随机效应合并（Spearman方差近似n−3）。常数变量组不计入权重，少于2个有效风格的I²不定义。K1平均Jaccard=0.1960；席位变化不能被解释为因果关系。K2协变量自身的残差方差为0，不参与化合物排名；近常数列的相关系数记0。
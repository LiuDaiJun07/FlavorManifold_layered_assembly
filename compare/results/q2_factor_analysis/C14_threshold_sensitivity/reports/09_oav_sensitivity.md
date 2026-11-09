# C14 OAV来源敏感性：前提不足

运行协议：FOLDS=10, TREE_SCALE=1.0, 当前并行任务上限=6（缓存可复用同参数的先前计算）。

检验为探索性：目标维度相关、CV训练集重叠，p值不可当独立重复证据；弱/强维为事后子集。

## sources

| compound | source_id | source_url | threshold | threshold_unit | matrix | concentration_unit | conversion_verified | concentration_scale | independence_group |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

## 结论与限制

未执行数值OAV稳健性检验：现有阈值表混合单位，原数据浓度单位未逐列证实，且没有两套独立、同基质且单位可核验的阈值。threshold_sources.csv仅为待核验数据结构，不把阈值倍数扰动冒充独立来源。检索记录见threshold_source_candidates.md。脚本已实现输入齐备后的单位换算、来源共同覆盖集Jaccard与翻转表；independence_group应填写原始阈值测量研究，不能用转载文献伪装独立来源。
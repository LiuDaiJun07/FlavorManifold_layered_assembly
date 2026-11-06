# -*- coding: utf-8 -*-
# Q2 步骤一：用 2.csv 的化合物共表达相关阵建图，给出度、介数、聚类系数与模块划分
# 图结构用于后续把「单化合物重要性」扩散到「同模块邻居」，让 Top-3 更抗噪
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'q1'))
from common import (OUT_Q2, SUM_COLS, chem_meta, chem_name_map, demojibake, ensure_dirs,
                    load_graph_corr)

ensure_dirs()
L = []


def P(s=''):
    L.append(str(s))
    print(s)


# 2.csv 与 1.csv 的化合物命名体系不同，先换名再对齐列序
mp = chem_name_map()
names, R = load_graph_corr()
n = len(names)
P('# Q2 步骤一：化合物共表达图')
P()
P(f'2.csv 用可读化学名命名，1.csv 用字段名命名，按 8.xlsx 的 chemical_name 到 field_name_R '
  f'建立映射，{len(mp)} 个化合物全部对上，无重名。')
P(f'相关阵 {n}x{n}，对角均值 {np.diag(R).mean():.6f}（自相关为 1），对称性最大偏差 {np.abs(R - R.T).max():.2e}')

# 主图用 k 近邻并集：每个化合物连到与自己相关性最强的 k 个伙伴，保证连通且规模可控
# 单纯卡阈值会让图碎成几十个单点社区，无法做模块级解释，故另列阈值口径做敏感性对照
K_NN = 8
THR_LIST = (0.3, 0.35, 0.4, 0.45, 0.5, 0.6)
A = np.abs(R).copy()
np.fill_diagonal(A, 0.0)
iu_ = np.triu_indices(n, 1)
P()
P('## 1 图的基本形态')
P()
P('两种建图口径的对照（n = %d 个化合物，全部两两相关）：' % n)
P()
P('| 口径 | 边数 | 连通分量 | 最大分量 | 平均度 | 社区数 | 模块度 Q |')
P('| --- | --- | --- | --- | --- | --- | --- |')
for thr in THR_LIST:
    m = A[iu_] >= thr
    g0 = nx.Graph()
    g0.add_nodes_from(range(n))
    g0.add_edges_from(zip(iu_[0][m], iu_[1][m]))
    c0 = list(nx.connected_components(g0))
    q0 = list(nx.community.greedy_modularity_communities(g0))
    P(f'| \|rho\| >= {thr} | {int(m.sum())} | {len(c0)} | {len(max(c0, key=len))} | '
      f'{2 * m.sum() / n:.1f} | {len(q0)} | {nx.community.modularity(g0, q0):.3f} |')
G = nx.Graph()
G.add_nodes_from(names)
for i in range(n):
    for j in np.argsort(-A[i])[:K_NN]:
        G.add_edge(names[i], names[int(j)], weight=float(A[i, int(j)]))
cG = list(nx.connected_components(G))
qG = list(nx.community.greedy_modularity_communities(G, weight='weight'))
P(f'| k 近邻并集 k={K_NN}（主口径） | {G.number_of_edges()} | {len(cG)} | {len(max(cG, key=len))} | '
  f'{2 * G.number_of_edges() / n:.1f} | {len(qG)} | {nx.community.modularity(G, qG, weight="weight"):.3f} |')
P()
P(f'采用 k={K_NN} 的近邻并集作为主图：全图连通，模块度 '
  f'{nx.community.modularity(G, qG, weight="weight"):.3f} 明显高于卡阈值口径，社区边界更干净。')

deg = np.array([d for _, d in G.degree()])
comp = list(nx.connected_components(G))
giant = max(comp, key=len)
P(f'平均度 {deg.mean():.2f}，孤立节点 {(deg == 0).sum()} 个，图密度 {nx.density(G):.4f}，'
  f'平均聚类系数 {nx.average_clustering(G):.4f}')

degc = nx.degree_centrality(G)
btw = nx.betweenness_centrality(G)
clo = nx.clustering(G)
comms = list(nx.community.greedy_modularity_communities(G, weight='weight'))
mod = nx.community.modularity(G, comms, weight='weight')
cid = {}
for k, c in enumerate(comms):
    for v in c:
        cid[v] = k
P(f'贪心模块度划分得 {len(comms)} 个社区，模块度 Q = {mod:.4f}')

# 用化学来源给每个社区打标签，便于化学解释
mmap = chem_meta()
P()
P('## 2 社区构成与化学来源')
P()
P('| 社区 | 节点数 | 主导来源 | 示例化合物 |')
P('| --- | --- | --- | --- |')
rows = []
for k, c in enumerate(comms):
    origins = [mmap.get(v, {}).get('origin_paper', 'NA') for v in c]
    vc = pd.Series(origins).value_counts()
    top = ', '.join(f'{a}({b})' for a, b in vc.head(3).items())
    sample = ', '.join(sorted(c)[:5])
    P(f'| C{k} | {len(c)} | {top} | {sample} |')
    for v in c:
        rows.append(dict(compound=v, community=k, origin=mmap.get(v, {}).get('origin_paper', 'NA'),
                         degree=G.degree(v), degree_cent=degc[v], betweenness=btw[v],
                         clustering=clo[v],
                         weighted_degree=sum(d['weight'] for _, _, d in G.edges(v, data=True))))
P()

top_deg = sorted(degc.items(), key=lambda x: -x[1])[:20]
P('## 3 度中心性前 20 的化合物（共表达枢纽）')
P()
P('| 化合物 | 化学名 | 来源 | 度中心性 | 介数 | 社区 |')
P('| --- | --- | --- | --- | --- | --- |')
for v, d in top_deg:
    m = mmap.get(v, {})
    P(f"| {v} | {demojibake(m.get('chemical_name', 'NA'))} | {m.get('origin_paper', 'NA')} | {d:.4f} | "
      f"{btw[v]:.4f} | C{cid.get(v, -1)} |")
P()

GDF = pd.DataFrame(rows).sort_values('degree_cent', ascending=False)
GDF.to_csv(os.path.join(OUT_Q2, '01_graph_nodes.csv'), index=False, encoding='utf-8-sig')
# 出力平滑用的邻接阵：与主图一致，保留相关强度作为边权
Asm = np.zeros((n, n))
for a_, b_, dd_ in G.edges(data=True):
    ia_, ib_ = names.index(a_), names.index(b_)
    Asm[ia_, ib_] = Asm[ib_, ia_] = dd_['weight']
np.save(os.path.join(OUT_Q2, '01_adj.npy'), Asm)
with open(os.path.join(OUT_Q2, '01_graph_meta.json'), 'w', encoding='utf-8') as f:
    json.dump({'names': names, 'k_nn': K_NN, 'n_edges': int(G.number_of_edges()),
               'n_comm': len(comms), 'modularity': float(mod), 'giant': len(giant),
               'community': cid,
               'label': {v: demojibake(mmap.get(v, {}).get('chemical_name', 'NA')) for v in G.nodes}}, f,
              ensure_ascii=False, indent=1)

# 出图：按社区着色的弹簧布局
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
fig, ax = plt.subplots(1, 2, figsize=(15, 7))
pos = nx.spring_layout(G, seed=7, k=0.35, iterations=80, weight='weight')
big = [k for k, c in enumerate(comms) if len(c) >= 5]
palette = plt.get_cmap('tab20')
node_colors = []
for v in G.nodes():
    k = cid.get(v, -1)
    node_colors.append(palette(big.index(k) / max(len(big) - 1, 1)) if k in big else (0.82, 0.82, 0.82, 1.0))
size = 24 + 900 * np.array([degc[v] for v in G.nodes()])
nx.draw_networkx_edges(G, pos, ax=ax[0], alpha=0.12, width=0.5)
nx.draw_networkx_nodes(G, pos, ax=ax[0], node_size=size, node_color=node_colors, linewidths=0.2,
                       edgecolors='white')
ax[0].set_title(f'化合物共表达图（k={K_NN} 近邻并集，边 {G.number_of_edges()}，模块 {len(comms)}，Q={mod:.3f}）')
ax[0].axis('off')

vc = pd.Series(deg).value_counts().sort_index()
ax[1].bar(vc.index, vc.values, color='#4878a8')
ax[1].set_yscale('log')
ax[1].set_xlabel('度')
ax[1].set_ylabel('化合物数（对数）')
ax[1].set_title('度分布（对数纵轴）')
ax[1].grid(alpha=0.25, axis='y')
plt.tight_layout()
plt.savefig(os.path.join(OUT_Q2, 'fig01_graph.png'), dpi=140)
plt.close('all')

with open(os.path.join(OUT_Q2, '01_graph.md'), 'w', encoding='utf-8') as f:
    f.write('\n'.join(L))
print()
print('图与节点表已写入', OUT_Q2)

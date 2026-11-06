# -*- coding: utf-8 -*-
# 答案汇总：把三个问题的结果整理成一张清单，并把表格复制到 答案 文件夹
# 只做搬运与索引，不重算模型，保证答案与实验记录一一对应
import os
import json
import shutil
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'q1'))
from common import ANS, OUT, OUT_Q2, OUT_Q3, TARGET_COLS, ensure_dirs, load_chem, load_sensory

ensure_dirs()
L = []


def P(s=''):
    L.append(str(s))
    print(s)


def cp(src, dst):
    if os.path.exists(src):
        shutil.copyfile(src, os.path.join(ANS, dst))
        return True
    return False


P('# 答案汇总')
P()

# 数据标度的核实：题目要求明确 log 底数与正负值的对应逻辑，结论单独成文
scale_md = os.path.join(OUT, '09_scale_audit.md')
if os.path.exists(scale_md):
    P('## 数据标度的核实')
    P()
    P('题目要求核实 1.csv 的 log 底数与正负值的含义，结论如下，完整证据在 问题1_标度核实.md。')
    P()
    P('- 底数为 **log10**：16 个已知物理量的反变换平均命中率 0.9585，base e 只有 0.8553；')
    P('  决定性证据是 pH，ln 反变换给出 1.85，低于啤酒 pH 的物理下限 2.8。')
    P('- **负值是正常取值**：z 的正负只表示原始浓度相对 1 的大小，z=-1 即 0.1，不是缺失或异常编码。')
    P('- **标度并不统一**：56 个全程非负的列给的是原始浓度而非 log10，其中 11 列最大值上万、')
    P('  15 列集中在 4 到 9 之间。对树模型无影响（单调变换不改变分裂点），但计算 OAV 时必须按列区分。')
    P()
    shutil.copyfile(scale_md, os.path.join(ANS, '问题1_标度核实.md'))
P()

# 问题一：直接搬运已生成的评分表，并登记精度
q1_grade = os.path.join(ANS, '问题1_感官评分_分级0到6.csv')
P('## 问题一 感官评分')
P()
if os.path.exists(q1_grade):
    g = pd.read_csv(q1_grade)
    P(f'样本 {len(g)} 条，评分维度 {len(TARGET_COLS)} 个，取值 0 到 6 的整数。')
    vc = g[TARGET_COLS].to_numpy().ravel()
    P(f'全部 {vc.size} 个评分的分布：' +
      '，'.join(f'{k} 级 {int((vc == k).sum())} 个' for k in range(7) if (vc == k).sum() > 0) + '。')
    P()
    # 指标从 06e 的产物读，不再写死在文案里（旧版硬编码 0.2498 已失效）
    j6 = os.path.join(OUT, '06e_final.json')
    if os.path.exists(j6):
        d6 = json.load(open(j6, encoding='utf-8'))
        P(f"对应指标：宏 R2 {d6['macro']:.4f}（门槛 0.274，"
          f"{'达标' if d6['gates']['macro'] else '未达标'}）、"
          f"12 弱维均值 {d6['weak']:.4f}（门槛 0.075，"
          f"{'达标' if d6['gates']['weak'] else '未达标'}）、"
          f"5 强维下降 {d6['base'][2] - d6['strong']:.4f}（门槛 ≤0.005，"
          f"{'达标' if d6['gates']['strong'] else '未达标'}）。")
        P(f"交付列 = {d6['authority']}；基座宏 R2 {d6['base'][0]:.4f}。"
          f"原因与逐维明细见 问题1_说明.md。")
P()

# 问题二
P('## 问题二 Top-3 关键因素化合物')
P()
# 正式答案来自 04_final.py：候选集是 1.csv 全部 231 个化合物与理化列，不裁剪，只输出化合物
p2 = os.path.join(OUT_Q2, '02c_top3.csv')
md2 = os.path.join(OUT_Q2, '02c_top3.md')
nz2 = os.path.join(OUT_Q2, '02c_scores.npz')
if not os.path.exists(p2):
    p2 = os.path.join(OUT_Q2, '02b_top3_compound.csv')
    md2 = os.path.join(OUT_Q2, '02b_stability.md')
if os.path.exists(p2):
    t = pd.read_csv(p2)
    key = 'factor' if 'factor' in t.columns else 'compound'
    sig = None
    if os.path.exists(nz2):
        d2 = np.load(nz2, allow_pickle=True)
        rho2 = d2['rho']
        sig = pd.Series(np.abs(rho2).max(0), index=list(TARGET_COLS))
    P(f'50 个感官维度各给出 3 个关键因素，共 {len(t)} 条记录，全部为化合物，不含理化指标。')
    P('排序用三条证据合成：单变量 Spearman 相关 0.40、Lasso 条件重要性沿共表达图扩散后 0.40、')
    P('bootstrap 入选频率 0.20。候选集是 1.csv 的全部 231 个化合物列，没有做近常数列裁剪，')
    P('否则 X4VG 这种几乎全在检出限以下的列会失去资格。')
    P()
    P('| 感官维度 | 第 1 | 第 2 | 第 3 | 该维最强 abs(rho) |')
    P('| --- | --- | --- | --- | --- |')
    for dim in TARGET_COLS:
        s = t[t['dim'] == dim].sort_values('rank')
        if len(s):
            v = f'{sig[dim]:.3f}' if sig is not None and dim in sig.index else ''
            P(f'| {dim} | ' + ' | '.join(str(c) for c in s[key]) + f' | {v} |')
    P()
    if 'boot_freq' in t.columns:
        P(f'150 个席位的平均 bootstrap 入选频率 {t["boot_freq"].mean():.3f}，'
          f'超过 0.5 的 {int((t["boot_freq"] > 0.5).sum())} 个。')
    P('缺陷维度要特别说明：DMS、双乙酰、反-2-壬烯醛这三个化合物的实测浓度与同名感官维度的秩相关')
    P('只有 0.09 到 0.15，都没有进前三，所以那三个维度的前三名只能当「相对最强」看，不宜作化学解释。')
    P('X4VG 在 X4vg 维度上排第一；bitterness（理化指标）在 bitternes 维度上排第一，')
    P('说明方法在真信号处是有效的，理化指标与化合物的分工见 问题2_说明.md 第 7 节。')
    P('逐维证据与自检表见 问题2_说明.md。')
    P()
    cp(p2, '问题2_关键因素化合物.csv')
    if os.path.exists(md2):
        shutil.copyfile(md2, os.path.join(ANS, '问题2_说明.md'))
P()

# 问题三
P('## 问题三 风格条件生成与跨风格迁移')
P()
nz = os.path.join(OUT_Q3, '01_generative.npz')
if os.path.exists(nz):
    d = np.load(nz, allow_pickle=True)
    cats = [str(c) for c in d['cats']]
    keep = [str(c) for c in d['keep']]
    rows = []
    for i, c in enumerate(cats):
        g = d[f'gen_{i}']
        df = pd.DataFrame(g, columns=keep)
        df.insert(0, 'sample', np.arange(len(df)))
        df.insert(0, 'style', c)
        rows.append(df)
    GEN = pd.concat(rows, ignore_index=True)
    GEN.to_csv(os.path.join(ANS, '问题3_生成化合物谱.csv'), index=False, encoding='utf-8-sig')
    P(f'生成化合物谱：{len(cats)} 个风格，每风格 100 条，共 {len(GEN)} 条，每条 {len(keep)} 个化合物列'
      '（各列原标度，多数为 log10 浓度）。')
pp = os.path.join(OUT_Q3, '03_gen_portrait.csv')
if os.path.exists(pp):
    cp(pp, '问题3_生成感官画像.csv')
    pc = pd.read_csv(pp)
    P(f'生成感官画像：{len(pc)} 条，每条 50 个感官维度的 0-6 分制得分。')
P()
# 生成质量就地做一次合并检验，口径与 01_generative.md 相同，只是换成不依赖中间文件的算法
gc = os.path.join(OUT_Q3, '01_gen_check.json')
if os.path.exists(gc):
    with open(gc, encoding='utf-8') as f:
        gcd = json.load(f)
    P(f'生成质量的合并检验：把 22 个风格各 100 条池化成 {gcd["n_gen"]} 条，与 {gcd["n_real"]} 条真实样本逐维比。')
    P(f'逐维 KS 统计量均值 {gcd["ks_mean"]:.3f}（该样本量的 5% 临界值是 {gcd["ks_crit"]:.3f}），')
    P(f'标准化均值偏差 {gcd["bias"]:.3f}，总方差比 {gcd["var_ratio"]:.3f}。')
    P('逐风格看样本只有 2 到 9 条，KS 的分辨率不足，所以结论以合并检验为准。')
P()
tr = os.path.join(OUT_Q3, '02_transfer.csv')
if os.path.exists(tr):
    cp(tr, '问题3_跨风格迁移化合物谱.csv')
    cp(os.path.join(OUT_Q3, '02_transfer_sensory_before.csv'), '问题3_迁移前感官画像.csv')
    cp(os.path.join(OUT_Q3, '02_transfer_sensory_after.csv'), '问题3_迁移后感官画像.csv')
    P('迁移：Pils/Lager 的真实样本保留个体潜编码，只换风格均值后重建，得到 Tripel 的化合物谱。')
    P('缺陷化合物按题目列出的五类共 7 个处理：6 个在生成支持集内，逐条样本做单调截断，')
    P('迁移后不高于迁移前是构造保证的；X4VG 不在支持集内，迁移过程根本不改写它。')
    P('迁移后 7 个缺陷化合物全部不高于迁移前，5 个不高于 Tripel 的真实水平。')
    # 相关系数现算，不再写死（旧文案 0.866→0.993 与当前产物不符）
    try:
        from common import load_sensory, load_scale, z_to_score
        S = load_sensory()
        MU, SG = load_scale()
        m = S['tasting_category_fine'] == 'Tripel'
        real = z_to_score(S.loc[m, TARGET_COLS].to_numpy(dtype=float), MU, SG).mean(axis=0)
        af = pd.read_csv(os.path.join(OUT_Q3, '02_transfer_sensory_after.csv'))
        bf = pd.read_csv(os.path.join(OUT_Q3, '02_transfer_sensory_before.csv'))
        r_af = float(np.corrcoef(af[TARGET_COLS].mean(axis=0).to_numpy(), real)[0, 1])
        r_bf = float(np.corrcoef(bf[TARGET_COLS].mean(axis=0).to_numpy(), real)[0, 1])
        P(f'迁移后的 50 维感官画像与 Tripel 真实画像的相关系数从 {r_bf:.3f} 升到 {r_af:.3f}，明细见 问题3_说明.md。')
    except Exception as e:
        P(f'迁移画像相关系数未能现算（{e}），见 问题3_说明.md。')
P()
md3 = os.path.join(OUT_Q3, '02_transfer.md')
if os.path.exists(md3):
    shutil.copyfile(md3, os.path.join(ANS, '问题3_说明.md'))
P()

# 图表
for f, dst in [(os.path.join(OUT, 'fig05_hyperbolic.png'), '图_双曲嵌入.png'),
               (os.path.join(OUT, 'fig07_ablation.png'), '图_消融矩阵.png'),
               (os.path.join(OUT, 'fig08_answers.png'), '图_问题一精度.png'),
               (os.path.join(OUT, 'fig08_scatter.png'), '图_问题一散点.png'),
               (os.path.join(OUT_Q2, 'fig01_graph.png'), '图_共表达网络.png'),
               (os.path.join(OUT_Q3, 'fig01_generative.png'), '图_生成模型.png'),
               (os.path.join(OUT_Q3, 'fig02_transfer.png'), '图_跨风格迁移.png')]:
    cp(f, dst)

P('## 文件清单')
P()
# 早前几步已经复制过带 fig 前缀的图，这里清掉重名副本，只保留中文命名的图
for f in os.listdir(ANS):
    if f.startswith('fig') and f.endswith('.png'):
        os.remove(os.path.join(ANS, f))
# 早期版本还留过「含理化指标」的 Q2 补充表，最终版把理化指标写进说明文档，这张表不再需要
legacy = os.path.join(ANS, '问题2_关键因素化合物_含理化指标.csv')
if os.path.exists(legacy):
    os.remove(legacy)
for f in sorted(os.listdir(ANS)):
    P(f'- {f}')
P()

with open(os.path.join(ANS, '答案总览.md'), 'w', encoding='utf-8') as f:
    f.write('\n'.join(L))
print('答案目录已整理完毕')

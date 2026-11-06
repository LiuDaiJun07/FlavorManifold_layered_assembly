# -*- coding: utf-8 -*-
# 公共配置与数据加载：路径解析、列分组、常量
# 所有脚本都从这里取路径与列定义，保证口径一致
import os
import re
import unicodedata

import numpy as np
import pandas as pd

# 出图统一用中文字体，否则标题与轴标签会渲染成方块
# 只设 rcParams，不引入 pyplot，避免影响各脚本自己的后端设置
import matplotlib as mpl
mpl.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'DengXian', 'SimHei', 'DejaVu Sans']
mpl.rcParams['axes.unicode_minus'] = False

SEED = 20260228
# 目标维度：4.csv 中除 id 与风格列外的 50 个感官维度，顺序与 3.xlsx summary 表一致
TARGET_COLS = [
    'A_malt_all', 'A_malt_grain', 'A_malt_bread', 'A_malt_cara', 'A_malt_burn',
    'A_hops_all', 'A_hops_citrus', 'A_hops_tropical', 'A_hops_noble', 'A_hops_woody',
    'A_esters_all', 'A_esters_ethac', 'A_esters_isoaa', 'A_esters_flower', 'A_esters_fruity',
    'F_malt_all', 'F_malt_grain', 'F_malt_bread', 'F_malt_cara', 'F_malt_burn',
    'F_hops_all', 'F_hops_citrus', 'F_hops_tropical', 'F_hops_noble', 'F_hops_woody',
    'F_esters_all', 'F_esters_ethac', 'F_esters_isoaa', 'F_esters_flower', 'F_esters_fruity',
    'acidity', 'bitternes', 'sweetness', 'X4vg', 'diacetyl', 'dms', 'metallic',
    'stale_hops', 't2n', 'orange', 'coriander', 'clove', 'lactic', 'acetic', 'barnyard',
    'alcohol', 'aftertaste', 'body', 'co2', 'overall',
]
# 题目表述中的 30 维核心集：A_ 15 + F_ 15，即 3.xlsx 里 Aroma 与 Flavour 两套香气/风味维度
CORE_COLS = [c for c in TARGET_COLS if c.startswith(('A_', 'F_'))]
# 其余 20 维：口感、缺陷、香料与其他
EXTRA_COLS = [c for c in TARGET_COLS if c not in CORE_COLS]
ID_COLS = ['beer', 'beer_id', 'tasting_category_fine']
# 1.csv 末尾 4 列是聚合求和列，来自原始化合物列的直接线性组合，建模时单列保留但需标注
SUM_COLS = ['esters_sum', 'aroma_hops_sum', 'acids_sum', 'sulfur_sum']

# 并行度：沙盒内只有 1 可用（多进程会碰命名管道权限），沙盒外设 -1 提速
NJOB = int(os.environ.get('NJOB', '1'))
# 协议折数：补丁要求统一 10 折，25 折只作复核；冒烟测试可用 2 折
N_FOLDS = int(os.environ.get('FOLDS', '10'))
# 树数缩放：正式跑为 1.0，冒烟测试设小值（如 0.05）缩短单次运行
TREE_SCALE = float(os.environ.get('TREE_SCALE', '1.0'))
# 主口径：反标准化 sigma 取训练折该维标准差，其余两口径只作并列披露
CALIB_MAIN = 'sd'
# ZITS 逐维开关的唯一硬判据：训练折零占比不低于该值
ZERO_SWITCH = 0.40
# 12 个弱维（宏 R2 < 0.11 的事后子集，须在报告中披露这一点）
WEAK_DIMS = ['clove', 'F_esters_flower', 'A_esters_flower', 'orange', 'stale_hops',
             'A_esters_fruity', 'A_malt_grain', 'barnyard', 'dms', 'diacetyl', 't2n', 'F_malt_grain']
# 5 个强维，用于监控叠加步骤是否损伤强信号
STRONG_DIMS = ['acidity', 'alcohol', 'lactic', 'acetic', 'bitternes']
WEAK_IDX = [TARGET_COLS.index(c) for c in WEAK_DIMS]
STRONG_IDX = [TARGET_COLS.index(c) for c in STRONG_DIMS]


# 从脚本目录向上找含「题目」子目录的目录作为工作区根
def find_ws(start=None):
    here = os.path.dirname(os.path.abspath(start or __file__))
    while not os.path.isdir(os.path.join(here, '题目')):
        parent = os.path.dirname(here)
        if parent == here:
            raise RuntimeError('未找到工作区根目录')
        here = parent
    return here


WS = find_ws(__file__)
ATT = os.path.join(WS, '题目', '附件')
MAIN = os.path.join(WS, 'main')
OUT = os.environ.get('Q1_OUT') or os.path.join(MAIN, 'Output', 'Q1')
OUT_Q2 = os.path.join(MAIN, 'Output', 'Q2')
OUT_Q3 = os.path.join(MAIN, 'Output', 'Q3')
# 允许把交付目录改到别处（冒烟测试用），默认仍为工作区根下的「答案」
ANS = os.environ.get('Q1_ANS') or os.path.join(WS, '答案')
# 理化列与化合物列的分界，用于对数比变换与分块消融
PHYS_COLS = ['SO2.mg.L.', 'acetic_acid.g.L.', 'ammonia.mg.L.', 'color.EBC.', 'betaglucan.mg.L.',
             'lactic_acid.mg.L.', 'glycerol', 'iron', 'pH', 'protein.g.L.', 'bitterness', 'GFS',
             'ethanol..v.v.', 'CO2.PSI.', 'unfermentables', 'kcalperc']


# 确保输出目录存在
def ensure_dirs():
    for d in (OUT, OUT_Q2, OUT_Q3, ANS):
        os.makedirs(d, exist_ok=True)


# 读 1.csv 化合物数据
def load_chem():
    return pd.read_csv(os.path.join(ATT, '1.csv'))


# 读 4.csv 感官数据（训练目标，已标准化）
def load_sensory():
    return pd.read_csv(os.path.join(ATT, '4.csv'))


# 读步骤二产出的 0-6 分级目标，行序与 4.csv 一致
def load_sensory_grade():
    g = pd.read_csv(os.path.join(OUT, '02_target_grade.csv'))
    return g[TARGET_COLS].to_numpy(int)


# 读 7.csv 描述词感官数据（仅作参考与外部校验，禁止进入特征）
def load_sensory7():
    return pd.read_csv(os.path.join(ATT, '7.csv'))


# 读 8.xlsx 指定 sheet
def load_x8(sheet):
    return pd.read_excel(os.path.join(ATT, '8.xlsx'), sheet_name=sheet)


# 取 1.csv 中化合物与理化列名（去掉 id 与风格）
def chem_feature_cols(df=None):
    df = load_chem() if df is None else df
    return [c for c in df.columns if c not in ID_COLS]


# 取 4.csv 中目标维度列名，按 TARGET_COLS 顺序对齐
def sensory_cols(df=None):
    df = load_sensory() if df is None else df
    return [c for c in TARGET_COLS if c in df.columns]


# 读 8.xlsx 化学物元数据并转成以 1.csv 列名为键的字典
def chem_meta():
    th = load_x8('Sensory thresholds')
    meta = {}
    for _, r in th.iterrows():
        key = r['field_name_R']
        if isinstance(key, str):
            meta[key] = r.to_dict()
    return meta


# 从 8.xlsx Sensory names 取「受训小组」维度到类别/中文别名映射
def sensory_meta():
    sn = load_x8('Sensory names')
    sn = sn[sn['dataset'] == 'Trained panel']
    return {r['R_name']: r.to_dict() for _, r in sn.iterrows()}


# 8.xlsx 的部分文本存在 UTF-8 被按 Latin-1 解码的乱码，反向修回
def demojibake(s):
    s = str(s)
    if any(0x80 <= ord(ch) <= 0xFF for ch in s):
        t = s.encode('latin-1', 'ignore').decode('utf-8', 'ignore')
        if t:
            return t
    return s


# 化学名归一化：修乱码、希腊字母转写、去重音、只留小写字母数字
def norm_name(s):
    s = demojibake(s)
    for a, b in (('α', 'alpha'), ('β', 'beta'), ('γ', 'gamma'), ('δ', 'delta')):
        s = s.replace(a, b)
    s = unicodedata.normalize('NFKD', s)
    s = ''.join(ch for ch in s if not unicodedata.combining(ch))
    return re.sub(r'[^a-z0-9]', '', s.lower())


# 2.csv 用的是可读化学名，1.csv 用的是 field_name_R 风格，这里建立二者映射
def chem_name_map():
    th = load_x8('Sensory thresholds').dropna(subset=['field_name_python'])
    keys = {}
    for _, r in th.iterrows():
        tgt = r['field_name_R']
        for col in ('chemical_name', 'chemical_name_short', 'field_name_python', 'field_name_R'):
            v = r.get(col)
            if isinstance(v, str):
                keys.setdefault(norm_name(v), tgt)
    out = {}
    for n in pd.read_csv(os.path.join(ATT, '2.csv'), index_col=0).index:
        k = norm_name(n)
        if k in keys:
            out[str(n)] = keys[k]
    return out


# 读 2.csv 相关阵并换成 1.csv 列名顺序，返回 (列名, 相关阵)
def load_graph_corr():
    d = pd.read_csv(os.path.join(ATT, '2.csv'), index_col=0)
    mp = chem_name_map()
    ren = [mp[str(n)] for n in d.index]
    d.index = ren
    d.columns = [mp[str(n)] for n in d.columns]
    cols = chem_feature_cols()
    d = d.loc[cols, cols]
    return cols, d.to_numpy(dtype=float)


# 读反标准化参数表，返回按 TARGET_COLS 对齐的 mu 与 sigma
def load_scale():
    m = pd.read_csv(os.path.join(OUT, '02_destandardize.csv')).set_index('dim')
    m = m.loc[TARGET_COLS]
    return m['mu'].to_numpy(float), m['sigma'].to_numpy(float)


# 标准化值还原成 0-6 连续分并裁剪
def z_to_score(z, mu, sg):
    return np.clip(mu[None, :] + sg[None, :] * z, 0.0, 6.0)


# 连续分取整成 0-6 等级
def score_to_grade(s):
    return np.clip(np.round(s), 0, 6).astype(int)


# 宏平均 R2：逐维算决定系数后取均值，逐维方差为 0 时记 0
def macro_r2(yt, yp):
    out = []
    for j in range(yt.shape[1]):
        ss_res = float(np.sum((yt[:, j] - yp[:, j]) ** 2))
        ss_tot = float(np.sum((yt[:, j] - yt[:, j].mean()) ** 2))
        out.append(1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0)
    return np.array(out)


# 森林等级经验分布期望：逐棵树还原 0-6 等级得经验分布 P(k|x)，再取期望并四舍五入
# 相比直接对连续预测取整，这一步把「等级」当离散标签投票，减少单棵树的连续外推误差
def expect_grade(model, X, mu, sg):
    hist = np.zeros((X.shape[0], len(TARGET_COLS), 7))
    for est in model.estimators_:
        g = score_to_grade(z_to_score(est.predict(X), mu, sg))
        for k in range(7):
            hist[:, :, k] += (g == k)
    hist /= np.maximum(hist.sum(axis=2, keepdims=True), 1.0)
    exp = np.tensordot(hist, np.arange(7, dtype=float), axes=([2], [0]))
    return score_to_grade(exp), exp, hist


# 等级分布的中位数：绝对误差损失下的贝叶斯最优点估计
# 先按行归一化：混合直方图一旦某个成员缺项，行和就不再是 1，
# 直接对累积和比 0.5 会静默变成「高分位」而不是中位数（漏洞 01 的放大器）
def median_grade(hist):
    s = hist.sum(axis=2, keepdims=True)
    h = np.divide(hist, np.maximum(s, 1e-12))
    c = np.cumsum(h, axis=2)
    return np.argmax(c >= 0.5, axis=2).astype(int)


# 按 TREE_SCALE 缩放树数，最少 30 棵，保证逐树经验分布不至于太抖
def trees(n):
    return max(30, int(round(n * TREE_SCALE)))


# 三口径反标准化参数：返回按 TARGET_COLS 顺序的 (mu, sigma)
# mode='sd' 主口径：sigma 取训练折该维标准差，mu_j = -sigma_j * z_min_j
# mode='unit'：sigma=1，mu=-z_min，等价于 z 直接线性搬到 0-6
# mode='minmax'：sigma=6/(zmax-zmin)，把训练折目标值域线性拉满 0-6
def calib_params(ytr, mode='sd'):
    Y = np.asarray(ytr, dtype=float)
    zmin = Y.min(axis=0)
    zmax = Y.max(axis=0)
    sd = Y.std(axis=0, ddof=1)
    sd = np.where(sd > 1e-9, sd, 1.0)
    if mode == 'unit':
        sg = np.ones_like(sd)
    elif mode == 'minmax':
        sg = 6.0 / np.maximum(zmax - zmin, 1e-9)
    else:
        sg = sd
    mu = -sg * zmin
    return mu, sg


# 三口径并列：同一组 z 空间预测在 unit / sd / minmax 下的连续分 MAE
def mae_caliber(yt, yp, ytr_fold):
    out = {}
    for mode in ('unit', 'sd', 'minmax'):
        mu, sg = calib_params(ytr_fold, mode)
        out[mode] = float(np.mean(np.abs(z_to_score(yt, mu, sg) - z_to_score(yp, mu, sg))))
    return out


# 折内口径的三口径 MAE：每折只用该折训练行估 mu/sigma，再按格点数加权平均
# 与 06b 的折内 calib 一致，避免「评估用全量刻度、建模用折内刻度」的口径不一致
def mae_caliber_oof(yt, yp, fold_id):
    out = {}
    fid = np.asarray(fold_id, dtype=int)
    for mode in ('unit', 'sd', 'minmax'):
        num = 0.0
        den = 0
        for f in np.unique(fid):
            te = np.where(fid == f)[0]
            tr = np.where(fid != f)[0]
            mu, sg = calib_params(yt[tr], mode)
            d = np.abs(z_to_score(yt[te], mu, sg) - z_to_score(yp[te], mu, sg))
            num += float(np.sum(d))
            den += int(d.size)
        out[mode] = num / max(den, 1)
    return out


# 宏平均 R2 与弱维/强维分组的均值，报告里到处要用
def r2_groups(yt, yp):
    r = macro_r2(yt, yp)
    return float(r.mean()), float(r[WEAK_IDX].mean()), float(r[STRONG_IDX].mean()), r


# K 折切分：类别样本够就分层，不够（如 22 风格里最小的只有 3 条）就退回打乱 KFold
# 折数不会被最小类别样本数压小，保证 10 折协议在任何数据上都是 10 折
def split_folds(strata, n_folds=None, seed=SEED):
    from sklearn.model_selection import KFold, StratifiedKFold
    s = pd.Series(list(strata))
    nf = int(N_FOLDS if n_folds is None else n_folds)
    nf = max(2, min(nf, len(s)))
    mn = int(s.value_counts().min())
    if mn >= nf:
        skf = StratifiedKFold(n_splits=nf, shuffle=True, random_state=seed)
        return [(tr, te) for tr, te in skf.split(np.zeros(len(s)), s.to_numpy())]
    kf = KFold(n_splits=nf, shuffle=True, random_state=seed)
    return [(tr, te) for tr, te in kf.split(np.zeros(len(s)))]

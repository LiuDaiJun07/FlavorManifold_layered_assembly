# -*- coding: utf-8 -*-
# 04 双曲嵌入模块：折内 kNN 图 -> 最短路距离 -> Poincaré 球 stress 嵌入 -> 双曲特征
# 环境无 torch/geoopt，故用纯 numpy 手写黎曼梯度下降
# 这样做的取舍：250 个点、维数不超过 8，黎曼优化成本极低，不值得引入 200MB 的深度学习栈
import numpy as np

EPS = 1e-9
# 球内点半径上限，避免双曲距离数值爆炸
RAD_CAP = 1.0 - 1e-5


# 把点限制在 Poincaré 球内部
def project_ball(Z, c=1.0):
    lim = RAD_CAP / np.sqrt(c)
    nrm = np.linalg.norm(Z, axis=1, keepdims=True)
    scale = np.minimum(1.0, lim / np.maximum(nrm, EPS))
    return Z * scale


# Möbius 加法 u (+) v，逐样本广播
def mobius_add(u, v, c=1.0):
    uu = np.sum(u * u, axis=1, keepdims=True)
    vv = np.sum(v * v, axis=1, keepdims=True)
    uv = np.sum(u * v, axis=1, keepdims=True)
    num = (1 + 2 * c * uv + c * vv) * u + (1 - c * uu) * v
    den = 1 + 2 * c * uv + c * c * uu * vv
    return num / np.maximum(den, EPS)


# 指数映射：把切空间向量从原点推到球上
def exp_map0(w, c=1.0):
    nrm = np.linalg.norm(w, axis=1, keepdims=True)
    sq = np.sqrt(c)
    coef = np.tanh(sq * nrm / 2.0) / np.maximum(sq * nrm, EPS)
    return coef * w


# 对数映射：把球上点拉回原点切空间
def log_map0(z, c=1.0):
    nrm = np.linalg.norm(z, axis=1, keepdims=True)
    sq = np.sqrt(c)
    coef = np.arctanh(np.minimum(sq * nrm, RAD_CAP)) / np.maximum(sq * nrm, EPS)
    return (2.0 / sq) * coef * z


# Poincaré 球内两点测地距离，输入 (n,d) 与 (m,d) 或 (n,d) 与 (n,d)
# 用 log1p 形式计算 arccosh(1+x)，避免 x 很小时（c 接近 0 的欧氏极限）出现精度损失
def poincare_dist(A, B, c=1.0, same=False):
    aa = np.sum(A * A, axis=1)[:, None]
    bb = np.sum(B * B, axis=1)[None, :]
    diff = aa + bb - 2.0 * (A @ B.T)
    diff = np.maximum(diff, 0.0)
    den = (1.0 - c * aa) * (1.0 - c * bb)
    x = 2.0 * c * diff / np.maximum(den, EPS)
    x = np.maximum(x, 0.0)
    D = np.log1p(x + np.sqrt(x * (x + 2.0)))
    if same:
        np.fill_diagonal(D, 0.0)
    return D


# 由特征矩阵构造 kNN 图并取最短路距离
# 相似度用 Pearson 相关，距离用 1 - r，图不连通时用最大有限距离补齐
def knn_graph_distance(X, k=10):
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import shortest_path
    Xs = (X - X.mean(axis=0)) / np.maximum(X.std(axis=0), EPS)
    Xs = np.nan_to_num(Xs, nan=0.0, posinf=0.0, neginf=0.0)
    R = np.corrcoef(Xs)
    R = np.nan_to_num(R, nan=0.0)
    W = 1.0 - R
    np.fill_diagonal(W, 0.0)
    n = W.shape[0]
    kk = int(min(k, n - 1))
    # 只保留每个点的 k 个最近邻，保证图稀疏
    idx = np.argpartition(W, kk, axis=1)[:, :kk + 1]
    A = np.full((n, n), np.inf)
    rows = np.repeat(np.arange(n), idx.shape[1])
    cols = idx.ravel()
    A[rows, cols] = W[rows, cols]
    A = np.minimum(A, A.T)
    A[~np.isfinite(A)] = 0.0
    D = shortest_path(csr_matrix(A), directed=False)
    fin = D[np.isfinite(D)]
    big = float(np.max(fin)) * 1.05 if fin.size else 1.0
    D = np.where(np.isfinite(D), D, big)
    np.fill_diagonal(D, 0.0)
    return D


# 双曲 stress 损失与其对坐标的欧氏梯度
# L = mean_ij (d_ij - D_ij)^2 / (D_ij^2 + eps)，按开发文档的加权应力形式
def stress_loss_grad(Z, Dt, c=1.0, mask=None):
    n = Z.shape[0]
    aa = np.sum(Z * Z, axis=1)
    # Pu = 1 - c||u||^2 只依赖单点，必须单独保留，否则对 u 求导时分母的幂次会错
    Pu = np.maximum(1.0 - c * aa, EPS)
    dot = Z @ Z.T
    S = np.maximum(aa[:, None] + aa[None, :] - 2.0 * dot, 0.0)
    den = Pu[:, None] * Pu[None, :]
    arg = np.maximum(1.0 + 2.0 * c * S / den, 1.0 + 1e-12)
    d = np.arccosh(arg)
    w = 1.0 / (Dt * Dt + 1e-6)
    if mask is not None:
        w = w * mask
    np.fill_diagonal(w, 0.0)
    resid = (d - Dt) * w
    loss = float(np.sum(resid * (d - Dt))) / (n * n)
    # dL/dd_ij = 2 (d - D)/(D^2+eps) / n^2
    gd = 2.0 * resid / (n * n)
    # 距离矩阵对称，把 (i,j) 与 (j,i) 两条贡献合并后一次性累加
    coef = (gd + gd.T) / np.sqrt(np.maximum(arg * arg - 1.0, EPS))
    Zc = Z[:, None, :] - Z[None, :, :]
    # dA_ij/du_i = 4c[(u_i - u_j) Pu_i + c S_ij u_i] / (Pu_i^2 Pu_j)
    numv = 4.0 * c * (Zc * Pu[:, None, None] + c * S[:, :, None] * Z[:, None, :])
    dsc = (Pu[:, None] ** 2) * Pu[None, :]
    G = np.sum(coef[:, :, None] * numv / dsc[:, :, None], axis=1)
    return loss, G


# 黎曼梯度下降拟合 Poincaré 嵌入，初值由经典 MDS 给，保证收敛稳定
def fit_poincare(Dt, dim=4, c=1.0, iters=600, lr=0.02, seed=0):
    n = Dt.shape[0]
    # 应力目标先除以均值，使各点对的权重处在同一量级，避免远点对被过度放大
    pos = Dt[Dt > 0]
    Dt = Dt / max(float(pos.mean()) if pos.size else 1.0, EPS)
    # 经典 MDS 作为初值，再经指数映射塞进球内
    J = np.eye(n) - 1.0 / n
    B = -0.5 * J @ (Dt ** 2) @ J
    ev, evec = np.linalg.eigh(B)
    order = np.argsort(ev)[::-1][:dim]
    ev = np.maximum(ev[order], 0.0)
    W0 = evec[:, order] * np.sqrt(ev)[None, :]
    sc = 0.5 / np.maximum(np.max(np.linalg.norm(W0, axis=1)), EPS)
    Z = exp_map0(W0 * sc, c)
    Z = project_ball(Z, c)
    m = np.zeros_like(Z)
    v = np.zeros_like(Z)
    b1, b2 = 0.9, 0.999
    for t in range(1, iters + 1):
        loss, G = stress_loss_grad(Z, Dt, c)
        # 欧氏梯度转黎曼梯度
        lam = (1.0 - c * np.sum(Z * Z, axis=1, keepdims=True))
        G = G * (lam ** 2) / 4.0
        m = b1 * m + (1 - b1) * G
        v = b2 * v + (1 - b2) * G * G
        mh = m / (1 - b1 ** t)
        vh = v / (1 - b2 ** t)
        step = lr * mh / (np.sqrt(vh) + 1e-8)
        # 用原点指数映射作为回缩，步长按切向范数自适应
        Z = project_ball(exp_map0(log_map0(Z, c) - step, c), c)
    return Z, loss


# 风格质心：球内点的爱因斯坦中点用切空间均值近似，误差在球内小半径时可接受
def frechet_mean(Z, c=1.0):
    return project_ball(exp_map0(log_map0(Z, c).mean(axis=0, keepdims=True), c), c)[0]


# 由化学特征回归切空间坐标，用于把折外样本映射进球
# 采用岭回归，输入化学特征，输出 log_0(Z) 的 dim 个坐标
def fit_tangent_ridge(X, Z, c=1.0, alpha=1.0):
    U = log_map0(Z, c)
    Xm = X.mean(axis=0)
    Xs = X.std(axis=0)
    Xs = np.where(Xs < EPS, 1.0, Xs)
    Xn = (X - Xm) / Xs
    A = Xn.T @ Xn + alpha * np.eye(Xn.shape[1])
    Wc = np.linalg.solve(A, Xn.T @ U)
    return {'Xm': Xm, 'Xs': Xs, 'Wc': Wc}


# 用拟合好的岭回归把新样本映射进球
def transform_tangent_ridge(model, X, c=1.0):
    Xn = (X - model['Xm']) / model['Xs']
    U = Xn @ model['Wc']
    return project_ball(exp_map0(U, c), c)


# 组装双曲特征：22 个风格质心距离 + 4 个典型性 + 2 个局部密度 + 1 个径向 + dim 个切空间坐标
# 质心一律用训练折样本计算，避免折外信息泄漏
def hyperbolic_features(Z_tr, Z_te, style_tr, style_te, c=1.0):
    styles = sorted(set(style_tr))
    d = Z_tr.shape[1]
    cents = {s: frechet_mean(Z_tr[style_tr == s], c) for s in styles}
    C = np.vstack([cents[s] for s in styles])
    out = {}
    # 到各风格质心的双曲距离
    for i, s in enumerate(styles):
        Dc = poincare_dist(np.vstack([Z_tr, Z_te]), C[i:i + 1], c)[:, 0]
        out['hyp_dist_' + s] = Dc
    Dall_tr = poincare_dist(Z_tr, C, c)
    Dall_te = poincare_dist(Z_te, C, c)
    # 典型性：到本风格质心距离、该距离在所属风格内的 z 分数、与最近异风格质心距离之比
    own_tr = np.array([styles.index(s) for s in style_tr])
    own_te = np.array([styles.index(s) if s in styles else 0 for s in style_te])
    d_own_tr = Dall_tr[np.arange(len(Z_tr)), own_tr]
    d_own_te = Dall_te[np.arange(len(Z_te)), own_te]
    mu_own = np.array([d_own_tr[own_tr == j].mean() for j in range(len(styles))])
    sd_own = np.array([max(d_own_tr[own_tr == j].std(), EPS) for j in range(len(styles))])
    out['hyp_typ_dist'] = np.concatenate([d_own_tr, d_own_te])
    out['hyp_typ_z'] = np.concatenate([(d_own_tr - mu_own[own_tr]) / sd_own[own_tr],
                                       (d_own_te - mu_own[own_te]) / sd_own[own_te]])
    Dall_tr2 = Dall_tr.copy()
    Dall_tr2[np.arange(len(Z_tr)), own_tr] = np.inf
    Dall_te2 = Dall_te.copy()
    Dall_te2[np.arange(len(Z_te)), own_te] = np.inf
    out['hyp_typ_ratio'] = np.concatenate([d_own_tr / np.maximum(Dall_tr2.min(axis=1), EPS),
                                           d_own_te / np.maximum(Dall_te2.min(axis=1), EPS)])
    # 局部密度：到最近 5 个训练点的平均双曲距离
    Dtr_tr = poincare_dist(Z_tr, Z_tr, c, same=True)
    Dte_tr = poincare_dist(Z_te, Z_tr, c)
    kk = min(5, len(Z_tr) - 1)
    out['hyp_dens5'] = np.concatenate([np.sort(Dtr_tr, axis=1)[:, 1:kk + 1].mean(axis=1),
                                       np.sort(Dte_tr, axis=1)[:, :kk].mean(axis=1)])
    # 径向坐标：离球心越远表示越特异
    out['hyp_radius'] = np.concatenate([np.linalg.norm(Z_tr, axis=1), np.linalg.norm(Z_te, axis=1)])
    # 切空间坐标，等价于在双曲几何下做了一次非线性降维
    U = np.vstack([log_map0(Z_tr, c), log_map0(Z_te, c)])
    for j in range(d):
        out['hyp_tan%d' % (j + 1)] = U[:, j]
    return out, styles

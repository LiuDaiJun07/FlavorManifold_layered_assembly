"""Shared, explicit statistical protocols (Q1/Q2/Q3)."""
import numpy as np
from scipy import stats


def wilcoxon(a, b):
    d = np.asarray(a) - np.asarray(b)
    d = d[np.isfinite(d)]
    return float(stats.wilcoxon(d).pvalue) if np.any(abs(d) > 1e-14) else 1.0


def paired(a, b):
    d = np.asarray(a) - np.asarray(b)
    return dict(delta_mean=float(d.mean()), delta_std=float(d.std(ddof=1)),
                wilcoxon_p=wilcoxon(a, b),
                t_p=float(stats.ttest_rel(a, b).pvalue) if np.any(abs(d) > 1e-14) else 1.)


def friedman_nemenyi(scores):
    """Rows=blocks, columns=methods; larger score is better."""
    a = np.asarray(scores)
    n, k = a.shape
    ranks = stats.rankdata(-a, axis=1)
    avg = ranks.mean(0)
    se = np.sqrt(k * (k + 1) / (6 * n))
    p = stats.studentized_range.sf(abs(avg[:, None] - avg[None, :]) / se * np.sqrt(2), k, np.inf)
    cd = stats.studentized_range.ppf(.95, k, np.inf) / np.sqrt(2) * se
    return dict(friedman_p=float(stats.friedmanchisquare(*a.T).pvalue),
                ranks=avg, p=p, cd=float(cd))


def permutation_p(observed, null, larger=True):
    a = np.asarray(null)
    return float((1 + np.sum(a >= observed if larger else a <= observed)) / (len(a) + 1))


def bca_ci(values, statistic=np.mean, seed=20260228, n_resamples=1000):
    v = np.asarray(values)
    if np.ptp(v) < 1e-14:
        return float(v[0]), float(v[0])
    r = stats.bootstrap((v,), statistic, method='BCa', random_state=seed,
                        n_resamples=n_resamples)
    return float(r.confidence_interval.low), float(r.confidence_interval.high)


def bca_from_replicates(point, boot, jack):
    """BCa from original-data bootstrap estimates and delete-one jackknife estimates."""
    boot=np.asarray(boot);jack=np.asarray(jack)
    proportion=((boot<point).sum()+.5*(boot==point).sum())/len(boot)
    z0=stats.norm.ppf(np.clip(proportion,1/(2*len(boot)),1-1/(2*len(boot))))
    delta=jack.mean()-jack
    acceleration=(delta**3).sum()/(6*max((delta**2).sum(),1e-20)**1.5)
    quantile=stats.norm.ppf([.025,.975])
    adjusted=stats.norm.cdf(z0+(z0+quantile)/(1-acceleration*(z0+quantile)))
    return np.quantile(boot,np.clip(adjusted,0,1))

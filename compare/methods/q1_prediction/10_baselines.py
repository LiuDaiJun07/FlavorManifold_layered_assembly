"""C1: nested, fold-local external baseline family."""
from protocol import *
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, ExtraTreesRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import lasso_path


class FoldFeatures(TransformerMixin, BaseEstimator):
    """All pruning, sentinel and style statistics fit anew inside every inner split."""
    def __init__(self,fill='sentinel'):
        self.fill=fill

    def fit(self, x, y=None):
        self.train_ = x.copy()
        self.keep_, _ = FE.prune_cols(x)
        return self

    def transform(self, x):
        both = pd.concat([self.train_, x], ignore_index=True)
        st = both.tasting_category_fine.astype(str).to_numpy()
        _, b, _ = FE.build(both, st, np.arange(len(self.train_)),
                          np.arange(len(self.train_),len(both)),
                          blocks=('chem','miss','style'), feat_cols=self.keep_)
        if self.fill=='median':
            raw=x[self.keep_].to_numpy(float)
            median=self.train_[self.keep_].median().to_numpy(float)
            b[:,:len(self.keep_)]=np.where(np.isnan(raw),median,raw)
        return b


def fit_baseline(name, tr, te, f):
    if name == 'B0':
        return np.tile(Y[tr].mean(0), (len(te), 1))
    if name == 'B1':
        return np.vstack([Y[tr][STYLE[tr] == s].mean(0) if np.any(STYLE[tr] == s)
                          else Y[tr].mean(0) for s in STYLE[te]])
    a, b = CHEM.iloc[tr], CHEM.iloc[te]
    inner = KFold(5, shuffle=True, random_state=C.SEED + f)
    if name in ['B3','B3_med']:
        # Explicit LassoCV-equivalent path: preprocess each inner fold once, then
        # warm-start all 13 alpha values. This avoids 32500 repeated dataframe fits.
        grid=np.logspace(3,-3,13);loss=np.zeros((13,50))
        for itr,ite in inner.split(a):
            transform=make_pipeline(FoldFeatures(fill='median' if name.endswith('_med') else 'sentinel'),StandardScaler()).fit(a.iloc[itr])
            aa,bb=transform.transform(a.iloc[itr]),transform.transform(a.iloc[ite])
            ytrain=Y[tr[itr]];center=ytrain.mean(0)
            for j in range(50):
                _,coefs,_=lasso_path(np.asfortranarray(aa),ytrain[:,j]-center[j],alphas=grid,max_iter=20000,tol=1e-3)
                prediction=bb@coefs+center[j]
                loss[:,j]+=np.mean((prediction-Y[tr[ite],j,None])**2,axis=0)
        best=grid[np.argmin(loss,axis=0)]
        transform=make_pipeline(FoldFeatures(fill='median' if name.endswith('_med') else 'sentinel'),StandardScaler()).fit(a)
        aa,bb=transform.transform(a),transform.transform(b)
        pred=np.column_stack([Lasso(alpha=best[j],max_iter=20000,tol=1e-3).fit(aa,Y[tr,j]).predict(bb) for j in range(50)])
        (OUT/f'10_{name}_params_f{f}.json').write_text(json.dumps(dict(alpha=best.tolist())),encoding='utf-8')
        return pred
    if name in ['B2','B2_med']:
        model = GridSearchCV(make_pipeline(FoldFeatures(fill='median' if name.endswith('_med') else 'sentinel'), StandardScaler(), Ridge()),
                             {'ridge__alpha': np.logspace(-3, 3, 13)}, cv=inner,
                             scoring='neg_mean_squared_error', n_jobs=1)
    elif name == 'B4':
        model = GridSearchCV(make_pipeline(FoldFeatures(), StandardScaler(), PLSRegression(scale=False)),
                             {'plsregression__n_components': list(range(2, 31))}, cv=inner,
                             scoring='neg_mean_squared_error', n_jobs=1)
    elif name.startswith('B5'):
        model = make_pipeline(FoldFeatures(), StandardScaler(), KNeighborsRegressor(n_neighbors=int(name.split('_')[1])))
    elif name == 'B6':
        model = make_pipeline(FoldFeatures(),MultiOutputRegressor(HistGradientBoostingRegressor(max_iter=100,
            max_leaf_nodes=15, l2_regularization=1, early_stopping=False, random_state=C.SEED + f)))
    else:
        # Joint-output tuning selects leaf/max_features; all 50 independent forests.
        model = GridSearchCV(make_pipeline(FoldFeatures(),MultiOutputRegressor(ExtraTreesRegressor(
            n_estimators=C.trees(50), random_state=C.SEED + f, n_jobs=1))),
            {'multioutputregressor__estimator__max_features': [.5, 1.],
             'multioutputregressor__estimator__min_samples_leaf': [1, 2]},
            cv=inner, scoring='neg_mean_squared_error', n_jobs=1,refit=False)
    model.fit(a, Y[tr])
    if hasattr(model, 'best_params_'):
        (OUT / f'10_{name}_params_f{f}.json').write_text(json.dumps(model.best_params_, default=str), encoding='utf-8')
    if name=='B7':
        final=make_pipeline(FoldFeatures(),MultiOutputRegressor(ExtraTreesRegressor(
            n_estimators=C.trees(200),random_state=C.SEED+f,n_jobs=1)))
        final.set_params(**model.best_params_).fit(a,Y[tr])
        return final.predict(b)
    return model.predict(b)


def main():
    ref = original('BASE5')
    rows = [metrics(k, original(k), ref) for k in ('E0', 'E1', 'BASE5')]
    for name in ['B0', 'B1', 'B2', 'B3', 'B4', 'B5_5', 'B5_15', 'B5_25', 'B6', 'B7','B2_med','B3_med']:
        p = cv('10_' + name, fitter=lambda tr, te, f, n=name: fit_baseline(n, tr, te, f))
        rows.append(metrics(name, p, ref))
        report(OUT / '10_baselines.md', 'C1 外部基线（按已完成配置增量落盘）',
               {'metrics': pd.DataFrame(rows)},
               'B6 因环境未安装 LightGBM/XGBoost，按设计降级为 sklearn HistGradientBoosting。'
               'B3用折内预处理的LassoCV等价warm-start路径；B7内层调4组超参。B2/B3沿用哨兵，额外B2_med/B3_med为常规中位数填充，防止用树专用哨兵不公平评价线性基线。'
               '差值方向均为当前模型−BASE5；只在全部基线完成后作家族比较。')
    diagnostics=[]
    for name in ['B2','B3','B4','B2_med','B3_med']:
        pred=np.load(OUT/f'10_{name}.npz')['pred'];bounded=pred.copy()
        for tr,te in FOLDS:bounded[te]=np.clip(pred[te],Y[tr].min(0),Y[tr].max(0))
        rows.append(metrics(name+'_bounded',bounded,ref))
        np.savez_compressed(OUT/f'10_{name}_bounded.npz',pred=bounded,fold_id=FID,r2_dim=C.macro_r2(Y,bounded))
        squared=(pred-Y)**2
        for i,j in np.argwhere(squared>=np.quantile(squared,.999)):
            diagnostics.append(dict(model=name,beer=int(CHEM.beer.iloc[i]),target=C.TARGET_COLS[j],
                actual=Y[i,j],predicted=pred[i,j],squared_error=squared[i,j]))
    best=max(rows,key=lambda r:r['macro_r2'])
    report(OUT/'10_baselines.md','C1 外部基线',{'metrics':pd.DataFrame(rows),'extreme_predictions':pd.DataFrame(diagnostics)},
           f'本表最高宏R²为{best["model"]}：{best["macro_r2"]:.6f}。'
           'B6环境缺LightGBM/XGBoost，降级HistGradientBoosting。B7内层5折×4组以50树筛选，选定参数后200树全量重训并外层评估；这是为控制实测开销采用的多保真选参预算。B3为独立折内预处理的LassoCV等价路径。'
           '线性基线存在少数极端外推，原始R²负值必须保留。med版改用常规折内中位数；bounded版再按训练折目标min/max约束预测，均未使用验证标签。'
           'bounded是观察到外推失败后新增的探索性诊断变体，不进入预定Friedman家族；不能把哨兵或异常外推造成的差距全归于森林方法优势。'
           '差值为模型−BASE5。弱/强维是事后子集，所有指标与50维p值并列披露。')


if __name__ == '__main__':
    main()

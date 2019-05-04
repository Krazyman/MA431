import statsmodels.stats.outliers_influence as st_inf
import numpy as np
# from statsmodels.graphics._regressionplots_doc import _plot_influence_doc

def summary_table(res, alpha=0.05):
    """
    Generate summary table of outlier and influence similar to SAS

    Parameters
    ----------
    alpha : float
       significance level for confidence interval

    Returns
    -------
    st : SimpleTable instance
       table with results that can be printed
    data : ndarray
       calculated measures and statistics for the table
    ss2 : list of strings
       column_names for table (Note: rows of table are observations)
    """

    from scipy import stats
    from statsmodels.sandbox.regression.predstd import wls_prediction_std


    #standard error for predicted mean
    #Note: using hat_matrix only works for fitted values
    predict_mean_se = np.sqrt(res.hat_matrix_diag*res.resid_pearson)

    tppf = stats.t.isf(alpha/2., res.df_resid)
    predict_mean_ci = np.column_stack([
                        res.fittedvalues - tppf * predict_mean_se,
                        res.fittedvalues + tppf * predict_mean_se])


    #standard error for predicted observation
    tmp = wls_prediction_std(res, alpha=alpha)
    predict_se, predict_ci_low, predict_ci_upp = tmp

    predict_ci = np.column_stack((predict_ci_low, predict_ci_upp))

    #standard deviation of residual
    resid_se = np.sqrt(res.resid_pearson * (1 - infl.hat_matrix_diag))

    table_sm = np.column_stack([
                                  np.arange(res.nobs) + 1,
                                  res.model.endog,
                                  res.fittedvalues,
                                  predict_mean_se,
                                  predict_mean_ci[:,0],
                                  predict_mean_ci[:,1],
                                  predict_ci[:,0],
                                  predict_ci[:,1],
                                  res.resid,
                                  resid_se,
                                  infl.resid_studentized_internal,
                                  infl.cooks_distance[0]
                                  ])


    #colnames, data = lzip(*table_raw) #unzip
    data = table_sm
    ss2 = ['Obs', 'Dep Var\nPopulation', 'Predicted\nValue', 'Std Error\nMean Predict', 'Mean ci\n95% low', 'Mean ci\n95% upp', 'Predict ci\n95% low', 'Predict ci\n95% upp', 'Residual', 'Std Error\nResidual', 'Student\nResidual', "Cook's\nD"]
    colnames = ss2
    #self.table_data = data
    #data = np.column_stack(data)
    from statsmodels.iolib.table import SimpleTable, default_html_fmt
    from statsmodels.iolib.tableformatting import fmt_base
    from copy import deepcopy
    fmt = deepcopy(fmt_base)
    fmt_html = deepcopy(default_html_fmt)
    fmt['data_fmts'] = ["%4d"] + ["%6.3f"] * (data.shape[1] - 1)
    #fmt_html['data_fmts'] = fmt['data_fmts']
    st = SimpleTable(data, headers=colnames, txt_fmt=fmt,
                       html_fmt=fmt_html)

    return st, data, ss2
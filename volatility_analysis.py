import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import os
from os.path import expanduser as ospath

def logReturn(series):
    return np.log(series/series.shift(1))

def readData(file):
    df = pd.read_excel(ospath(os.getcwd() + '/data/' + file), index_col=0, skiprows=[0,1,2], header = None)
    df.columns = ['date', 'price']
    df["date"] = pd.to_datetime(df["date"])
    df['logReturn'] = logReturn(df.price)
    return df

data = {'M1': '2019-03-01_15-12-06_Evolucao_historica_da__Media__na_serie_M__1_na_fonte_Convencional_SE_.xlsx' ,
        'M2': '2019-03-01_15-15-00_Evolucao_historica_da__Media__na_serie_M__2_na_fonte_Convencional_SE_.xlsx',
        'M3': '2019-03-01_15-15-12_Evolucao_historica_da__Media__na_serie_M__3_na_fonte_Convencional_SE_.xlsx' ,
        'A0': '2019-03-01_15-16-21_Evolucao_historica_da__Media__na_serie_A__0_na_fonte_Convencional_SE_.xlsx',
        'A1': '2019-03-01_15-16-32_Evolucao_historica_da__Media__na_serie_A__1_na_fonte_Convencional_SE_.xlsx', 
        'A2': '2019-03-01_15-16-45_Evolucao_historica_da__Media__na_serie_A__2_na_fonte_Convencional_SE_.xlsx',
        'A3': '2019-03-01_15-16-58_Evolucao_historica_da__Media__na_serie_A__3_na_fonte_Convencional_SE_.xlsx',
        'A4': '2019-03-01_15-17-12_Evolucao_historica_da__Media__na_serie_A__4_na_fonte_Convencional_SE_.xlsx'}


#Best ARMA selection
def select_best_arma(Returns, data):
    for k in data.keys():
        best_aic = np.inf 
        best_order = None
        best_mdl = None


        a = np.empty([3,4])

        for i in range(1,3):
            for j in range(1,4):
                warnings.filterwarnings('ignore')
                try:
                    tmp_mdl = smt.ARMA(Returns[k], order=(i, j)).fit(
                        method='mle', trend='nc'
                    )
                    tmp_aic = tmp_mdl.aic
                    a[i,j] = tmp_aic
                    if tmp_aic < best_aic:
                        best_aic = tmp_aic
                        best_order = (i, j)
                        best_mdl = tmp_mdl
                except: continue

        print(k)
        print('Best aic: {:6.5f} | order: {}'.format(best_aic, best_order))


#Heterocedasticity
from statsmodels.stats.diagnostic import het_arch
from statsmodels.stats.diagnostic import het_white

def engle_test_resid(series, order = (1,1)):
    model = smt.ARMA(series, order=order).fit(method='mle', trend='nc')      
    p_val = het_arch(model.resid)[1]
    print(series.name + ' p-value Engle: ' + str(p_val))
    with open(os.getcwd() + '/model_results/ARMA_'+ series.name + '.tex', "w") as text_file:
        text_file.write(model.summary().as_latex())
    return p_val

def engle_test_series(data, Returns):
    products = sorted(data.keys())
    for p in products:
        print(p + ':  p-value Engle: ' + str(het_arch(Returns[p])[1]))

#volatility
from arch import arch_model
def GARCH_Model(series):
    model = arch_model(series, vol='GARCH')
    fitted = model.fit()
    with open(os.getcwd() + '/model_results/GARCH_'+ series.name + '.tex', "w") as text_file:
        text_file.write(fitted.summary().as_latex())
    return fitted

def EGARCH_Model(series):
    model = arch_model(series, vol='EGARCH', o =1)
    fitted = model.fit()
    with open(os.getcwd() + '/model_results/EGARCH_'+ series.name + '.tex', "w") as text_file:
        text_file.write(fitted.summary().as_latex())
    return fitted

def GJR(series):
    model = arch_model(series,  p=1, o=1, q=1, vol='GARCH')
    fitted = model.fit()
    with open(os.getcwd() + '/model_results/GJR_'+ series.name + '.tex', "w") as text_file:
        text_file.write(fitted.summary().as_latex())
    return fitted

def MSV_fit(series):
     # Fit the model
    mod_kns = sm.tsa.MarkovRegression(series, k_regimes=2, trend='nc', switching_variance=True)
    res_kns = mod_kns.fit()
    return res_kns

def MSV_save_results (fitted_model, label):
    with open(os.getcwd() + '/model_results/MSV_'+ label + '.tex', "w") as text_file:
        text_file.write(fitted_model.summary().as_latex())

def MSV_plot(series, fitted_model, label):
    fig, axes = plt.subplots(3, figsize=(10,7))
    
    ax=axes[0]
    ax.plot(series)
    ax.set_ylabel('Retorno')
    ax.set_title(series.name + ' retornos')
    
    ax = axes[1]
    ax.plot(fitted_model.smoothed_marginal_probabilities[0])
    ax.set(title='Probabilidade de um regime de baixa variância para retornos futuros')
    ax.set_ylabel('Probabilidade')

    ax = axes[2]
    ax.plot(fitted_model.smoothed_marginal_probabilities[1])
    ax.set(title='Probabilidade de um regime de alta variância para retornos futuros')
    ax.set_ylabel('Probabilidade')
    fig.tight_layout()

    fig.savefig(ospath(os.getcwd() + '/figs/'+ label +'MSV.png'))
    
def MSV (series):
    fitted_model = MSV_fit(series)
    MSV_save_results(fitted_model, series.name)
    MSV_plot(series, fitted_model, series.name)
    return fitted_model

def compute_vol(r, arch=False):
    if arch:
        fitted_model = MSV_fit(r)
        initial_state_probs = fitted_model.smoothed_marginal_probabilities.iloc[-1]
        transition_matrix = fitted_model.regime_transition.T
        states_vol = np.array([np.sqrt(fitted_model.params[2]), np.sqrt(fitted_model.params[3])]).reshape((2,1))
        std= np.dot(np.dot(initial_state_probs, transition_matrix), states_vol)[0][0]
    else:
        std= r.std()
    print(r.name, std)
    return std
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 15:36:22 2020

@author: nicol_000
"""

import pandas as pd
import numpy as np
from Functions_MFMP import *
import matplotlib.pyplot as plt
import statsmodels.api as sm
import os


cwd = os.getcwd() +"/"
########## Importation des données et retraitements ##########

### Import des séries du fichiers ###
base = pd.read_excel(cwd+'DATA_REPLICATION.xlsx', 'Base', index_col=0)
macro = pd.read_excel(cwd+'DATA_REPLICATION.xlsx', 'Macro', index_col=0)
libor = pd.read_excel(cwd+'DATA_REPLICATION.xlsx', 'Libor', index_col=0)

### Excess returns over USD Libor 1 month for non spread base assets ###
non_spread_assets = ['WEQ', 'GLT', 'GOLD', 'INM', 'ENG', 'DXY']
base[non_spread_assets] = base[non_spread_assets] - libor.values / 12

### Turbulence Index by Chow ###
Turb_index = Turbulence_Index(base)

# Calcul du facteur de Financial Stress
macro['Financial Stress'] = ((macro['Financial Stress'].values + Turb_index.loc[:np.shape(Turb_index)[0] - 13, 0]) / 2).values

### Z-score sur les données macros ###
macro = (macro - macro.mean()) / (macro.std() * 100)
macro.std()
########## Statistiques descriptives ##########

### Corrélation totale ###
total_corr = np.corrcoef(macro, rowvar=False)

### Corrélation partielle ###
partial_corr = par_corr(macro)

# Observations:
# - Le facteur macro Growth est le facteur dominant parmi les 3 facteurs macroéconomiques de l'étude car influence
#   les autres variables sans être lui-même influencé
# - Les variables macréoconomiques sont corrélées, d'où l'importance du framework GFMP permettant
#   de prendre en compte la corrélation de ces variables

########## Représentation graphique des séries macroéconomiques normalisées ##########
graph_macro = macro.reset_index()
#plt.plot('Date', 'Growth', data=graph_macro)
#plt.show()
#plt.plot('Date', 'Inflation surprises', data=graph_macro)
#plt.show()
#plt.plot('Date', 'Financial Stress', data=graph_macro)
#plt.show()
########## OLS Regression ##########

### Version capitalisée ###
base_capi = np.cumprod(1 + base.iloc[::-1]).iloc[::-1]
base_rolling = (base_capi / base_capi.shift(-12) - 1)[:-12:]

### Version sommée ###
base_s_rolling = base.iloc[::-1].rolling(12).sum().iloc[::-1][:-12:]

### Facteurs = données macroéconomiques ###
facteurs = macro
macro_factors = macro.columns.tolist()

########## Estimation des Macro-FMPs à partir de régressions OLS/GLS ##########

##### OLS  Multivariate regression of base asset returns on macro factors (1st Regression) #####
base_assets = base_rolling.columns.tolist()
OLS_results = {}
loadings = pd.DataFrame(np.zeros((9, 4)), columns=['Intercept', 'Growth', 'Inflation surprises', 'Financial Stress'],
                        index=base_assets)

X_macro = sm.add_constant(facteurs)

for asset in base_assets:
    Y = base_rolling[asset]
    model = sm.OLS(Y, X_macro)
    OLS_results[asset] = model.fit()
    loadings.loc[asset] = model.fit().params.values

Betas = loadings.drop(columns=['Intercept'])

# Betas_art = [0.05,8.20,-0.20,-3.54]
# Y_art = np.dot(X_macro,Betas_art)
# dates = X_macro.reset_index()['Date']
# plt.plot(dates,Y_art)
# plt.plot(dates,Y)
# correl = np.corrcoef(Y_art,Y)
Base_cov = np.cov(base_rolling, rowvar=False)

####### Two-Pass Cross Sectional Regression Approaches (CSR) #######
#######          GLS CSR - Lehman and Modest(1988)           #######

### Estimation des Facteurs (2nd Regression) ###
Y_base = base_rolling.mean()
X_loadings = sm.add_constant(Betas)
model = sm.OLS(Y_base, X_loadings)
Factors_results = model.fit()
Factors_estim = model.fit().params

### Estimation Poids GLS-CSR  ###
Left_CSR = np.dot(np.linalg.inv(Base_cov), Betas)
Right_CSR = np.linalg.inv(np.dot(Betas.T, np.dot(np.linalg.inv(Base_cov), Betas)))
CSR_Weights = pd.DataFrame(np.dot(Left_CSR, Right_CSR), columns=macro_factors, index=base_assets)
CSR_Weights_Scaled = CSR_Weights / (CSR_Weights.std() * 100)

### Calcul des Betas des Portefeuilles ###
CSR_Beta_Growth = np.dot(Betas.T, CSR_Weights['Growth'])
CSR_Beta_IS = np.dot(Betas.T, CSR_Weights['Inflation surprises'])
CSR_Beta_FS = np.dot(Betas.T, CSR_Weights['Financial Stress'])

CSR_Beta_PF = pd.DataFrame(np.array([CSR_Beta_Growth, CSR_Beta_IS, CSR_Beta_FS]), columns=macro_factors,
                           index=macro_factors)

### Série de facteurs macros estimée ###
CSR_macro = pd.DataFrame(np.dot(base_rolling, CSR_Weights), columns=macro_factors, index=facteurs.index)




### Corrélation partielle avec les facteurs observés
CSR_par_corr = PF_par_corr(CSR_macro, facteurs, macro_factors)

#######          Maximum Correlation Approaches (MCP)         #######
####### MCP - Huberman et al., 1987, and Breeden et al., 1989 #######

### OLS univariate regression of base asset returns on macro factors (1st Regression) ###
uni_loadings = pd.DataFrame(np.zeros((9, 3)), columns=macro_factors, index=base_assets)

for asset in base_assets:
    for F in macro_factors:
        Y = base_rolling[asset]
        X = facteurs[F]
        model = sm.OLS(Y, X)
        uni_loadings[F].loc[asset] = model.fit().params.values

Betas_uni = uni_loadings

### Estimation Poids MCP  ###
MCP_Weights = pd.DataFrame(np.dot(np.linalg.inv(Base_cov), Betas_uni), columns=macro_factors, index=base_assets)
MCP_Weights_Scaled = MCP_Weights / (MCP_Weights.std() * 100)

### Calcul des Betas des Portefeuilles ###
MCP_Beta_Growth = np.dot(Betas_uni.T, MCP_Weights['Growth'])
MCP_Beta_IS = np.dot(Betas_uni.T, MCP_Weights['Inflation surprises'])
MCP_Beta_FS = np.dot(Betas_uni.T, MCP_Weights['Financial Stress'])

MCP_Beta_PF = pd.DataFrame(np.array([MCP_Beta_Growth, MCP_Beta_IS, MCP_Beta_FS]), columns=macro_factors,
                           index=macro_factors)

### Calcul de la matrice objectif BK (Matrice Objectif)  ###
Factors_cov = np.cov(facteurs, rowvar=False)
BK_MCP = pd.DataFrame(np.dot(Betas_uni.T, np.dot(np.linalg.inv(Base_cov), Betas_uni)), columns=macro_factors,
                      index=macro_factors)

### Série de facteurs macros estimée ###
MCP_macro = pd.DataFrame(np.dot(base_rolling, MCP_Weights), columns=macro_factors, index=facteurs.index)

### Corrélation partielle avec les facteurs observés
MCP_par_corr = PF_par_corr(MCP_macro, facteurs, macro_factors)

#######        Generalized Factor Mimicking Portfolios        #######
####### GFMP - Emmanuel Jurczenko and Jérôme Teiletche (2019) #######

##### OLS  Multivariate regression of base asset returns on macro factors (1st Regression) #####
Betas_multi = Betas
Base_inv_cov = np.linalg.inv(Base_cov)
BK_GFMP = np.cov(facteurs, rowvar=False)

### Estimation Poids GFMP  ###
Left_GFMP = np.dot(Base_inv_cov, Betas_multi)
Mid_GFMP = np.linalg.inv(np.dot(Betas_multi.T, np.dot(Base_inv_cov, Betas_multi)))
Right_GFMP = BK_GFMP

GFMP_Weights = pd.DataFrame(np.dot(Left_GFMP, np.dot(Mid_GFMP, Right_GFMP)), columns=macro_factors, index=base_assets)
GFMP_Weights_Scaled = GFMP_Weights / (GFMP_Weights.std() * 100)

### Calcul des Betas des Portefeuilles ###
GFMP_Beta_Growth = np.dot(Betas_multi.T, GFMP_Weights['Growth'])
GFMP_Beta_IS = np.dot(Betas_multi.T, GFMP_Weights['Inflation surprises'])
GFMP_Beta_FS = np.dot(Betas_multi.T, GFMP_Weights['Financial Stress'])

GFMP_Beta_PF = pd.DataFrame(np.array([GFMP_Beta_Growth, GFMP_Beta_IS, GFMP_Beta_FS]), columns=macro_factors,
                            index=macro_factors)

### Série de facteurs macros estimée ###
GFMP_macro = pd.DataFrame(np.dot(base_rolling, GFMP_Weights), columns=macro_factors, index=facteurs.index)

### Corrélation partielle avec les facteurs observés
GFMP_par_corr = PF_par_corr(GFMP_macro, facteurs, macro_factors)

##########    Amélioration de l'estimation des Macro-FMPs    ##########

##### PCs extraction from Covariance matrix of base asset returns #####
from sklearn.decomposition import PCA
from sklearn.linear_model import LassoLarsIC

### Etape 1 : Extraction des n premières composantes principales ###
pca = PCA(n_components=9)
pca_results = pca.fit_transform(base_rolling)
F_macros = pd.DataFrame(np.zeros(np.shape(facteurs)), columns=macro_factors, index=facteurs.index)

### Etape 2 : Sélection des Composantes principales pertinentes pour chaque facteur Macroéconomique ###
for F in macro_factors:
    model_bic = LassoLarsIC(criterion='bic')
    model_bic.fit(pca_results, facteurs[F])
    LASSO_coeffs = model_bic.coef_
    F_macros[F] = np.dot(pca_results, LASSO_coeffs)

### Etape 3 : Regréssion des actifs sur les facteurs estimés à partir des composantes principales ###
OLS_post_LASSO_results = {}
loadings_ML = pd.DataFrame(np.zeros((9, 3)), columns=macro_factors, index=base_assets)

for asset in base_assets:
    Y = base_rolling[asset]
    model = sm.OLS(Y, F_macros)
    OLS_post_LASSO_results[asset] = model.fit()
    loadings_ML.loc[asset] = model.fit().params.values

Betas_ML = loadings_ML

### Estimation Poids GFMP (Machine Learning) ###
Base_inv_cov = np.linalg.inv(Base_cov)
BK_GFMP_ML = np.cov(F_macros, rowvar=False)

Left_GFMP_ML = np.dot(Base_inv_cov, Betas_ML)
Mid_GFMP_ML = np.linalg.inv(np.dot(Betas_ML.T, np.dot(Base_inv_cov, Betas_ML)))
Right_GFMP_ML = BK_GFMP_ML

GFMP_ML_Weights = pd.DataFrame(np.dot(Left_GFMP_ML, np.dot(Mid_GFMP_ML, Right_GFMP_ML)), columns=macro_factors,
                               index=base_assets)
GFMP_ML_Weights_Scaled = GFMP_ML_Weights / (GFMP_ML_Weights.std() * 100)

dates = X_macro.reset_index()['Date']
FMP_Estim = pd.DataFrame(np.dot(base_rolling, GFMP_ML_Weights_Scaled), columns=macro_factors, index=facteurs.index)

facteurs.to_csv('observed_infla_growth_stress.csv')
FMP_Estim.to_csv('FMP_estimation_infla_growth_stress.csv')

### Growth p.23 ###
"""
plt.title('Growth')
plt.plot(dates, facteurs['Growth'], label='Observed')
plt.plot(dates, FMP_Estim['Growth'], label='GFMP')
plt.legend(loc='best')
plt.show()"""

### Inflation surprises p.23 ###
plt.title('Inflation surprises')
plt.plot(dates, facteurs['Inflation surprises'], label='Observed')
plt.plot(dates, FMP_Estim['Inflation surprises'], label='GFMP')
plt.legend(loc='best')
plt.show()
### Financial Stress p.23 ###
plt.title('Financial Stress')
plt.plot(dates, facteurs['Financial Stress'], label='Observed')
plt.plot(dates, FMP_Estim['Financial Stress'], label='GFMP')
plt.legend(loc='best')
plt.show()
### Growth & Inflation surprises combined ###
# Problème corrélation car les grandes variations d'inflations moins prises en compte par les GFMPs
# Donc corrélation plus élevée entre GFMPs que pour les variables réelles
# Pose problème pour les régressions
plt.plot(dates, facteurs['Growth'], label='Growth Observed')
plt.plot(dates, facteurs['Inflation surprises'], label='IS Observed')
plt.legend(loc='best')
plt.show()
"""
plt.plot(dates, FMP_Estim['Growth'], label='Growth GFMP')
plt.plot(dates, FMP_Estim['Inflation surprises'], label='IS GFMP')
plt.legend(loc='best')
plt.show()"""

"""
# PARTIE 5.1
MoMandValue = pd.read_excel('MomentumAndValue.xlsx', index_col=0)
MoMandValue = MoMandValue.set_index(facteurs.index)

MoMVal = MoMandValue.columns.tolist()

Corr_MomVal = MoMandValue.corr()
Corr_MomVal = Corr_MomVal.loc['MOMENTUM']['VALUE']

OLS_factor_LASSO_results = {}
loadings_MomVal = pd.DataFrame(np.zeros((2, 3)), columns=macro_factors, index=MoMVal)

for factor in MoMVal:
    Y_bis = MoMandValue[factor]
    model_bis = sm.OLS(Y_bis, F_macros)
    OLS_factor_LASSO_results[factor] = model_bis.fit()
    loadings_MomVal.loc[factor] = model_bis.fit().params.values

Betas_MomVal = loadings_MomVal

estim_Momentum = loadings_MomVal.loc['MOMENTUM'] * F_macros
estim_Momentum = estim_Momentum['Growth'] + estim_Momentum['Inflation surprises'] + estim_Momentum['Financial Stress']

estim_Value = loadings_MomVal.loc['VALUE'] * F_macros
estim_Value = estim_Value['Growth'] + estim_Value['Inflation surprises'] + estim_Value['Financial Stress']

estimate = pd.DataFrame({'MOMENTUM': estim_Momentum, 'VALUE': estim_Value})

estim_Corr_MomVal = estimate.corr()
estim_Corr_MomVal = estim_Corr_MomVal.loc['MOMENTUM']['VALUE']

compare_corr = pd.DataFrame({'CORRELATION': [Corr_MomVal, estim_Corr_MomVal]}, index=['OBSERVED', 'IMPLIED'])

########## Practical use of macro factor mimicking portfolios ########
##########   Hedging macro risks for a multi-asset portfolio  ##########

##### Institutional Portfolio Construction #####
Index_Names = ['EQ', 'IG', 'HY', 'HFR']
Instit = pd.read_excel('DATA_REPLICATION.xlsx', 'Institutional', index_col=0).drop(columns=['PE'])
I_PF_Weights = [0.7, 0.15, 0.05, 0.1]
I_PF_Rdt = pd.DataFrame(np.dot(Instit, I_PF_Weights), columns=['Institutional Portfolio'], index=Instit.index)
# I_PF_Rdt.mean()
I_PF = I_PF_Rdt.resample('Q', convention='end').agg('sum')
I_PF_rolling = I_PF.rolling(4).sum()[3:]

##### Exposure of the diversified institutional portfolio to ML macro GFMPs #####
start_dt = I_PF_rolling.index[0]
end_dt = I_PF_rolling.index[-1]

FMP_Estim_Q = FMP_Estim.resample('Q', convention='end').agg('sum')
X_mod = FMP_Estim_Q.ix[start_dt:end_dt]
X_mod.index = I_PF_rolling.index
X_mod['Inflation surprises'] = X_mod['Inflation surprises'] - X_mod[
    'Growth']  # Annule la corrélation entre Growth et Inflation surprises
X = sm.add_constant(X_mod)
Y = I_PF_rolling
model = sm.OLS(Y, X)
OLS_IPF = model.fit()
Exposure_IPF = pd.DataFrame(model.fit().params.values, columns=['Exposure'],
                            index=['Intercept', 'Growth', 'Inflation', 'Stress'])
Adj_R2_IPF = model.fit().rsquared_adj

##### Institutional diversified portfolio and its macro-hedged version : Performances and Drawdowns #####
Macro_Hedge_weights = Exposure_IPF
Macro_Hedge_PF = np.dot(X_mod, Macro_Hedge_weights.iloc[1:, :])

Hedged_PF = I_PF_rolling - Macro_Hedge_PF  # Portefeuille autofinancé car base assets sont tous des spreads (vs cash ou autre actif)
Hedged_PF = Hedged_PF.rename(columns={'Institutional Portfolio': 'Macro-Hedged Portfolio'})

### Performances (one-year rolling windows) ###
plt.plot(I_PF_rolling.index, I_PF_rolling, label='Institutional Diversified Portfolio')
plt.plot(I_PF_rolling.index, Hedged_PF, label='Macro-Hedged Portfolio')
plt.legend(loc='best')
plt.title('Performances (one-year rolling windows)')
plt.show()
##### Maximum Drawdown #####
PF_Names = ['Institutional Portfolio', 'Macro-Hedged Portfolio']
Max_DD = pd.DataFrame(np.zeros([np.shape(I_PF_rolling)[0], 2]), columns=PF_Names, index=I_PF_rolling.index)

Cum_I_PF = np.cumprod(1 + I_PF_rolling)
Cum_H_PF = np.cumprod(1 + Hedged_PF)

for i in Max_DD.index:
    # Institutional Portfolio#
    Temp_Max = Cum_I_PF.loc[:i].max()
    DD_I = (Cum_I_PF.loc[i] / Temp_Max) - 1
    Max_DD['Institutional Portfolio'].loc[i] = -np.abs(DD_I).max()

    # Macro-Hedged Portfolio #
    Temp_Max = Cum_H_PF.loc[:i].max()
    DD_H = (Cum_H_PF.loc[i] / Temp_Max) - 1
    Max_DD['Macro-Hedged Portfolio'].loc[i] = -np.abs(DD_H).max()

plt.plot(Max_DD.index, Max_DD['Institutional Portfolio'], label='Institutional Portfolio')
plt.plot(Max_DD.index, Max_DD['Macro-Hedged Portfolio'], label='Macro-Hedged Portfolio')

##### Portfolio Return Metrics #####
Metrics_Instit = {}
Metrics_Hedged = {}

### Institutional Diversified Portfolio ###
Metrics_Instit['Average'] = I_PF_rolling.mean().values
Metrics_Instit['Volatility'] = I_PF_rolling.std().values
Metrics_Instit['Minimum'] = I_PF_rolling.min().values
Metrics_Instit['Maximum'] = I_PF_rolling.max().values
Metrics_Instit['Skewness'] = I_PF_rolling.skew().values
Metrics_Instit['Kurtosis'] = I_PF_rolling.kurt().values

### Macro-Hedged Institutional Diversified Portfolio ###
Metrics_Hedged['Average'] = Hedged_PF.mean().values
Metrics_Hedged['Volatility'] = Hedged_PF.std().values
Metrics_Hedged['Minimum'] = Hedged_PF.min().values
Metrics_Hedged['Maximum'] = Hedged_PF.max().values
Metrics_Hedged['Skewness'] = Hedged_PF.skew().values
Metrics_Hedged['Kurtosis'] = Hedged_PF.kurt().values

# Portfolio returns - Density Plot #
import seaborn as sns

sns.distplot(I_PF_rolling['Institutional Portfolio'], hist=False, kde=True, color='darkblue', kde_kws={'linewidth': 4},
             label='Institutional Portfolio')
sns.distplot(Hedged_PF['Macro-Hedged Portfolio'], hist=False, kde=True, color='green', kde_kws={'linewidth': 4},
             label='Macro-Hedged Portfolio')

########## Practical use of macro factor mimicking portfolios ########
##########              Extension to April 2020             ##########

##### Nowcasting #####

### Import des séries du fichiers ###
base_april = pd.read_excel('DATA_EXTENSION.xlsx', 'Base', index_col=0)
libor_april = pd.read_excel('DATA_EXTENSION.xlsx', 'Libor', index_col=0)
instit_april = pd.read_excel('DATA_EXTENSION.xlsx', 'Instit', index_col=0)

### Excess returns over USD Libor 1 month for non spread base assets ###
nsa_april = ['WEQ', 'GLT', 'GOLD', 'INM', 'ENG', 'DXY']
base_april[non_spread_assets] = base_april[non_spread_assets] - libor_april.values / 12

### Version capitalisée ###
base_c_april = np.cumprod(1 + base_april.iloc[::-1]).iloc[::-1]
base_r_april = (base_c_april / base_c_april.shift(-12) - 1)[:-12:]

### Nowcasting estimations with GFMP Machine Learning Method ###
GFMP_ML_Model = GFMP_ML_Weights_Scaled

Dates_Extension = base_r_april.reset_index()['Date']
FMP_Estim_April = pd.DataFrame(np.dot(base_r_april, GFMP_ML_Weights_Scaled), columns=macro_factors,
                               index=Dates_Extension)

### Hedging Macro - April 2020 ###
Index_Names = ['EQ', 'IG', 'HY', 'HFR']
I_PF_W_April = [0.7, 0.15, 0.05, 0.1]
I_PF_R_April = pd.DataFrame(np.dot(instit_april, I_PF_W_April), columns=['Institutional Portfolio'],
                            index=instit_april.index)
I_PF_April = I_PF_R_April.resample('Q', convention='end').agg('sum')
I_PF_R_April = I_PF_April.rolling(4).sum()[3:]

start_dt_April = I_PF_R_April.index[0]
end_dt_April = I_PF_R_April.index[-1]

FMP_Estim_April_Q = FMP_Estim_April.resample('Q', convention='end').agg('sum')
FMP_Estim_Ap_Q = FMP_Estim_April_Q.ix[start_dt_April:end_dt_April]

X_mod_ap = FMP_Estim_Ap_Q.ix[start_dt_April:end_dt_April]
X_mod_ap.index = I_PF_R_April.index
X_mod_ap['Inflation surprises'] = X_mod_ap['Inflation surprises'] - X_mod_ap[
    'Growth']  # Annule la corrélation entre Growth et Inflation surprises
X_ap = sm.add_constant(X_mod_ap)
Y_ap = I_PF_R_April
model_April = sm.OLS(Y_ap, X_ap)
OLS_IPF_April = model_April.fit()
Exposure_IPF_April = pd.DataFrame(model_April.fit().params.values, columns=['Exposure'],
                                  index=['Intercept', 'Growth', 'Inflation', 'Stress'])
Adj_R2_IPF_April = model_April.fit().rsquared_adj

##### Institutional diversified portfolio and its macro-hedged version : Performances and Drawdowns #####
Macro_Hedge_weights_April = Exposure_IPF_April
Macro_Hedge_PF_April = np.dot(X_mod_ap, Macro_Hedge_weights_April.iloc[1:, :])

plt.plot(I_PF_R_April.index, I_PF_R_April, label="PF")
plt.plot(I_PF_R_April.index, Macro_Hedge_PF_April, label="Macro")
plt.legend(loc='best')
plt.show()

Hedged_PF_April = I_PF_R_April - Macro_Hedge_PF_April  # Portefeuille autofinancé car base assets sont tous des spreads (vs cash ou autre actif)
Hedged_PF_April = Hedged_PF_April.rename(columns={'Institutional Portfolio': 'Macro-Hedged Portfolio'})

### Performances (one-year rolling windows) ###
plt.plot(I_PF_R_April.index, I_PF_R_April, label='Institutional Diversified Portfolio')
plt.plot(I_PF_R_April.index, Hedged_PF_April, label='Macro-Hedged Portfolio')
plt.legend(loc='best')
plt.title('Performances (one-year rolling windows)')
plt.show()

# Calcul des Maximum Drawdown
PF_Names = ['Institutional Portfolio', 'Macro-Hedged Portfolio']
Max_DD_April = pd.DataFrame(np.zeros([np.shape(I_PF_R_April)[0], 2]), columns=PF_Names, index=I_PF_R_April.index)

Cum_I_PF_April = np.cumprod(1 + I_PF_R_April)
Cum_H_PF_April = np.cumprod(1 + Hedged_PF_April)

for i in Max_DD_April.index:
    # Institutional Portfolio#
    Temp_Max = Cum_I_PF_April.loc[:i].max()
    DD_I_April = (Cum_I_PF_April.loc[i] / Temp_Max) - 1
    Max_DD_April['Institutional Portfolio'].loc[i] = -np.abs(DD_I_April).max()

    # Macro-Hedged Portfolio #
    Temp_Max = Cum_H_PF_April.loc[:i].max()
    DD_H_April = (Cum_H_PF_April.loc[i] / Temp_Max) - 1
    Max_DD_April['Macro-Hedged Portfolio'].loc[i] = -np.abs(DD_H_April).max()

plt.plot(Max_DD_April.index, Max_DD_April['Institutional Portfolio'], label='Institutional Portfolio')
plt.plot(Max_DD_April.index, Max_DD_April['Macro-Hedged Portfolio'], label='Macro-Hedged Portfolio')
plt.legend(loc='lower left')
plt.show()

"""
# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np
from scipy import linalg, stats


##### TURBULENCE INDEX - CHOW (1999) #####

def Turbulence_Index(Data):
    # Calcul de la distance de Mahalanobis #
    x_m = Data - np.mean(Data)
    x_m_T = (x_m).T
    Cov = np.cov(Data, rowvar=False)
    Cov_inv = linalg.inv(Cov)
    M_Dist = np.sqrt(np.dot(x_m, np.dot(Cov_inv, x_m_T)).diagonal())

    # Calcul de l'index de Turbulence de Chow #
    Turb_index = pd.DataFrame(M_Dist).rolling(6).mean().fillna(method='bfill')
    return Turb_index


##### CORRELATION PARTIELLE #####

def par_corr(Data):
    C = np.asarray(Data)
    p = C.shape[1]
    P_corr = np.zeros((p, p), dtype=np.float)
    for i in range(p):
        P_corr[i, i] = 1
        for j in range(i + 1, p):
            idx = np.ones(p, dtype=np.bool)
            idx[i] = False
            idx[j] = False
            beta_i = linalg.lstsq(C[:, idx], C[:, j])[0]
            beta_j = linalg.lstsq(C[:, idx], C[:, i])[0]

            res_j = C[:, j] - C[:, idx].dot(beta_i)
            res_i = C[:, i] - C[:, idx].dot(beta_j)

            corr = stats.pearsonr(res_i, res_j)[0]
            P_corr[i, j] = corr
            P_corr[j, i] = corr

    return P_corr


def PF_par_corr(Serie, facteurs, macro_factors):
    Output = {}
    for macro_name in macro_factors:
        temp_macro = facteurs.drop(columns=macro_name)
        temp_macro[macro_name] = Serie[macro_name]
        partial_corr = pd.DataFrame(par_corr(temp_macro), columns=macro_factors, index=macro_factors)
        Output[macro_name] = partial_corr
    return Output
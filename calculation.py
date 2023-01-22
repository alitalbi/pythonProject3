from scipy.stats import norm
import numpy as np


import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import newton

# Données de prix d'options pour différents strikes, à une maturité donnée
call_prices = [10.5, 12, 14.5, 16, 18.5, 20,21.5,23]
strikes = [90, 95, 100, 105, 110, 115]

# Données de l'actif sous-jacent pour la même période
S0 = 100

# Informations sur le taux d'intérêt
r0 = 0.05

# Maturité donnée
T = 1

sigma_r = 0.03
rho = 0.4

# Calcul de la volatilité implicite pour chaque strike
import numpy as np
from scipy.stats import norm

N_prime = norm.pdf
N = norm.cdf


def black_scholes_call(S, K, T, r, sigma_S,sigma_r=0.03):
    '''

    :param S: Asset price
    :param K: Strike price
    :param T: Time to maturity
    :param r: risk-free rate (treasury bills)
    :param sigma: volatility
    :return: call price
    '''

    ###standard black-scholes formula
    d1 = (np.log(S0/K) + (r0 + (1/2) * sigma_S**2 - (1/2) * sigma_r**2) * T) / (sigma_S * np.sqrt(T)) + (sigma_S * rho * sigma_r) / np.sqrt(T)
    d2 = d1 - sigma_S * np.sqrt(T)

    call = S * N(d1) - N(d2) * K * np.exp(-r * T)
    return call


def vega(S, K, T, r, sigma_S,sigma_r=0.03):
    '''

    :param S: Asset price
    :param K: Strike price
    :param T: Time to Maturity
    :param r: risk-free rate (treasury bills)
    :param sigma: volatility
    :return: partial derivative w.r.t volatility
    '''

    ### calculating d1 from black scholes
    d1 = (np.log(S0/K) + (r0 + (1/2) * sigma_S**2 - (1/2) * sigma_r**2) * T) / (sigma_S * np.sqrt(T)) + (sigma_S * rho * sigma_r) / np.sqrt(T)

    # see hull derivatives chapter on greeks for reference
    vega = S * N_prime(d1) * np.sqrt(T)
    return vega


def implied_volatility_call(C, S, K, T, r, tol=0.0001,
                            max_iterations=100):
    '''

    :param C: Observed call price
    :param S: Asset price
    :param K: Strike Price
    :param T: Time to Maturity
    :param r: riskfree rate
    :param tol: error tolerance in result
    :param max_iterations: max iterations to update vol
    :return: implied volatility in percent
    '''

    ### assigning initial volatility estimate for input in Newton_rap procedure
    sigma = 0.3

    for i in range(max_iterations):

        ### calculate difference between blackscholes price and market price with
        ### iteratively updated volality estimate
        diff = black_scholes_call(S, K, T, r, sigma) - C

        ###break if difference is less than specified tolerance level
        if abs(diff) < tol:
            print(f'found on {i}th iteration')
            print(f'difference is equal to {diff}')
            break

        ### use newton rapshon to update the estimate
        sigma = sigma - diff / vega(S, K, T, r, sigma)

    return sigma

implied_vols = []
for i in range(len(strikes)):
    K = strikes[i]
    # Initial guess for the solution
    x0 = 0.5
    # Increase the maximum number of iterations
   # maxiter = 10000
    # Increase the tolerance

    implied_vols.append(implied_volatility_call(call_prices[i],S0,strikes[i],T,r0))


# Tracer la volatilité implicite en fonction du strike
plt.plot(strikes, implied_vols)
plt.xlabel('Strike')
plt.ylabel('Volatilité implicite')
plt.show()
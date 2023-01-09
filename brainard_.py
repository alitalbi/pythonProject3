import pandas as pd
import matplotlib.pyplot as plt
from fredapi import Fred
import datetime
import numpy as np
import yfinance as yf
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
# Evaluation du training set
from sklearn.metrics import r2_score
import statsmodels.api as sm

def regression(y,x):
    X_train, X_test, Y_train, Y_test = train_test_split(x, y,
                                                        test_size=0.3, random_state=5)

    lmodellineaire = LinearRegression()
    lmodellineaire.fit(X_train, Y_train)
    y_train_predict = lmodellineaire.predict(X_train)
    rmse = (np.sqrt(mean_squared_error(Y_train, y_train_predict)))
    r2 = r2_score(Y_train, y_train_predict)

    print('-------------TRAIN SET-------------------------')
    print('Lerreur quadratique moyenne est {}'.format(rmse))
    print('le score R2 est {}'.format(r2))
    print('\n')

    # model evaluation for testing set
    y_test_predict = lmodellineaire.predict(X_test)

    print('-------------TEST SET-------------------------')
    rmse = (np.sqrt(mean_squared_error(Y_test, y_test_predict)))
    r2 = r2_score(Y_test, y_test_predict)
    print('Lerreur quadratique moyenne est {}'.format(rmse))
    print("R2 : {}".format(r2))

    print('La performance du modèle sur la base de test')
    print('--------------------------------------')
    print('Lerreur quadratique moyenne est {}'.format(rmse))

date_start = "2018-01-01"
date_end = "2022-12-27"

PATH_DATA = "/Users/talbi/Downloads/"
fred = Fred(api_key='f40c3edb57e906557fcac819c8ab6478')
_30y = pd.read_csv(PATH_DATA+"30y.csv").iloc[:,:2]
_30y.set_index("Date",inplace=True,drop=True)
_30y['Dernier'] = _30y['Dernier'].apply(lambda x:float(x.replace(",",".")))
_30y_index = pd.Series(_30y.index.to_list()[::-1]).apply(lambda x:datetime.datetime.strptime(x,"%d/%m/%Y"))
_30y = _30y[::-1]
_30y.index = _30y_index







_5y = pd.DataFrame(fred.get_series("DFII5", observation_start="1970-01-01", observation_end="2022-12-30", freq="daily"))
_5y.columns = ['Dernier']

spread = _30y - _5y
def pct_changee(x):
    if len(x):
        return (x[-1] - x[0]) / x[0]
cwd = os.getcwd()+"/"
max=1000
_30y = pd.read_csv(cwd + "_30y.csv", index_col="Date")
_30y.index = pd.to_datetime(_30y.index)

_5y_real = pd.read_csv(cwd + "_5y_real.csv", index_col="Unnamed: 0")
_5y_real.index = pd.to_datetime(_5y_real.index)

cooper = pd.read_csv(cwd + "cooper_prices.csv", index_col="Date")
cooper.index = pd.to_datetime(cooper.index)

_5y_nominal = pd.read_csv(cwd + "_5y_nominal.csv", index_col="Date")
_5y_nominal.index = pd.to_datetime(_5y_nominal.index)

print("8")
spread = _30y - _5y_real

merged = pd.concat([cooper,spread,_5y_nominal],axis=1)

merged.columns = ["cooper","spread_30y_5y","5y_nominal"]
merged.dropna(inplace=True)
average = merged.rolling(90).mean()

# Calculate the annualized growth rate
annualized_6m_smoothed_growth_rate = ((merged[90:] / average) ** (12 / 3) - 1)[90:]


annualized_6m_smoothed_growth_rate.dropna(inplace=True)
for col in annualized_6m_smoothed_growth_rate.columns:
    annualized_6m_smoothed_growth_rate[col+"_dummy"] = np.where(annualized_6m_smoothed_growth_rate[col]>0,1,0)

annualized_6m_smoothed_growth_rate['regimes'] = annualized_6m_smoothed_growth_rate[['cooper_dummy',"spread_30y_5y_dummy"]].sum(axis=1)

regimes_5y_nominal =annualized_6m_smoothed_growth_rate[["regimes","5y_nominal_dummy"]]


regimes_5y_nominal.plot()
plt.show()
print("hi")
#cooper_norm = (spread - np.mean(spread)) / np.std(spread)
"""cooper_norm = (cooper - np.mean(cooper)) / np.std(cooper)
spread_coop_diff = spread_norm - cooper_norm
#plt.plot(cooper.index.to_list(),cooper[['Close']])
merged_data = pd.concat([spread_coop_diff,spread_norm,cooper_norm,_5y_nominal],axis=1)
merged_data.columns = ["30y_5y_spread - Cooper (normalized)","spread normalized", "cooper normalized","5y nominal"]
merged_data[['30y_5y_spread - Cooper (normalized)','5y nominal']].plot()
plt.title("30y_5y_spread - Cooper (normalized)  vs   5y nominal")
plt.xlabel("30y_5y_spread - Cooper (normalized)")
plt.ylabel("5y nominal")
plt.plot()
#merged_data.plot()
merged_data.dropna(inplace=True)

merged_data['X'] = (merged_data[['30y_5y_spread - Cooper (normalized)']]).resample("3M").last().diff()
X_diff = merged_data['X']
merged_data['X_no_diff'] = (merged_data[['30y_5y_spread - Cooper (normalized)']]).resample("3M").last()
merged_data['Y'] = merged_data['5y nominal'].resample("3M").last().diff()
merged_data['dummy_5y_nominal']  = np.where(merged_data['X'] >=0,1,-1)
merged_data.dropna(inplace=True)

regression(merged_data[['Y']],merged_data[['X','cooper normalized','spread normalized']])
#merged_.plot()

print(merged_data.corr(method='spearman'))
plt.show()"""


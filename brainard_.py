import pandas as pd
import matplotlib.pyplot as plt
from fredapi import Fred
import datetime
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
# Evaluation du training set
from sklearn.metrics import r2_score

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


cooper_prices = pd.read_csv(PATH_DATA+"cooper_prices.csv").iloc[:,:2]
cooper_prices.set_index("Date",inplace=True,drop=True)
cooper_prices['Dernier'] = cooper_prices['Dernier'].apply(lambda x:float(x.replace(",",".")))
cooper_prices_index = pd.Series(cooper_prices.index.to_list()[::-1]).apply(lambda x:datetime.datetime.strptime(x,"%d/%m/%Y"))
cooper_prices = cooper_prices[::-1]
cooper_prices.index = cooper_prices_index
merged_data = pd.concat([spread,_5y,cooper_prices],axis=1)
merged_data.dropna(inplace=True)
merged_data.columns = ["spread 30_5yr","5y","cooper"]
merged_data_spread_var = pd.DataFrame(merged_data.iloc[:, 0].diff())
merged_data_5y_var = pd.DataFrame(merged_data.iloc[:, 1].diff())
merged_data_cooper_ret = pd.DataFrame(merged_data.iloc[:, 2].pct_change())
merged_ = pd.concat([merged_data_spread_var,merged_data_5y_var,merged_data_cooper_ret],axis=1)
merged_.dropna(inplace=True)
merged_['dummy_cooper'] = np.where(merged_['cooper']>0,-1,1)

X= merged_[['dummy_cooper','cooper','spread 30_5y']]
Y = merged_['5y']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=5)

lmodellineaire = LinearRegression()
lmodellineaire.fit(X_train, Y_train)
y_train_predict = lmodellineaire.predict(X_train)
rmse = (np.sqrt(mean_squared_error(Y_train, y_train_predict)))
r2 = r2_score(Y_train, y_train_predict)

print('La performance du modèle sur la base dapprentissage')
print('--------------------------------------')
print('Lerreur quadratique moyenne est {}'.format(rmse))
print('le score R2 est {}'.format(r2))
print('\n')

# model evaluation for testing set
y_test_predict = lmodellineaire.predict(X_test)
rmse = (np.sqrt(mean_squared_error(Y_test, y_test_predict)))
r2 = r2_score(Y_test, y_test_predict)

print('La performance du modèle sur la base de test')
print('--------------------------------------')
print('Lerreur quadratique moyenne est {}'.format(rmse))
print('le score R2 est {}'.format(r2))
merged_data.plot()

print(merged_data.corr(method='spearman'))
plt.show()
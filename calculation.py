from fredapi import Fred
import pandas as pd
import matplotlib.pyplot as plt
import wget
import os
#need openpyxl
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
import yfinance as yf
import numpy as np
# Parameters for the request from the FRED Website
PATH_DATA = r"/Users/talbi/Downloads/"



frequency = 'monthly'
d= yf.download("LQD", start="1971-01-01", end="2018-06-29", interval="1mo")[['Adj Close']]
d.index = pd.to_datetime(d.index)
d.to_excel(PATH_DATA+"LQD.xlsx")
print(d)
d.plot()
plt.show()

def commo_smooth_data(internal_ticker, date_start,date_start2,date_end):
    data_ = pd.read_csv(internal_ticker + ".csv", index_col="Date")

    data_ = data_.loc[(data_.index > date_start) & (data_.index < date_end)]
    data_.index = pd.to_datetime(data_.index)

    data_2 = pd.read_csv(internal_ticker + ".csv", index_col="Date")
    data_2 = data_2.loc[(data_2.index > date_start2) & (data_2.index < date_end)]
    data_2.index = pd.to_datetime(data_2.index)
    # creating 6m smoothing growth column and 10yr average column
    # Calculate the smoothed average
    average = data_.iloc[:, 0].rolling(11).mean()

    # Calculate the annualized growth rate
    annualized_6m_smoothed_growth_rate = (data_.iloc[:, 0][11:] / average) ** (12 / 6) - 1

    # Multiply the result by 100 and store it in the _6m_smoothing_growth column
    data_['_6m_smoothing_growth'] = 100 * annualized_6m_smoothed_growth_rate
    data_2['mom_average'] = 1000 * data_2.iloc[:, 0].pct_change(periods=1)
    data_2['10 yr average'] = data_2['mom_average'].rolling(120).mean()
    data_.dropna(inplace=True)
    data_2.dropna(inplace=True)
    return data_[['_6m_smoothing_growth']], data_2[['10 yr average']]
def regression(X,y):
    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Create the model
    model = SVR(kernel='rbf')

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Calculate the mean squared error
    mse = mean_squared_error(y_test, y_pred)

    # Plot the predicted values versus the real values
    plt.plot(y_test, y_pred, 'o')
    plt.xlabel('Real Values')
    plt.ylabel('Predicted Values')

    print("mse : ",mse)
    plt.show()
def fred_data(ticker):
    date_start = "2017-01-01"
    date_end = "2022-05-27"
    frequency = 'monthly'

    fred = Fred(api_key='f40c3edb57e906557fcac819c8ab6478')
    # get data as an array and transforming it into a dataframe
    return fred.get_series(ticker, observation_start=date_start, observation_end=date_end, freq=frequency)


if __name__=="__main__":
    #wheat,wheat10 = commo_smooth_data("cooper_prices",date_start,date_start2,date_end)
    #wheat.plot()
    #wheat10.plot()
    #plt.show()
    import pandas as pd



    cli = "MEI_CLI_10012023142556508.csv"
    nfci = "nfci-data-series-csv.csv"
    PATH_DATA = r"/Users/talbi/Downloads/"

    growth_csv = (pd.read_csv(PATH_DATA + cli, index_col="TIME")[['Value']])
    growth_csv.index = pd.to_datetime(growth_csv.index)

    nfci_csv = pd.read_csv(PATH_DATA + nfci,index_col="Friday_of_Week")['NFCI']
    nfci_csv.index = pd.to_datetime(nfci_csv.index)
    # National Financial Index (NFCI) :  https://www.chicagofed.org/research/data/nfci/current-data
    nfci_csv =nfci_csv.loc[~nfci_csv.index.duplicated(keep='first')]

    growth_csv.loc[~growth_csv.index.duplicated(keep='first')]

    file2 = (pd.read_excel(PATH_DATA + "DATA_REPLICATION.xlsx",'Macro',index_col=0))['Growth'].iloc[::-1]

    growth_resample =growth_csv.resample("1M").mean()
    growth_resample.columns=["Growth"]
    nfci_resample = nfci_csv.resample("1M").mean()

    # libor_rates = http://iborate.com/usd-libor/
    libor_us = pd.read_csv(PATH_DATA+"LIBOR USD.csv",index_col="Date")
    libor_us.index = pd.to_datetime(libor_us.index)
    libor_us_resample = libor_us.resample("1M").mean()
    libor_us_resample.columns = ["1M_LIBOR_US"]

    resample_ = pd.concat([growth_resample, nfci_resample,libor_us_resample], axis=1)
    resample_.dropna(inplace=True)

    #resample_['financial_stress']=(turb_index+ resample_['NFCI'])/2

    MSCI_WORLD = yf.download("SWDA.MI", start="1971-01-01", end=date_end, interval="1mo")[['Close']]
    barc_us_treasury = yf.download("USTY.DE", start="1971-01-01", end=date_end, interval="1mo")[['Close']]
    _10y_nominal_us = yf.download("^TNX", start="1971-01-01", end=date_end, interval="1mo")[['Close']]
    US_IG_bonds = yf.download("FBNDX", start="1971-01-01", end=date_end, interval="1mo")[['Close']]
    #iShares $ TIPS 0-5 UCITS ETF USD (Dist) for inflation linked bonds
    ilb = yf.download("TIP5.L", start="1971-01-01", end=date_end, interval="1mo")[['Close']]

    #American Funds Inflation Linked Bond Fund
    ilb_ = yf.download("BFIGX", start="1971-01-01", end=date_end, interval="1mo")[['Close']]

    #gold
    gold = yf.download("GC=F", start="1971-01-01", end=date_end, interval="1mo")[['Close']]

    #SP GSCI industrial metal index
    yf.download("^SPGSCI", start="1971-01-01", end=date_end, interval="1mo")[['Close']]
    print("H")
    #fig_ =

#acheter des bonds et augmenter les taux ? problème de liquidité sur la dette marocaine
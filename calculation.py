from fredapi import Fred
import pandas as pd
import matplotlib.pyplot as plt
import wget
import os
#need openpyxl

# Parameters for the request from the FRED Website

date_start = "2017-01-01"
date_end = "2022-04-10"

frequency = 'monthly'


def smoothed_2(ticker, date_start, date_end, frequency):
    fred = Fred(api_key='f40c3edb57e906557fcac819c8ab6478')

    # get data as an array and transforming it into a dataframe
    data_ = pd.DataFrame(
        fred.get_series("CPIAUCSL", observation_start=date_start, observation_end=date_end, freq=frequency))
    date_start2 = "2009-01-01"
    data_2 = pd.DataFrame(
        fred.get_series("CPIAUCSL", observation_start=date_start2, observation_end=date_end, freq=frequency))
    print(data_2)
    # creating 6m smoothing growth column
    data_['_6m_smoothing_growth'] = 100 * ((data_.iloc[:, 0][11:] / data_.iloc[:, 0].rolling(12).mean() - 1) * 2)[
                                          len(data_) - 19:]
    data_2['10 yr average'] = 100 * (data_.iloc[:, 0].rolling(10).mean().pct_change())
    print(data_2['10 yr average'])

    print(data_2)

    fig, ax = plt.subplots(1, figsize=(11, 6))
    # remove spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    # plot characteristics
    # plt.figure(figsize=(11,7))

    # drop the blank values
    data_.dropna(inplace=True)

    # ploting the data
    plt.plot(data_._6m_smoothing_growth, label='6m smoothing growth', marker='o', color="black")
    plt.plot(data_2['10 yr average'][len(data_2) - 19:], label='10 yr average', marker="o", color="green")
    for x, y in zip(data_.index[len(data_) - 19:len(data_):3],
                    data_._6m_smoothing_growth[len(data_) - 19:len(data_):3]):
        label = "{:.1f}".format(y) + " %"
        plt.annotate(label, (x, y), textcoords="offset points", xytext=(0, 8), ha='center', color="white", size=12,
                     bbox=dict(fc="black", ec="w", lw=0))
        # ec for the color du contour et lw is the width of les contours
    for x2, y2 in zip(data_2.index[len(data_2) - 19:len(data_2):3],
                      data_2['10 yr average'][len(data_2) - 19:len(data_2):3]):
        label = "{:.1f}".format(y2) + " %"
        plt.annotate(label, (x2, y2), textcoords="offset points", xytext=(0, 8), ha='center', color="white", size=12,
                     bbox=dict(fc="green", ec="w", lw=0))
    plt.legend()
    plt.title(ticker, fontsize=20, loc="center", style="italic")
    plt.xlabel('Date')
    plt.ylabel("growth rate (%)")

    return plt.show()
    # saving the figures in the static file



def fred_data(ticker):
    date_start = "2017-01-01"
    date_end = "2022-05-27"
    frequency = 'monthly'

    fred = Fred(api_key='f40c3edb57e906557fcac819c8ab6478')
    # get data as an array and transforming it into a dataframe
    return fred.get_series(ticker, observation_start=date_start, observation_end=date_end, freq=frequency)


if __name__=="__main__":
    data_fred = fred_data("INDPRO")
    cwd = os.getcwd()
    wget.download("http://atlantafed.org/-/media/documents/datafiles/chcs/wage-growth-tracker/wage-growth-data.xlsx")
    data_ = (pd.read_excel(cwd+"/wage-growth-data.xlsx")).iloc[:,11]
    data_['_6m_smoothing_growth'] = 100 * ((data_.iloc[3:, 0][11:] / data_.iloc[3:, 0].rolling(12).mean() - 1) * 2)[
                                          len(data_) - 19:]
    print(data_)
    #fig_ = smoothed_2("INDPRO",date_start,date_end,frequency)

#acheter des bonds et augmenter les taux ? problème de liquidité sur la dette marocaine
    print(data_fred)
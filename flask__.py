from flask import render_template, flash, redirect, url_for, request
from app import app
# from app.forms import LoginForm
import matplotlib.pyplot as plt
import numpy as np
import yfinance as yf
# from fredapi import Fred
from fredapi import Fred
import matplotlib.pyplot as plt
import pandas as pd
# from scipy.stats import norm
from flask_sqlalchemy import SQLAlchemy
import faulthandler
import json
import requests
# import fred_
from math import *

faulthandler.enable()

# api key for lemon markets data
api_key = "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJhdWQiOiJsZW1vbi5tYXJrZXRzIiwiaXNzIjoibGVtb24ubWFya2V0cyIsInN1YiI6InVzcl9xeUhYVkNDNjZUV3gxM3pUcFo1V3NCUVFUcTZINTBSQzQwIiwiZXhwIjoxNjg0NDI4NzY4LCJpYXQiOjE2NTI4OTI3NjgsImp0aSI6ImFwa19xeUhYVkNDNzdmRHpMakdqS0pqSzgzMGtEVnhrWFAwNkdLIiwibW9kZSI6InBhcGVyIn0.k4B1O-DyYePps29Jd4Mnoa7ERMfLA7WL8_m1fNCBqo4"


def phi(x):
    # 'Cumulative distribution function for the standard normal distribution'
    return (1.0 + erf(x / sqrt(2.0))) / 2.0


def gamma_expo(ticker):
    from datetime import datetime, timedelta, date

    pd.options.display.float_format = '{:,.4f}'.format

    # Inputs and Parameters
    filename = '/Users/talbi/Downloads/spx_quotedata.csv'

    # Black-Scholes European-Options Gamma
    def calcGammaEx(S, K, vol, T, r, q, optType, OI):
        if T == 0 or vol == 0:
            return 0

        dp = (np.log(S / K) + (r - q + 0.5 * vol ** 2) * T) / (vol * np.sqrt(T))
        dm = dp - vol * np.sqrt(T)

        if optType == 'call':
            gamma = np.exp(-q * T) * phi(dp) / (S * vol * np.sqrt(T))
            return OI * 100 * S * S * 0.01 * gamma
        else:  # Gamma is same for calls and puts. This is just to cross-check
            gamma = K * np.exp(-r * T) * phi(dm) / (S * S * vol * np.sqrt(T))
            return OI * 100 * S * S * 0.01 * gamma

    def isThirdFriday(d):
        return d.weekday() == 4 and 15 <= d.day <= 21

    # This assumes the CBOE file format hasn't been edited, i.e. table beginds at line 4
    optionsFile = open(filename)
    optionsFileData = optionsFile.readlines()
    optionsFile.close()

    # Get SPX Spot
    spotLine = optionsFileData[1]
    spotPrice = float(spotLine.split('Last:')[1].split(',')[0])
    fromStrike = 0.8 * spotPrice
    toStrike = 1.2 * spotPrice

    # Get Today's Date
    dateLine = optionsFileData[2]
    todayDate = dateLine.split('Date: ')[1].split(',')
    monthDay = todayDate[0].split(' ')

    # Handling of US/EU date formats
    if len(monthDay) == 2:
        year = int(todayDate[1])
        month = monthDay[0]
        day = int(monthDay[1])
    else:
        year = int(monthDay[2])
        month = monthDay[1]
        day = int(monthDay[0])

    todayDate = datetime.strptime("May", '%B')
    todayDate = todayDate.replace(day=day, year=year)

    # Get SPX Options Data
    df = pd.read_csv(filename, sep=",", header=None, skiprows=4)
    df.columns = ['ExpirationDate', 'Calls', 'CallLastSale', 'CallNet', 'CallBid', 'CallAsk', 'CallVol',
                  'CallIV', 'CallDelta', 'CallGamma', 'CallOpenInt', 'StrikePrice', 'Puts', 'PutLastSale',
                  'PutNet', 'PutBid', 'PutAsk', 'PutVol', 'PutIV', 'PutDelta', 'PutGamma', 'PutOpenInt']

    df['ExpirationDate'] = pd.to_datetime(df['ExpirationDate'], format='%a %b %d %Y')
    df['ExpirationDate'] = df['ExpirationDate'] + timedelta(hours=16)
    df['StrikePrice'] = df['StrikePrice'].astype(float)
    df['CallIV'] = df['CallIV'].astype(float)
    df['PutIV'] = df['PutIV'].astype(float)
    df['CallGamma'] = df['CallGamma'].astype(float)
    df['PutGamma'] = df['PutGamma'].astype(float)
    df['CallOpenInt'] = df['CallOpenInt'].astype(float)
    df['PutOpenInt'] = df['PutOpenInt'].astype(float)

    # ---=== CALCULATE SPOT GAMMA ===---
    # Gamma Exposure = Unit Gamma * Open Interest * Contract Size * Spot Price
    # To further convert into 'per 1% move' quantity, multiply by 1% of spotPrice
    df['CallGEX'] = df['CallGamma'] * df['CallOpenInt'] * 100 * spotPrice * spotPrice * 0.01
    df['PutGEX'] = df['PutGamma'] * df['PutOpenInt'] * 100 * spotPrice * spotPrice * 0.01 * -1

    df['TotalGamma'] = (df.CallGEX + df.PutGEX) / 10 ** 9
    dfAgg = df.groupby(['StrikePrice']).sum()
    strikes = dfAgg.index.values

    # Chart 1: Absolute Gamma Exposure
    plt.grid()
    plt.bar(strikes, dfAgg['TotalGamma'].to_numpy(), width=6, linewidth=0.1, edgecolor='k', label="Gamma Exposure")
    plt.xlim([fromStrike, toStrike])
    chartTitle = "Total Gamma: $" + str("{:.2f}".format(df['TotalGamma'].sum())) + " Bn per 1% SPX Move"
    plt.title(chartTitle, fontweight="bold", fontsize=20)
    plt.xlabel('Strike', fontweight="bold")
    plt.ylabel('Spot Gamma Exposure ($ billions/1% move)', fontweight="bold")
    plt.axvline(x=spotPrice, color='r', lw=1, label="SPX Spot: " + str("{:,.0f}".format(spotPrice)))
    plt.legend()
    plt.savefig('/Users/talbi/Downloads/microblog/app/static/' + ticker + "1.png")

    # Chart 2: Absolute Gamma Exposure by Calls and Puts
    plt.grid()
    plt.bar(strikes, dfAgg['CallGEX'].to_numpy() / 10 ** 9, width=6, linewidth=0.1, edgecolor='k', label="Call Gamma")
    plt.bar(strikes, dfAgg['PutGEX'].to_numpy() / 10 ** 9, width=6, linewidth=0.1, edgecolor='k', label="Put Gamma")
    plt.xlim([fromStrike, toStrike])
    chartTitle = "Total Gamma: $" + str("{:.2f}".format(df['TotalGamma'].sum())) + " Bn per 1% SPX Move"
    plt.title(chartTitle, fontweight="bold", fontsize=20)
    plt.xlabel('Strike', fontweight="bold")
    plt.ylabel('Spot Gamma Exposure ($ billions/1% move)', fontweight="bold")
    plt.axvline(x=spotPrice, color='r', lw=1, label="SPX Spot:" + str("{:,.0f}".format(spotPrice)))
    plt.legend()
    plt.savefig('/Users/talbi/Downloads/microblog/app/static/' + ticker + "2.png")

    # ---=== CALCULATE GAMMA PROFILE ===---
    levels = np.linspace(fromStrike, toStrike, 60)

    # For 0DTE options, I'm setting DTE = 1 day, otherwise they get excluded
    df['daysTillExp'] = [1 / 262 if (np.busday_count(todayDate.date(), x.date())) == 0 \
                             else np.busday_count(todayDate.date(), x.date()) / 262 for x in df.ExpirationDate]

    nextExpiry = df['ExpirationDate'].min()

    df['IsThirdFriday'] = [isThirdFriday(x) for x in df.ExpirationDate]
    thirdFridays = df.loc[df['IsThirdFriday'] == True]
    nextMonthlyExp = thirdFridays['ExpirationDate'].min()

    totalGamma = []
    totalGammaExNext = []
    totalGammaExFri = []

    # For each spot level, calc gamma exposure at that point
    for level in levels:
        df['callGammaEx'] = df.apply(lambda row: calcGammaEx(level, row['StrikePrice'], row['CallIV'],
                                                             row['daysTillExp'], 0, 0, "call", row['CallOpenInt']),
                                     axis=1)

        df['putGammaEx'] = df.apply(lambda row: calcGammaEx(level, row['StrikePrice'], row['PutIV'],
                                                            row['daysTillExp'], 0, 0, "put", row['PutOpenInt']), axis=1)

        totalGamma.append(df['callGammaEx'].sum() - df['putGammaEx'].sum())

        exNxt = df.loc[df['ExpirationDate'] != nextExpiry]
        totalGammaExNext.append(exNxt['callGammaEx'].sum() - exNxt['putGammaEx'].sum())

        exFri = df.loc[df['ExpirationDate'] != nextMonthlyExp]
        totalGammaExFri.append(exFri['callGammaEx'].sum() - exFri['putGammaEx'].sum())

    totalGamma = np.array(totalGamma) / 10 ** 9
    totalGammaExNext = np.array(totalGammaExNext) / 10 ** 9
    totalGammaExFri = np.array(totalGammaExFri) / 10 ** 9

    # Find Gamma Flip Point
    zeroCrossIdx = np.where(np.diff(np.sign(totalGamma)))[0]

    negGamma = totalGamma[zeroCrossIdx]
    posGamma = totalGamma[zeroCrossIdx + 1]
    negStrike = levels[zeroCrossIdx]
    posStrike = levels[zeroCrossIdx + 1]

    # Writing and sharing this code is only possible with your support!
    # If you find it useful, consider supporting us at perfiliev.com/support :)
    zeroGamma = posStrike - ((posStrike - negStrike) * posGamma / (posGamma - negGamma))
    zeroGamma = zeroGamma[0]

    # Chart 3: Gamma Exposure Profile
    fig, ax = plt.subplots()
    plt.grid()
    plt.plot(levels, totalGamma, label="All Expiries")
    plt.plot(levels, totalGammaExNext, label="Ex-Next Expiry")
    plt.plot(levels, totalGammaExFri, label="Ex-Next Monthly Expiry")
    chartTitle = "Gamma Exposure Profile, SPX, " + todayDate.strftime('%d %b %Y')
    plt.title(chartTitle, fontweight="bold", fontsize=20)
    plt.xlabel('Index Price', fontweight="bold")
    plt.ylabel('Gamma Exposure ($ billions/1% move)', fontweight="bold")
    plt.axvline(x=spotPrice, color='r', lw=1, label="SPX Spot: " + str("{:,.0f}".format(spotPrice)))
    plt.axvline(x=zeroGamma, color='g', lw=1, label="Gamma Flip: " + str("{:,.0f}".format(zeroGamma)))
    plt.axhline(y=0, color='grey', lw=1)
    plt.xlim([fromStrike, toStrike])
    trans = ax.get_xaxis_transform()
    plt.fill_between([fromStrike, zeroGamma], min(totalGamma), max(totalGamma), facecolor='red', alpha=0.1,
                     transform=trans)
    plt.fill_between([zeroGamma, toStrike], min(totalGamma), max(totalGamma), facecolor='green', alpha=0.1,
                     transform=trans)
    plt.legend()
    plt.savefig('/Users/talbi/Downloads/microblog/app/static/' + ticker + "3.png")


# Parameters for the request from the FRED Website

date_start = "2017-01-01"
date_end = "2022-04-10"
date_start2 = "2009-01-01"
frequency = 'monthly'


def smoothed_2(ticker, date_start, date_end, frequency):
    fred = Fred(api_key='f40c3edb57e906557fcac819c8ab6478')

    # get data as an array and transforming it into a dataframe
    data_ = pd.DataFrame(
        fred.get_series(ticker, observation_start=date_start, observation_end=date_end, freq=frequency))
    data_2 = pd.DataFrame(
        fred.get_series(ticker, observation_start=date_start2, observation_end=date_end, freq=frequency))

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

    # saving the figures in the static file
    plt.savefig('/Users/talbi/Downloads/microblog/app/static/' + ticker + ".png")


def fred_data(ticker):
    date_start = "2017-01-01"
    date_end = "2022-05-27"
    frequency = 'monthly'

    fred = Fred(api_key='f40c3edb57e906557fcac819c8ab6478')
    # get data as an array and transforming it into a dataframe
    return fred.get_series(ticker, observation_start=date_start, observation_end=date_end, freq=frequency)


app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:////tmp/test.db'
db = SQLAlchemy(app)


class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)

    def __repr__(self):
        return '<User %r>' % self.email


db.create_all()


@app.route("/")
def ap():
    # ticker = request.form['ticker']
    return render_template("index.html")


@app.route("/index", methods=['POST'])
def index():
    # upd_file = request.files['file']
    # if upd_file.filename != '':
    #	upd_file.save(upd_file.filename)
    ticker = request.form['ticker']
    c = smoothed_2(ticker, date_start, date_end, frequency)
    # plt.plot(c)
    # plt.savefig('/Users/talbi/Downloads/microblog/app/static/'+ticker+'.png')
    url = url_for('static', filename=str(ticker) + '.png')
    return render_template('index.html', ticker=ticker, url=url, fred_request=True)


@app.route("/page2")
def next():
    return render_template("page2.html")


@app.route("/gamma", methods=["POST"])
def gamma_exposure():
    ticker_cboe = request.form["ticker_cboe"]
    url1 = url_for('static', filename=str(ticker_cboe) + '1.png')
    url2 = url_for('static', filename=str(ticker_cboe) + '2.png')
    url3 = url_for('static', filename=str(ticker_cboe) + '3.png')
    return render_template("page2.html", ticker_cboe=ticker_cboe, delayed_quote=True, url1=url1, url2=url2, url3=url3)


@app.route('/option_calculator', methods=['POST'])
def option_calculator():
    S = float(request.form["S"])
    r = float(request.form["r"])
    K = float(request.form["K"])
    sigma = float(request.form["sigma"])
    T = float(request.form["T"])

    d1 = (np.log(S / K) + (r + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    # N = norm.cdf
    price = S * phi(d1) - K * exp(-r * T) * phi(d2)
    # price=2*10
    return render_template("page2.html", S=S, r=r, K=K, sigma=sigma, T=T, price=price, calculation_success=True)


@app.route('/fair_price', methods=['POST'])
def fair_price():
    bid = float(request.form['bid'])
    ask = float(request.form['ask'])
    bid_volume = float(request.form['bid_v'])
    ask_volume = float(request.form['ask_v'])
    fair_price = (bid * ask_volume + ask * bid_volume) / (2 * (ask_volume + bid_volume))
    return render_template("page2.html", bid=bid, ask=ask, bid_v=bid_volume, ask_v=ask_volume, fair_price=fair_price,
                           calculation_success2=True)


"""
@app.route('/login')
def login():
	form = LoginForm()
	return render_template('login.html',title='Sign In', form=form)
"""


@app.route("/graph")
def graph():
    url = url_for('static', filename="a.png")
    return render_template("graph.html", graph="Production", url=url)


@app.route("/graph1")
def graph1():
    url = url_for('static', filename="NonFarm Payroll.png")
    return render_template("graph.html", graph="Income", url=url)


@app.route("/graph2")
def graph2():
    url = url_for('static', filename="Real Personal Consumption Expenditures.png")
    return render_template("graph.html", graph="Consumption", url=url)


@app.route("/quote")
def display_quote():
    date_start = "2017-01-01"
    date_end = "2022-04-10"
    frequency = 'monthly'
    fred = Fred(api_key='f40c3edb57e906557fcac819c8ab6478')

    # get data as an array and transforming it into a dataframe
    data_ = pd.DataFrame(
        fred.get_series("INDPRO", observation_start=date_start, observation_end=date_end, freq=frequency))
    data_.index = np.arange(0, len(data_), 1)

    return pd.Series.to_json(data_)


@app.route("/request_quote", methods=["POST"])
def request_quote():
    ISIN = request.form['ISIN']
    request_ = requests.get("https://data.lemon.markets/v1/quotes/?isin=" + str(ISIN),
                            headers={"Authorization": "Bearer " + str(api_key)})
    data = request_.json()
    bid_volume = float(data['results'][0]['b_v'])
    ask_volume = float(data['results'][0]['a_v'])
    ask = float(data['results'][0]['a'])
    bid = float(data['results'][0]['b'])

    fair_price = (bid * ask_volume + ask * bid_volume) / ((ask_volume + bid_volume))
    return render_template('page2.html', bid=bid, ask=ask, bid_v=bid_volume, ask_v=ask_volume, fair_price=fair_price,
                           ISIN=ISIN, quote_request_success=True)


# return render_template("page2.html",ISIN=ISIN)

@app.route("/register", methods=["GET", "POST"])
def register_():
    if request.method == "POST":
        req = request.form
        print(req)
        # username = req["username"]
        # id = req["id"]
        # email = req["email"]

        # user_ = User(id=id, username = username, email = email)

        # add the user_ object to the database
        # db.session.add(user_)

        # commit changes
        # db.session.commit()

        return "Regisration Successful"
    return render_template('register.html')


if __name__ == '__main__':
    app.debug(True)
    app.run()
"""@app.route('/login', methods=['GET', 'POST'])
def login():
	form = LoginForm()
	if form.validate_on_submit():
		flash('Login requested for user {}, remember_me={}'.format(
			form.username.data, form.remember_me.data))
		return redirect('/')
	return render_template('login.html', title='Sign In', form=form)

"""


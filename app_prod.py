import dash
from dash import Dash, dcc, html, Input, Output
import dash_table
import plotly.graph_objects as go
import plotly.express as px
from fredapi import Fred
import pandas as pd
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import dash_bootstrap_components as dbc
from datetime import timedelta
import socket
import os
import datetime
import numpy as np
import sklearn
import yfinance as yf
from sklearn.model_selection import train_test_split


def smooth_data(ticker, date_start, date_start2, date_end):
    frequency = "monthly"
    fred = Fred(api_key='f40c3edb57e906557fcac819c8ab6478')
    data_ = pd.DataFrame(
        fred.get_series(ticker, observation_start=date_start, observation_end=date_end, freq=frequency))
    data_2 = pd.DataFrame(
        fred.get_series(ticker, observation_start=date_start2, observation_end=date_end, freq=frequency))

    # creating 6m smoothing growth column and 10yr average column
    # Calculate the smoothed average
    average =  data_.iloc[:, 0].rolling(11).mean()

    # Calculate the annualized growth rate
    annualized_6m_smoothed_growth_rate = (data_.iloc[:, 0][11:] / average) ** (365 / 180) - 1

    # Multiply the result by 100 and store it in the _6m_smoothing_growth column
    data_['_6m_smoothing_growth'] = 100 * annualized_6m_smoothed_growth_rate
    data_2['mom_average'] = 100 * (data_2).iloc[:, 0].pct_change(periods=1)
    data_2['10 yr average'] = data_2['mom_average'].rolling(120).mean()
    data_.dropna(inplace=True)
    data_2.dropna(inplace=True)
    return data_[['_6m_smoothing_growth']], data_2[['10 yr average']]

#dol down => bearish => etre long bond
def score_table(index, data_, data_10):
    score_table = pd.DataFrame.from_dict({"trend vs history ": 1 if data_.iloc[-1, 0] > data_10.iloc[-1, 0] else 0,
                                          "growth": 1 if data_.iloc[-1, 0] > 0 else 0,
                                          "Direction of Trend": 1 if (data_.resample("3M").last().diff())[-2:].sum()[
                                                                         0] > 1 else 0}, orient="index").T
    score_table['Score'] = score_table.sum(axis=1)
    score_table['Indicators'] = index

    return score_table


def fred_data(ticker):
    date_start = "2017-01-01"
    date_end = "2022-05-27"
    frequency = 'monthly'

    fred = Fred(api_key='f40c3edb57e906557fcac819c8ab6478')
    # get data as an array and transforming it into a dataframe
    return fred.get_series(ticker, observation_start=date_start, observation_end=date_end, freq=frequency)


# (new factor) : variation of real rate
# Parameters for the request from the FRED Website

date_start_ = "2017-01-01"
date_end_ = "2022-12-10"

frequency_ = 'monthly'

external_stylesheets = [
    {
        "href": "https://codepen.io/chriddyp/pen/bWLwgP.css"
                "family=Lato:wght@400;700&display=swap",
        "rel": "stylesheet",
    }, ]

app = Dash(__name__, external_stylesheets=external_stylesheets)

app.config.suppress_callback_exceptions = True

timeout = 20

app.layout = html.Div(style={'backgroundColor': "rgb(17, 17, 17)"}, children=[
    # dcc.Interval(id="interval_comp",interval=15*1000),
    dbc.Tabs(
        [
            dbc.Tab(label="Volatility Indicators", tab_id="Volatility Indicators"),
            dbc.Tab(label="Macroeconomic Indicators", tab_id="Macroeconomic Indicators"),
            dbc.Tab(label="Directional Indicators", tab_id="Directional Indicators"),
            dbc.Tab(label="Trend-Momentum Indicators", tab_id="Trend-Momentum Indicators"),

        ],
        id="tabs",
        active_tab="Macroeconomic Indicators",
    ),

    html.H1("Economic Framework (Talbi & Co)", style={"margin-left": "550px", "color": "white"}),
    html.Br(),
    html.Div([dcc.Input("INDPRO", placeholder="ticker", id="ticker_fred"),
              dcc.Input(date_start_, placeholder="start date", id="date_start"),
              dcc.Input(date_end_, placeholder="end date", id="date_end")], style={'margin-left': '650px'}),
    html.Div(id="graph_indicator", style={'margin-left': '100px'}),

    # html.Div(,style={"margin-top":"10px"}),

    html.Div(
        dcc.Dropdown(
            id='dropdown',
            options=[
                {'label': 'Single Indicator search', 'value': 'fred_search'},
                {'label': 'brainard_test', 'value': 'brainard_test'},
                {'label': 'Growth', 'value': 'Growth'},
                {'label': 'Inflation Outlook', 'value': 'Inflation Outlook'},
            ],
            value='Growth',
            style={
                "width": "180px",
                "font-size": "16px",
                "padding": "10px",
                "border-radius": "5px",
                "border": "1px solid white",
                "background-color": "rgb(17,17,17)",
                "color": "#000000"
            }
        ),
        style={"width": "150px", "margin-left": "120px"}
    ),
    html.Div(id="trends_graphs", style={"margin-left": "100px"})
])


@app.callback(Output("trends_graphs", "children"),
              # Input("interval_comp","n_intervals"),
              [Input("ticker_fred", "value"),
               Input("dropdown", "value"),
               Input("tabs", "active_tab"),
               Input("date_start", "value"),
               Input("date_end", "value"),

               ]
              )
def trends(ticker_fred, dropdown, tabs, date_start, date_end):
    fred = Fred(api_key='f40c3edb57e906557fcac819c8ab6478')
    date_start2 = "2004-01-01"
    print("1")
    # try:
    PATH_DATA = "/Users/talbi/Downloads/"

    _30y = yf.download("^TYX", start=date_start, end=date_end, interval="1d")[['Close']]

    _5y_nominal = yf.download("^FVX", start=date_start, end=date_end, interval="1d")[['Close']]

    cooper = yf.download("HG=F", start=date_start, end=date_end, interval="1d")[['Close']]
    cooper_prices = cooper * 100
    _5y_nominal_var = _5y_nominal.diff()

    _5y_real = pd.DataFrame(
        fred.get_series("DFII5", observation_start=date_start, observation_end=date_end, freq="daily"))
    _5y_real.columns = ['Close']

    spread = _30y - _5y_real

    merged_data = pd.concat([spread, _5y_nominal, cooper_prices], axis=1)
    merged_data.dropna(inplace=True)
    merged_data.columns = ["spread 30_5yr", "5y", "cooper", ]

    # Data importing
    print("2")
    cpi, cpi_10 = smooth_data("CPIAUCSL", date_start, date_start2, date_end)
    single_search, single_search_10 = smooth_data(ticker_fred, date_start, date_start2, date_end)
    pcec96, pcec96_10 = smooth_data("PCEC96", date_start, date_start2, date_end)
    # pcec96 = pcec96[["_6m_smoothing_growth"]]-cpi[["_6m_smoothing_growth"]]
    # pcec96_10 = pcec96_10[['10 yr average']]-cpi_10[['10 yr average']]
    print("3")
    indpro, indpro_10 = smooth_data("INDPRO", date_start, date_start2, date_end)
    print('4')
    nonfarm, nonfarm_10 = smooth_data("PAYEMS", date_start, date_start2, date_end)
    print("5")
    real_pers, real_pers_10 = smooth_data("W875RX1", date_start, date_start2, date_end)
    # real_pers = real_pers[["_6m_smoothing_growth"]] - cpi[["_6m_smoothing_growth"]]
    # real_pers_10 = real_pers_10[['10 yr average']] - cpi_10[['10 yr average']]

    retail_sales, retail_sales_10 = smooth_data("RRSFS", date_start, date_start2, date_end)
    # retail_sales = retail_sales[["_6m_smoothing_growth"]] - cpi[["_6m_smoothing_growth"]]
    # retail_sales_10 = retail_sales_10[['10 yr average']] - cpi_10[['10 yr average']]

    employment_level, employment_level_10 = smooth_data("CE16OV", date_start, date_start2, date_end)
    employment_level.dropna(inplace=True)
    shelter_prices, shelter_prices_10 = smooth_data("CUSR0000SAH1", date_start, date_start2, date_end)
    shelter_prices = shelter_prices[["_6m_smoothing_growth"]] - cpi[["_6m_smoothing_growth"]]
    shelter_prices_10 = shelter_prices_10[['10 yr average']] - cpi_10[['10 yr average']]

    cwd = os.getcwd()
    # wget.download("http://atlantafed.org/-/media/documents/datafiles/chcs/wage-growth-tracker/wage-growth-data.xlsx")
    wage_tracker = pd.DataFrame(pd.read_excel(cwd + "/wage-growth-data.xlsx").iloc[3:, [0, 11]])
    wage_tracker.columns = ['date', "wage_tracker"]
    wage_tracker.set_index('date', inplace=True)

    wheat_ = yf.download("ZW=F", start=date_start, end=datetime.datetime.now(), interval="1d")[['Close']]
    oil_ = yf.download("CL=F", start=date_start, end=date_end, interval="1d")[['Close']]
    gas_ = yf.download("NG=F", start=date_start, end=date_end, interval="1d")[['Close']]

    employment_level_wage_tracker = pd.concat([employment_level, wage_tracker], axis=1)
    employment_level_wage_tracker.dropna(inplace=True)
    wages, wages_10 = smooth_data("CES0500000003", date_start, date_start2, date_end)
    wages = wages[["_6m_smoothing_growth"]] - cpi[["_6m_smoothing_growth"]]
    wages_10 = wages_10[['10 yr average']] - cpi_10[['10 yr average']]

    core_cpi, core_cpi_10 = smooth_data("CPILFESL", date_start, date_start2, date_end)
    core_pce, core_pce_10 = smooth_data("DPCCRC1M027SBEA", date_start, date_start2, date_end)

    composite_data = pd.concat(
        [pcec96[['_6m_smoothing_growth']], indpro[['_6m_smoothing_growth']], nonfarm[['_6m_smoothing_growth']],
         real_pers[['_6m_smoothing_growth']], retail_sales[['_6m_smoothing_growth']],
         employment_level[['_6m_smoothing_growth']]], axis=1)
    composite_data.dropna(inplace=True)
    composite_growth = pd.DataFrame(composite_data.mean(axis=1))
    composite_growth.columns = ["_6m_smoothing_growth"]
    composite_growth_10 = pd.concat(
        [pcec96_10[['10 yr average']], indpro_10[['10 yr average']], nonfarm_10[['10 yr average']],
         real_pers_10[['10 yr average']], retail_sales_10[['10 yr average']], employment_level_10[['10 yr average']]],
        axis=1)
    composite_growth_10.dropna(inplace=True)
    composite_growth_10 = pd.DataFrame(composite_growth_10.mean(axis=1))
    composite_growth_10.columns = ["10 yr average"]

    spread_norm = (spread - np.mean(spread)) / np.std(spread)

    # cooper_norm = (spread - np.mean(spread)) / np.std(spread)
    cooper_norm = (cooper - np.mean(cooper)) / np.std(cooper)

    merged_data_norm = pd.concat([spread_norm, cooper_norm], axis=1)
    merged_data_norm.columns = ["spread normalized", "cooper normalized"]
    merged_data_norm.dropna(inplace=True)
    # plt.plot(cooper.index.to_list(),cooper[['Close']])

    print("6")
    retail_sales_title = fred.get_series_info("MRTSSM44X72USS").title

    employment_level_title = fred.get_series_info("CE16OV").title

    pce_title = fred.get_series_info("PCEC96").title
    print("7")
    indpro_title = fred.get_series_info("INDPRO").title
    print('8')
    nonfarm_title = fred.get_series_info("PAYEMS").title
    print("9")
    real_personal_income_title = fred.get_series_info("W875RX1").title
    print("3")
    cpi_title = fred.get_series_info("CPIAUCSL").title
    core_cpi_title = fred.get_series_info("CPILFESL").title
    core_pce_title = fred.get_series_info("DPCCRC1M027SBEA").title

    shelter_title = fred.get_series_info("CUSR0000SAH1").title
    wages_title = fred.get_series_info("CES0500000003").title

    score_table_merged = pd.concat(
        [score_table("PCE", pcec96, pcec96_10), score_table("Industrial Production", indpro, indpro_10),
         score_table("NonFarm Payroll", nonfarm, nonfarm_10),
         score_table("Real Personal Income", real_pers, real_pers_10),
         score_table("Real Retail Sales", retail_sales, retail_sales_10),
         score_table("Employment Level", employment_level, employment_level_10),
         score_table("COMPOSITE GROWTH", composite_growth, composite_growth_10)], axis=0)
    score_table_merged = score_table_merged.iloc[:, [4, 0, 1, 2, 3]]

    score_table_merged_infla = pd.concat([score_table("CPI", cpi, cpi_10),
                                          score_table("Core CPI", core_cpi, core_cpi_10),
                                          score_table("PCE", pcec96, pcec96_10),
                                          score_table("Core PCE", core_pce, core_pce_10),
                                          score_table("Shelter Prices", shelter_prices, shelter_prices_10)], axis=0)

    score_table_merged_infla = score_table_merged_infla.iloc[:, [4, 0, 1, 2, 3]]
    # score_table_merged.set_index("index", inplace=True)

    print(dropdown)

    fig_cyclical_trends = make_subplots(rows=3, cols=2, subplot_titles=[pce_title, indpro_title
        , nonfarm_title, real_personal_income_title, retail_sales_title, employment_level_title])

    fig_cyclical_trends.add_trace(
        go.Scatter(x=pcec96.index.to_list(), y=pcec96._6m_smoothing_growth / 100, name="6m growth average",
                   mode="lines", line=dict(width=2, color='white')), row=1, col=1)
    fig_cyclical_trends.add_trace(go.Scatter(x=(pcec96_10.index.to_list()),
                                             y=(pcec96_10['10 yr average']) / 100, mode="lines",
                                             line=dict(width=2, color='green'),
                                             name="10yr average"), row=1, col=1)

    fig_cyclical_trends.add_trace(
        go.Scatter(x=indpro.index.to_list(), y=indpro._6m_smoothing_growth / 100, name="6m growth average",
                   mode="lines", line=dict(width=2, color='white'), showlegend=False), row=1, col=2)
    fig_cyclical_trends.add_trace(go.Scatter(x=(indpro_10.index.to_list()),
                                             y=indpro_10['10 yr average'] / 100, line=dict(width=2, color='green'),
                                             mode="lines",
                                             name="10yr average", showlegend=False), row=1, col=2)

    fig_cyclical_trends.add_trace(
        go.Scatter(x=nonfarm.index.to_list(), y=nonfarm._6m_smoothing_growth / 100, name="6m growth average",
                   mode="lines", line=dict(width=2, color='white'), showlegend=False), row=2, col=1)
    fig_cyclical_trends.add_trace(go.Scatter(x=(nonfarm_10.index.to_list()),
                                             y=nonfarm_10['10 yr average'] / 100, line=dict(width=2, color='green'),
                                             mode="lines",
                                             name="10yr average", showlegend=False), row=2, col=1)

    fig_cyclical_trends.add_trace(
        go.Scatter(x=real_pers.index.to_list(), y=real_pers._6m_smoothing_growth / 100, name="6m growth average",
                   mode="lines", line=dict(width=2, color='white'), showlegend=False), row=2, col=2)
    fig_cyclical_trends.add_trace(go.Scatter(x=(real_pers_10.index.to_list()),
                                             y=real_pers_10['10 yr average'] / 100, line=dict(width=2, color='green'),
                                             mode="lines",
                                             name="10yr average", showlegend=False), row=2, col=2)

    fig_cyclical_trends.add_trace(
        go.Scatter(x=retail_sales.index.to_list(), y=retail_sales._6m_smoothing_growth / 100, name="6m growth average",
                   mode="lines", line=dict(width=2, color='white'), showlegend=False), row=3, col=1)
    fig_cyclical_trends.add_trace(go.Scatter(x=(retail_sales_10.index.to_list()),
                                             y=retail_sales_10['10 yr average'] / 100,
                                             line=dict(width=2, color='green'), mode="lines",
                                             name="10yr average", showlegend=False), row=3, col=1)

    fig_cyclical_trends.add_trace(
        go.Scatter(x=employment_level.index.to_list(), y=employment_level._6m_smoothing_growth / 100,
                   name="6m growth average",
                   mode="lines", line=dict(width=2, color='white'), showlegend=False), row=3, col=2)
    fig_cyclical_trends.add_trace(go.Scatter(x=(employment_level_10.index.to_list()),
                                             y=employment_level_10['10 yr average'] / 100,
                                             line=dict(width=2, color='green'), mode="lines",
                                             name="10yr average", showlegend=False), row=3, col=2)

    fig_cyclical_trends.update_layout(template="plotly_dark",
                                      height=1000, width=1500)
    fig_cyclical_trends.update_layout(  # customize font and legend orientation & position
        yaxis=dict(tickformat=".0%"),
        title_font_family="Arial Black",
        font=dict(
            family="Rockwell",
            size=18),
        legend=dict(
            title=None, orientation="h", y=1.02, yanchor="bottom", x=0.5, xanchor="center"
        )
    )
    fig_cyclical_trends.layout.yaxis2.tickformat = ".1%"
    fig_cyclical_trends.layout.yaxis3.tickformat = ".1%"
    fig_cyclical_trends.layout.yaxis4.tickformat = ".1%"
    fig_cyclical_trends.layout.yaxis5.tickformat = ".1%"
    fig_cyclical_trends.layout.yaxis6.tickformat = ".1%"
    date_start = (datetime.datetime.strptime(date_start, "%Y-%m-%d") + timedelta(days=130)).strftime("%Y-%m-%d")
    fig_cyclical_trends.update_layout(xaxis_range=[date_start, date_end])

    fig_secular_trends = make_subplots(rows=3, cols=2, specs=[[{"secondary_y": True}, {"secondary_y": True}],
                                                              [{"secondary_y": True}, {"secondary_y": True}],
                                                              [{"secondary_y": True}, {"secondary_y": True}]],
                                       subplot_titles=["CPI", "Core CPI"
                                           , pce_title, core_pce_title, shelter_title,
                                                       "Employment (Growth) and Wage tracker levels"])

    fig_secular_trends.add_trace(
        go.Scatter(x=cpi.index.to_list(), y=cpi._6m_smoothing_growth / 100, name="6m growth average",
                   mode="lines", line=dict(width=2, color='white')), secondary_y=False, row=1, col=1)
    fig_secular_trends.add_trace(go.Scatter(x=(cpi_10.index.to_list()),
                                            y=(cpi_10['10 yr average']) / 100, mode="lines",
                                            line=dict(width=2, color='green'),
                                            name="10yr average"), secondary_y=False, row=1, col=1)

    fig_secular_trends.add_trace(
        go.Scatter(x=core_cpi.index.to_list(), y=core_cpi._6m_smoothing_growth / 100, name="6m growth average",
                   mode="lines", line=dict(width=2, color='white'), showlegend=False), secondary_y=False, row=1, col=2)
    fig_secular_trends.add_trace(go.Scatter(x=(core_cpi_10.index.to_list()),
                                            y=core_cpi_10['10 yr average'] / 100,
                                            line=dict(width=2, color='green'), mode="lines",
                                            name="10yr average", showlegend=False), secondary_y=False, row=1, col=2)

    fig_secular_trends.add_trace(
        go.Scatter(x=pcec96.index.to_list(), y=pcec96._6m_smoothing_growth / 100, name="6m growth average",
                   mode="lines", line=dict(width=2, color='white'), showlegend=False), secondary_y=False, row=2, col=1)
    fig_secular_trends.add_trace(go.Scatter(x=(pcec96_10.index.to_list()),
                                            y=pcec96_10['10 yr average'] / 100,
                                            line=dict(width=2, color='green'), mode="lines",
                                            name="10yr average", showlegend=False), secondary_y=False, row=2, col=1)

    fig_secular_trends.add_trace(
        go.Scatter(x=core_pce.index.to_list(), y=core_pce._6m_smoothing_growth / 100, name="6m growth average",
                   mode="lines", line=dict(width=2, color='white'), showlegend=False), row=2, col=2)
    fig_secular_trends.add_trace(go.Scatter(x=(core_pce_10.index.to_list()),
                                            y=core_pce_10['10 yr average'] / 100,
                                            line=dict(width=2, color='green'), mode="lines",
                                            name="10yr average", showlegend=False), secondary_y=False, row=2, col=2)
    fig_secular_trends.add_trace(
        go.Scatter(x=shelter_prices.index.to_list(), y=shelter_prices._6m_smoothing_growth / 100,
                   name="6m growth average",
                   mode="lines", line=dict(width=2, color='white'), showlegend=False), secondary_y=False, row=3, col=1)
    fig_secular_trends.add_trace(go.Scatter(x=(shelter_prices_10.index.to_list()),
                                            y=shelter_prices_10['10 yr average'] / 100,
                                            line=dict(width=2, color='green'), mode="lines",
                                            name="10yr average", showlegend=False), secondary_y=False, row=3, col=1)

    fig_secular_trends.add_trace(
        go.Scatter(x=employment_level_wage_tracker.index.to_list(),
                   y=employment_level_wage_tracker._6m_smoothing_growth, name="Employment level 6m annualized growth",
                   mode="lines", line=dict(width=2, color='white'), showlegend=True), secondary_y=False, row=3, col=2)
    fig_secular_trends.add_trace(
        go.Scatter(x=employment_level_wage_tracker.index.to_list(), y=employment_level_wage_tracker.wage_tracker,
                   name="Atlanta Fed wage tracker",
                   mode="lines", line=dict(width=2, color='blue'), showlegend=True), secondary_y=True, row=3, col=2)
    fig_secular_trends.update_layout(template="plotly_dark",
                                     height=1000, width=1500)
    fig_secular_trends.update_layout(  # customize font and legend orientation & position
        yaxis=dict(tickformat=".1%"),
        title_font_family="Arial Black",
        font=dict(
            family="Rockwell",
            size=15),
        legend=dict(
            title=None, orientation="h", y=1.02, yanchor="bottom", x=0.5, xanchor="center"
        )
    )

    fig_secular_trends.layout.yaxis2.tickformat = ".1%"
    fig_secular_trends.layout.yaxis3.tickformat = ".1%"
    fig_secular_trends.layout.yaxis4.tickformat = ".1%"
    fig_secular_trends.layout.yaxis5.tickformat = ".1%"

    fig_secular_trends_2 = make_subplots(rows=2, cols=2)

    fig_secular_trends_2.add_trace(go.Scatter(x=wheat_.index.to_list(), y=wheat_['Close'], name="Wheat prices",
                                              mode="lines", line=dict(width=2, color='white')), row=1, col=1)
    fig_secular_trends_2.add_trace(go.Scatter(x=wheat_.index.to_list(), y=cooper['Close'], name="Cooper prices",
                                              mode="lines", line=dict(width=2, color='orange')), row=1, col=2)
    fig_secular_trends_2.add_trace(go.Scatter(x=gas_.index.to_list(), y=gas_['Close'], name="Gas prices",
                                              mode="lines", line=dict(width=2, color='green')), row=2, col=1)
    fig_secular_trends_2.add_trace(go.Scatter(x=oil_.index.to_list(), y=oil_['Close'], name="Oil prices",
                                              mode="lines", line=dict(width=2, color='blue')), row=2, col=2)

    fig_secular_trends_2.update_layout(
        template="plotly_dark",
        title={
            'text': "Commodities prices",
            'y': 0.9,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'})

    fig_secular_trends_2.update_layout(  # customize font and legend orientation & position
        title_font_family="Arial Black",
        font=dict(
            family="Rockwell",
            size=16),
        legend=dict(
            title=None, orientation="h", y=0.97, yanchor="bottom", x=0.5, xanchor="center"
        )
    )
    fig_secular_trends_2.update_layout(height=650, width=1500)
    fig_ = go.Figure()

    # drop the blank values

    # ploting the data
    # composite_growth_10 = 100 * (composite_growth.iloc[:, 0].rolling(10).mean().pct_change())
    fig_.add_trace(go.Scatter(x=composite_growth.index.to_list(), y=composite_growth._6m_smoothing_growth / 100,
                              name="6m growth average",
                              mode="lines", line=dict(width=2, color='white')))
    fig_.add_trace(go.Scatter(x=composite_growth_10.index.to_list(),
                              y=composite_growth_10['10 yr average'] / 100,
                              name="6m growth average",
                              mode="lines", line=dict(width=2, color='green')))

    for x, y in zip(composite_growth.index[len(composite_growth) - 19:len(composite_growth):3],
                    composite_growth.iloc[len(composite_growth) - 19:len(composite_growth):3, 0] / 100):
        label = "{:.1f}".format(y * 100) + " %"
        fig_.add_annotation(x=x, y=y,
                            text=label,
                            showarrow=False, yshift=10)

    for x, y in zip(composite_growth_10.index[len(composite_growth_10) - 19:len(composite_growth_10):3],
                    composite_growth_10.iloc[len(composite_growth_10) - 19:len(composite_growth_10):3, 0] / 100):
        label = "{:.1f}".format(y * 100) + " %"
        fig_.add_annotation(x=x, y=y,
                            text=label,
                            showarrow=False, yshift=10)

    fig_brainard = make_subplots(cols=1, rows=2, specs=[[{"secondary_y": True}], [{"secondary_y": True}]],
                                 subplot_titles=["Levels", "Returns"])

    fig_brainard.add_trace(
        go.Scatter(x=merged_data.index.to_list(), y=merged_data.iloc[:, 0], name="Spread",
                   mode="lines", line=dict(width=2, color='white')), secondary_y=False, col=1, row=1)
    fig_brainard.add_trace(go.Scatter(x=merged_data.index.to_list(), y=merged_data.iloc[:, 1], name="US 5y",
                                      mode="lines", line=dict(width=2, color='purple')), secondary_y=False, col=1,
                           row=1)
    fig_brainard.add_trace(go.Scatter(x=merged_data.index.to_list(), y=merged_data.iloc[:, 2], name="Cooper prices",
                                      mode="lines", line=dict(width=2, color='orange')), secondary_y=True, col=1, row=1)
    fig_brainard.add_trace(
        go.Scatter(x=merged_data_norm.index.to_list(), y=merged_data_norm['spread normalized'],
                   name="spread normalized",
                   mode="lines", line=dict(width=2, color='green'), showlegend=True), secondary_y=True,
        col=1, row=1)
    fig_brainard.add_trace(
        go.Scatter(x=merged_data_norm.index.to_list(), y=merged_data_norm['cooper normalized'], name="cooper normalizd",
                   mode="lines", line=dict(width=2, color='blue'), showlegend=True), secondary_y=True,
        col=1, row=1)
    merged_data_spread_var = pd.DataFrame(merged_data.iloc[:, 0].diff())
    merged_data_5y_var = pd.DataFrame(merged_data.iloc[:, 1].diff())
    merged_data_cooper_ret = pd.DataFrame(merged_data.iloc[:, 2].pct_change())
    merged_ = pd.concat([merged_data_spread_var, merged_data_5y_var, merged_data_cooper_ret], axis=1)
    #   merged_['dummy_cooper'] = np.where(merged_['cooper'])
    merged_data_spread_var = pd.DataFrame(merged_data.iloc[:, 0].resample("3M").last().diff())
    merged_data_5y_var = pd.DataFrame(merged_data.iloc[:, 1].resample("3M").last().diff())
    merged_data_cooper_ret = (
        pd.DataFrame(merged_data.iloc[:, 2].resample("6M").agg(lambda x: x[-1]))).pct_change()
    merged_ = pd.concat([merged_data_spread_var, merged_data_5y_var, merged_data_cooper_ret], axis=1)
    merged_.dropna(inplace=True)
    merged_['dummy_cooper'] = np.where(merged_['cooper'] > 0, 1, -1)
    fig_brainard.add_trace(
        go.Scatter(x=merged_.index.to_list(), y=merged_["spread 30_5yr"], name="Spread 30-5y",
                   mode="lines", line=dict(width=2, color='white'), showlegend=False), secondary_y=False, col=1, row=2)
    fig_brainard.add_trace(go.Scatter(x=merged_.index.to_list(), y=merged_["5y"], name="US 5y",
                                      mode="lines", line=dict(width=2, color='purple'), showlegend=False),
                           secondary_y=False, col=1,
                           row=2)
    fig_brainard.add_trace(go.Scatter(x=merged_.index.to_list(), y=merged_["cooper"], name="Cooper prices",
                                      mode="lines", line=dict(width=2, color='orange'), showlegend=False),
                           secondary_y=True, col=1, row=2)
    fig_brainard.add_trace(go.Scatter(x=merged_.index.to_list(), y=merged_["dummy_cooper"], name="dummy Up/Down Cooper",
                                      mode="lines", line=dict(width=2, color='red'), showlegend=True), secondary_y=True,
                           col=1, row=1)

    fig_fred_search = go.Figure()

    fig_fred_search.add_trace(go.Scatter(x=single_search.index.to_list(), y=single_search._6m_smoothing_growth / 100,
                                         name="6m growth average",
                                         mode="lines", line=dict(width=2, color='white')))
    fig_fred_search.add_trace(go.Scatter(x=single_search_10.index.to_list(),
                                         y=single_search_10['10 yr average'] / 100,
                                         name="6m growth average",
                                         mode="lines", line=dict(width=2, color='green')))
    fig_secular_trends.layout.xaxis.range = [date_start, date_end]
    fig_secular_trends.layout.xaxis2.range = [date_start, date_end]
    fig_secular_trends.layout.xaxis3.range = [date_start, date_end]
    fig_secular_trends.layout.xaxis4.range = [date_start, date_end]
    fig_secular_trends.layout.xaxis5.range = [date_start, date_end]
    fig_secular_trends.layout.xaxis6.range = [date_start, date_end]

    fig_secular_trends_2.layout.xaxis.range = [date_start, date_end]
    fig_secular_trends_2.layout.xaxis2.range = [date_start, date_end]
    fig_secular_trends_2.layout.xaxis3.range = [date_start, date_end]
    fig_secular_trends_2.layout.xaxis4.range = [date_start, date_end]


    fig_cyclical_trends.layout.xaxis.range = [date_start, date_end]
    fig_cyclical_trends.layout.xaxis2.range = [date_start, date_end]
    fig_cyclical_trends.layout.xaxis3.range = [date_start, date_end]
    fig_cyclical_trends.layout.xaxis4.range = [date_start, date_end]
    fig_cyclical_trends.layout.xaxis5.range = [date_start, date_end]
    fig_cyclical_trends.layout.xaxis6.range = [date_start, date_end]

    fig_cyclical_trends.update_layout(xaxis_range=[date_start, date_end])

    fig_fred_search.update_layout(xaxis_range=[date_start, date_end])
    fig_.update_layout(
        template="plotly_dark",
        xaxis={'range':[str(date_start), str(date_end)]},
        title={
            'text': "COMPOSITE GROWTH",
            'y': 0.9,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'})

    fig_.update_layout(  # customize font and legend orientation & position
        yaxis=dict(tickformat=".0%"),

        title_font_family="Arial Black",
        font=dict(
            family="Rockwell",
            size=16),
        legend=dict(
            title=None, orientation="h", y=0.97, yanchor="bottom", x=0.5, xanchor="center"
        )
    )

    fig_fred_search.update_layout(height=650, width=1500)
    fig_fred_search.update_layout(
        template="plotly_dark",
        title={
            'text': fred.get_series_info(ticker_fred).title,
            'y': 0.9,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'})

    fig_fred_search.update_layout(  # customize font and legend orientation & position
        yaxis=dict(tickformat=".0%"),
        title_font_family="Arial Black",
        font=dict(
            family="Rockwell",
            size=16),
        legend=dict(
            title=None, orientation="h", y=0.97, yanchor="bottom", x=0.5, xanchor="center"
        )
    )
    fig_fred_search.update_layout(height=650, width=1500)
    fig_brainard.update_layout(
        template="plotly_dark",
        title={
            'text': "BRAINARD",
            'y': 1,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'})

    fig_brainard.update_layout(  # customize font and legend orientation & position

        title_font_family="Arial Black",
        font=dict(
            family="Rockwell",
            size=16),
        legend=dict(
            title=None, orientation="h", y=1, yanchor="bottom", x=0.5, xanchor="center"
        )
    )
    fig_brainard.update_layout(height=1000, width=1500)

    if tabs == "Macroeconomic Indicators":
        if dropdown == "Growth":
            return html.Div(dash_table.DataTable(score_table_merged.to_dict('records'),
                                                 [{"name": i, "id": i} for i in score_table_merged.columns],
                                                 sort_action='native',

                                                 style_data_conditional=[
                                                     {'if': {'column_id': 'Indicator'},
                                                      'width': '20px'},

                                                     {'if': {
                                                         'filter_query': '{Score} = 0',
                                                         # comparing columns to each other
                                                         'column_id': 'Score'
                                                     },
                                                         'backgroundColor': 'rgba(255, 36, 71, 1)'
                                                     },
                                                     {'if': {
                                                         'filter_query': '{Score} = 1',
                                                         # comparing columns to each other
                                                         'column_id': 'Score'
                                                     },
                                                         'backgroundColor': 'rgba(255, 36, 71, 0.4)'
                                                     },

                                                     {'if': {
                                                         'filter_query': '{Score} = 2',
                                                         # comparing columns to each other
                                                         'column_id': 'Score'
                                                     },
                                                         'backgroundColor': 'rgba(53, 108, 0, 1)'
                                                     },

                                                     {'if': {
                                                         'filter_query': '{Score} = 3',
                                                         # comparing columns to each other
                                                         'column_id': 'Score'
                                                     },
                                                         'backgroundColor': 'rgba(138, 255,0, 1)'
                                                     }
                                                 ],
                                                 fill_width=False,
                                                 style_header={'backgroundColor': 'rgb(30, 30, 30)', 'color': 'white'},
                                                 style_data={'backgroundColor': 'rgb(30, 30, 30)', 'color': 'white',
                                                             'whiteSpace': 'normal', 'height': 'auto'}),
                            style={'margin-left': '450px'}), dcc.Graph(figure=fig_), dcc.Graph(
                figure=fig_cyclical_trends)
        elif dropdown == "Inflation Outlook":

            return html.Div(dash_table.DataTable(score_table_merged_infla.to_dict('records'),
                                                 [{"name": i, "id": i} for i in score_table_merged_infla.columns],
                                                 sort_action='native',

                                                 style_data_conditional=[
                                                     {'if': {'column_id': 'Indicator'},
                                                      'width': '20px'},

                                                     {'if': {
                                                         'filter_query': '{Score} = 0',
                                                         # comparing columns to each other
                                                         'column_id': 'Score'
                                                     },
                                                         'backgroundColor': 'rgba(255, 36, 71, 1)'
                                                     },
                                                     {'if': {
                                                         'filter_query': '{Score} = 1',
                                                         # comparing columns to each other
                                                         'column_id': 'Score'
                                                     },
                                                         'backgroundColor': 'rgba(255, 36, 71, 0.4)'
                                                     },

                                                     {'if': {
                                                         'filter_query': '{Score} = 2',
                                                         # comparing columns to each other
                                                         'column_id': 'Score'
                                                     },
                                                         'backgroundColor': 'rgba(138, 255,0, 1)'
                                                     },

                                                     {'if': {
                                                         'filter_query': '{Score} = 3',
                                                         # comparing columns to each other
                                                         'column_id': 'Score'
                                                     },
                                                         'backgroundColor': 'rgba(53, 108, 0, 1)'
                                                     }
                                                 ],
                                                 fill_width=False,
                                                 style_header={'backgroundColor': 'rgb(30, 30, 30)', 'color': 'white'},
                                                 style_data={'backgroundColor': 'rgb(30, 30, 30)', 'color': 'white',
                                                             'whiteSpace': 'normal', 'height': 'auto'}),
                            style={"margin-left": "450px"}), \
                dcc.Graph(figure=fig_secular_trends_2), dcc.Graph(figure=fig_secular_trends)
        elif dropdown == "brainard_test":
            return dcc.Graph(figure=fig_brainard)
        elif dropdown == "fred_search":
            return dcc.Graph(figure=fig_fred_search)

    elif tabs == "Directional Indicators":

        pass

    elif tabs == "Trend-Momentum Indicators":

        pass

    elif tabs == "Volatility Indicators":

        pass


#   except ValueError:
# pass
if __name__ == "__main__":
    # host = socket.gethostbyname(socket.gethostname())
    app.run_server(debug=True, port=8080)




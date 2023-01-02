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
import socket
import os
import datetime
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split


def smooth_data(ticker, date_start, date_start2, date_end):
    frequency = "monthly"
    fred = Fred(api_key='f40c3edb57e906557fcac819c8ab6478')
    data_ = pd.DataFrame(
        fred.get_series(ticker, observation_start=date_start, observation_end=date_end, freq=frequency))
    data_2 = pd.DataFrame(
        fred.get_series(ticker, observation_start=date_start2, observation_end=date_end, freq=frequency))

    # creating 6m smoothing growth column
    data_['_6m_smoothing_growth'] = 100 * ((data_.iloc[:, 0][11:] / data_.iloc[:, 0].rolling(
        12).mean() - 1) * 2)[
                                          len(data_) - 19:]
    data_2['10 yr average'] = 100 * (data_2.iloc[:, 0].rolling(10).mean().pct_change())

    # creating 6m smoothing growth column and 10yr average column
    data_['_6m_smoothing_growth'] = 100 * ((data_.iloc[:, 0][11:] / data_.iloc[:, 0].rolling(
        12).mean() - 1) * 2)[len(data_) - 19:]
    data_2['10 yr average'] = 100 * (data_2.iloc[:, 0].rolling(10).mean().pct_change())
    data_.dropna(inplace=True)
    data_2.dropna(inplace=True)
    return data_[['_6m_smoothing_growth']], data_2[['10 yr average']]

def score_table(index,data_,data_10):
    score_table = pd.DataFrame.from_dict({"trend vs history ": 1 if data_.iloc[-1, 0] > data_10.iloc[-1, 0] else 0,
                                          "growth": 1 if data_.iloc[-1, 0] > 0 else 0,
                                          "Direction of Trend": 1 if (data_.resample("3M").last().diff())[-2:].sum()[
                                                                         0] > 1 else 0}, orient="index").T
    score_table['Score'] = score_table.sum(axis=1)
    score_table['Indicators']= index

    return score_table
def fred_data(ticker):
    date_start = "2017-01-01"
    date_end = "2022-05-27"
    frequency = 'monthly'

    fred = Fred(api_key='f40c3edb57e906557fcac819c8ab6478')
    # get data as an array and transforming it into a dataframe
    return fred.get_series(ticker, observation_start=date_start, observation_end=date_end, freq=frequency)


#(new factor) : variation of real rate
# Parameters for the request from the FRED Website

date_start_ = "2017-01-01"
date_end_ = "2022-12-10"

frequency_ = 'monthly'

external_stylesheets = [
    {
        "href": "https://codepen.io/chriddyp/pen/bWLwgP.css"
                "family=Lato:wght@400;700&display=swap",
        "rel": "stylesheet",
    },]

app = Dash(__name__, external_stylesheets = external_stylesheets)

app.config.suppress_callback_exceptions = True

timeout = 20

app.layout = html.Div(style = {'backgroundColor':"rgba(0, 0, 0,0)"},children = [
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

    html.H1("Economic Framework (Talbi & Co)",style={"margin-left":"550px"}),
    html.Br(),
    html.Div([
            dcc.Input(date_start_,placeholder="start date",id="date_start"),
            dcc.Input(date_end_,placeholder="end date",id="date_end")],style={'margin-left':'650px'}),
              html.Div(id="graph_indicator",style={'margin-left':'100px'}),

    #html.Div(,style={"margin-top":"10px"}),

        html.Div(dcc.Dropdown( id = 'dropdown',
        options = [
            {'label':'brainard_test', 'value':'brainard_test' },
            {'label':'Growth', 'value':'Growth' },
            {'label': 'Inflation Outlook', 'value':'Inflation Outlook'},
            ],
        value = 'Growth'),style={"width":"150px","margin-left":"120px"}),
    html.Div(id="trends_graphs",style={"margin-left":"100px"})
    ])


"""
@app.callback(Output("graph_indicator","children"),
              #Input("interval_comp","n_intervals"),
              [Input("tabs","active_tab"),
               Input("ticker","value"),
               Input("date_start","value"),
               Input("date_end","value"),
              ]

)
def smoothed_2(tabs,ticker, date_start, date_end):
    try:

        fred = Fred(api_key='f40c3edb57e906557fcac819c8ab6478')
        date_start2 = "2009-01-01"
        # get data as an array and transforming it into a dataframe

        indpro,indpro_10 = smooth_data(ticker,date_start,date_start2,date_end)


        fig_ = go.Figure()

        # drop the blank values


        # ploting the data
        fig_.add_trace(go.Scatter(x=indpro.index.to_list(),y =indpro._6m_smoothing_growth/100,name="6m growth average",mode="lines",line=dict(width=2, color='white')))
        fig_.add_trace(go.Scatter(x=(indpro_10.index.to_list())[len(indpro_10) - 19:],y =indpro_10['10 yr average'][len(indpro_10) - 19:]/100,mode="lines",line=dict(width=2, color='green'),name="10yr average"))

        fig_.update_layout(
            template="plotly_dark",
            title={
                'text': fred.get_series_info(ticker).title,
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
        fig_.update_layout(height=650,width=1150)

        # saving the figures in the static file
        #plt.savefig('/Users/talbi/Downloads/microblog/app/static/' + ticker + ".png")
        if tabs == "Macroeconomic Indicators":
            return dcc.Graph(figure=fig_),
        elif tabs == "Directional Indicators":
            pass
        elif tabs == "Trend-Momentum Indicators":
            pass
        elif tabs == "Volatility Indicators":
            pass

    except ValueError:
        pass

"""
@app.callback(Output("trends_graphs", "children"),
              # Input("interval_comp","n_intervals"),
              [Input("dropdown", "value"),
               Input("tabs","active_tab"),
               Input("date_start","value"),
               Input("date_end","value"),

               ]
              )
def trends(dropdown,tabs,date_start,date_end):
    fred = Fred(api_key='f40c3edb57e906557fcac819c8ab6478')
    date_start2 = "2009-01-01"
    print("1")
    #try:
    PATH_DATA = "/Users/talbi/Downloads/"

    _30y = pd.read_csv(PATH_DATA + "30y.csv").iloc[:, :2]
    _30y.set_index("Date", inplace=True, drop=True)
    _30y['Dernier'] = _30y['Dernier'].apply(lambda x: float(x.replace(",", ".")))
    _30y_index = pd.Series(_30y.index.to_list()[::-1]).apply(lambda x: datetime.datetime.strptime(x, "%d/%m/%Y"))
    _30y = _30y[::-1]
    _30y.index = _30y_index

    _5y_nominal = pd.read_csv(PATH_DATA + "5y.csv").iloc[:, :2]
    _5y_nominal.set_index("Date", inplace=True, drop=True)
    _5y_nominal['Dernier'] = _5y_nominal['Dernier'].apply(lambda x: float(x.replace(",", ".")))
    _5y_nominal_index = pd.Series(_5y_nominal.index.to_list()[::-1]).apply(lambda x: datetime.datetime.strptime(x, "%d/%m/%Y"))
    _5y_nominal = _5y_nominal[::-1]
    _5y_nominal.index = _5y_nominal_index

    _5y_nominal_var = _5y_nominal.diff()

    _5y_real = pd.DataFrame(
        fred.get_series("DFII5", observation_start="1970-01-01", observation_end="2022-12-30", freq="daily"))
    _5y_real.columns = ['Dernier']

    spread = _30y - _5y_real

    cooper_prices = pd.read_csv(PATH_DATA + "cooper_prices.csv").iloc[:, :2]
    cooper_prices.set_index("Date", inplace=True, drop=True)
    cooper_prices['Dernier'] = cooper_prices['Dernier'].apply(lambda x: float(x.replace(",", ".")))
    cooper_prices_index = pd.Series(cooper_prices.index.to_list()[::-1]).apply(
        lambda x: datetime.datetime.strptime(x, "%d/%m/%Y"))
    cooper_prices = cooper_prices[::-1]*100
    cooper_prices.index = cooper_prices_index
    merged_data = pd.concat([spread, _5y_nominal, cooper_prices], axis=1)
    merged_data.dropna(inplace=True)
    merged_data.columns = ["spread 30_5yr", "5y", "cooper"]

    # Data importing
    print("2")
    cpi, cpi_10 = smooth_data("CPIAUCSL", date_start, date_start2, date_end)

    pcec96, pcec96_10 = smooth_data("PCEC96", date_start, date_start2, date_end)
    #pcec96 = pcec96[["_6m_smoothing_growth"]]-cpi[["_6m_smoothing_growth"]]
    #pcec96_10 = pcec96_10[['10 yr average']]-cpi_10[['10 yr average']]
    print("3")
    indpro, indpro_10 = smooth_data("INDPRO", date_start, date_start2, date_end)
    print('4')
    nonfarm, nonfarm_10 = smooth_data("PAYEMS", date_start, date_start2, date_end)
    print("5")
    real_pers, real_pers_10 = smooth_data("W875RX1", date_start, date_start2, date_end)
    #real_pers = real_pers[["_6m_smoothing_growth"]] - cpi[["_6m_smoothing_growth"]]
    #real_pers_10 = real_pers_10[['10 yr average']] - cpi_10[['10 yr average']]


    retail_sales,retail_sales_10 = smooth_data("RRSFS", date_start, date_start2, date_end)
    #retail_sales = retail_sales[["_6m_smoothing_growth"]] - cpi[["_6m_smoothing_growth"]]
    #retail_sales_10 = retail_sales_10[['10 yr average']] - cpi_10[['10 yr average']]


    employment_level, employment_level_10 = smooth_data("CE16OV", date_start, date_start2, date_end)
    employment_level.dropna(inplace=True)
    shelter_prices,shelter_prices_10 = smooth_data("CUSR0000SAH1", date_start, date_start2, date_end)
    shelter_prices = shelter_prices[["_6m_smoothing_growth"]] - cpi[["_6m_smoothing_growth"]]
    shelter_prices_10 = shelter_prices_10[['10 yr average']] - cpi_10[['10 yr average']]




    cwd = os.getcwd()
    # wget.download("http://atlantafed.org/-/media/documents/datafiles/chcs/wage-growth-tracker/wage-growth-data.xlsx")
    wage_tracker = pd.DataFrame(pd.read_excel(cwd + "/wage-growth-data.xlsx").iloc[3:, [0, 11]])
    wage_tracker.columns = ['date', "wage_tracker"]
    wage_tracker.set_index('date', inplace=True)



    employment_level_wage_tracker = pd.concat([employment_level, wage_tracker], axis=1)
    employment_level_wage_tracker.dropna(inplace=True)
    wages, wages_10 = smooth_data("CES0500000003", date_start, date_start2, date_end)
    wages = wages[["_6m_smoothing_growth"]] - cpi[["_6m_smoothing_growth"]]
    wages_10 = wages_10[['10 yr average']] - cpi_10[['10 yr average']]


    core_cpi, core_cpi_10 = smooth_data("CPILFESL", date_start, date_start2, date_end)
    core_pce, core_pce_10 = smooth_data("DPCCRC1M027SBEA", date_start, date_start2, date_end)




    composite_data = pd.concat([pcec96[['_6m_smoothing_growth']],indpro[['_6m_smoothing_growth']],nonfarm[['_6m_smoothing_growth']],real_pers[['_6m_smoothing_growth']],retail_sales[['_6m_smoothing_growth']],employment_level[['_6m_smoothing_growth']]],axis=1)
    composite_data.dropna(inplace=True)
    composite_growth = pd.DataFrame(composite_data.sum(axis=1))
    composite_growth.columns = ["_6m_smoothing_growth"]
    composite_growth_10 = pd.concat([pcec96_10[['10 yr average']],indpro_10[['10 yr average']],nonfarm_10[['10 yr average']],real_pers_10[['10 yr average']],retail_sales_10[['10 yr average']],employment_level_10[['10 yr average']]],axis=1)
    composite_growth_10.dropna(inplace=True)
    composite_growth_10 = pd.DataFrame(composite_growth_10.sum(axis=1))
    composite_growth_10.columns=["10 yr average"]

    print("6")
    # pcec96 =

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

    score_table_merged = pd.concat([score_table(pce_title, pcec96, pcec96_10), score_table(indpro_title,indpro, indpro_10),
                                    score_table(nonfarm_title,nonfarm, nonfarm_10),
                                    score_table(real_personal_income_title,real_pers,real_pers_10),
                                    score_table(retail_sales_title,retail_sales,retail_sales_10),
                                    score_table(employment_level_title,employment_level,employment_level_10),
                                    score_table("COMPOSITE GROWTH",composite_growth,composite_growth_10)], axis = 0)

    score_table_merged_infla = pd.concat([score_table(cpi_title, cpi, cpi_10), score_table(indpro_title,indpro, indpro_10),
                                    score_table(core_cpi_title,core_cpi, core_cpi_10),
                                    score_table(pce_title,pcec96,pcec96_10),
                                    score_table(core_pce_title, core_pce, core_pce_10),
                                    score_table(shelter_title,shelter_prices,shelter_prices_10)], axis = 0)
    score_table_merged = score_table_merged.iloc[:,[4,0,1,2,3]]
    score_table_merged_infla = score_table_merged_infla.iloc[:, [4, 0, 1, 2, 3]]
    #score_table_merged.set_index("index", inplace=True)

    print(dropdown)


    fig_cyclical_trends = make_subplots(rows=3, cols=2,subplot_titles=[pce_title,indpro_title
                                                                       ,nonfarm_title,real_personal_income_title,retail_sales_title,employment_level_title])


    fig_cyclical_trends.add_trace(go.Scatter(x=pcec96.index.to_list(), y=pcec96._6m_smoothing_growth/100, name="6m growth average",mode="lines",line=dict(width=2, color='white')),row=1, col=1)
    fig_cyclical_trends.add_trace(go.Scatter(x=(pcec96_10.index.to_list())[len(pcec96_10) - 19:],
                                             y=(pcec96_10['10 yr average'][len(pcec96_10) - 19:])/100,mode="lines",line=dict(width=2, color='green'),
                                             name="10yr average"),row=1, col=1)

    fig_cyclical_trends.add_trace(go.Scatter(x=indpro.index.to_list(), y=indpro._6m_smoothing_growth/100, name="6m growth average",mode="lines",line=dict(width=2, color='white'),showlegend=False), row=1,col=2)
    fig_cyclical_trends.add_trace(go.Scatter(x=(indpro_10.index.to_list())[len(indpro_10) - 19:],
                                             y=indpro_10['10 yr average'][len(indpro_10) - 19:]/100,line=dict(width=2, color='green'),mode="lines",
                                             name="10yr average",showlegend=False), row=1, col=2)

    fig_cyclical_trends.add_trace(go.Scatter(x=nonfarm.index.to_list(), y=nonfarm._6m_smoothing_growth/100, name="6m growth average",mode="lines",line=dict(width=2, color='white'),showlegend=False), row=2,col=1)
    fig_cyclical_trends.add_trace(go.Scatter(x=(nonfarm_10.index.to_list())[len(nonfarm_10) - 19:],
                                             y=nonfarm_10['10 yr average'][len(nonfarm_10) - 19:]/100,line=dict(width=2, color='green'),mode="lines",
                                             name="10yr average",showlegend=False), row=2, col=1)

    fig_cyclical_trends.add_trace(go.Scatter(x=real_pers.index.to_list(), y=real_pers._6m_smoothing_growth/100, name="6m growth average",mode="lines",line=dict(width=2, color='white'),showlegend=False), row=2,col=2)
    fig_cyclical_trends.add_trace(go.Scatter(x=(real_pers_10.index.to_list())[len(real_pers_10) - 19:],
                                             y=real_pers_10['10 yr average'][len(real_pers_10) - 19:]/100,line=dict(width=2, color='green'),mode="lines",
                                             name="10yr average",showlegend=False), row=2, col=2)

    fig_cyclical_trends.add_trace(
        go.Scatter(x=retail_sales.index.to_list(), y=retail_sales._6m_smoothing_growth / 100, name="6m growth average",
                   mode="lines", line=dict(width=2, color='white'), showlegend=False), row=3, col=1)
    fig_cyclical_trends.add_trace(go.Scatter(x=(retail_sales_10.index.to_list())[len(retail_sales_10) - 19:],
                                             y=retail_sales_10['10 yr average'][len(retail_sales_10) - 19:] / 100,
                                             line=dict(width=2, color='green'), mode="lines",
                                             name="10yr average", showlegend=False), row=3, col=1)

    fig_cyclical_trends.add_trace(
        go.Scatter(x=employment_level.index.to_list(), y=employment_level._6m_smoothing_growth / 100, name="6m growth average",
                   mode="lines", line=dict(width=2, color='white'), showlegend=False), row=3, col=2)
    fig_cyclical_trends.add_trace(go.Scatter(x=(employment_level_10.index.to_list())[len(employment_level_10) - 19:],
                                             y=employment_level_10['10 yr average'][len(employment_level_10) - 19:] / 100,
                                             line=dict(width=2, color='green'), mode="lines",
                                             name="10yr average", showlegend=False), row=3, col=2)

    fig_cyclical_trends.update_layout(template="plotly_dark",
                                      height=1000,width=1500)
    fig_cyclical_trends.update_layout(  # customize font and legend orientation & position
        yaxis=dict(tickformat=".0%"),
        title_font_family="Arial Black",
        font=dict(
            family="Rockwell",
            size=18)
    )
    fig_cyclical_trends.layout.yaxis2.tickformat =".1%"
    fig_cyclical_trends.layout.yaxis3.tickformat = ".1%"
    fig_cyclical_trends.layout.yaxis4.tickformat = ".1%"
    fig_cyclical_trends.layout.yaxis5.tickformat = ".1%"
    fig_cyclical_trends.layout.yaxis6.tickformat = ".1%"



    fig_secular_trends = make_subplots(rows=3, cols=2, specs=[[{"secondary_y": True},{"secondary_y": True}],[{"secondary_y": True},{"secondary_y": True}],[{"secondary_y": True},{"secondary_y": True}]],subplot_titles=[cpi_title, core_cpi_title
        , pce_title, core_pce_title,shelter_title,"Employment (Growth) and Wage tracker levels"])

    fig_secular_trends.add_trace(
        go.Scatter(x=cpi.index.to_list(), y=cpi._6m_smoothing_growth / 100, name="6m growth average",
                   mode="lines", line=dict(width=2, color='white')),secondary_y=False, row=1, col=1)
    fig_secular_trends.add_trace(go.Scatter(x=(cpi_10.index.to_list())[len(cpi_10) - 19:],
                                             y=(cpi_10['10 yr average'][len(cpi_10) - 19:]) / 100, mode="lines",
                                             line=dict(width=2, color='green'),
                                             name="10yr average"),secondary_y=False, row=1, col=1)

    fig_secular_trends.add_trace(
        go.Scatter(x=core_cpi.index.to_list(), y=core_cpi._6m_smoothing_growth / 100, name="6m growth average",
                   mode="lines", line=dict(width=2, color='white'), showlegend=False),secondary_y=False, row=1, col=2)
    fig_secular_trends.add_trace(go.Scatter(x=(core_cpi_10.index.to_list())[len(core_cpi_10) - 19:],
                                             y=core_cpi_10['10 yr average'][len(core_cpi_10) - 19:] / 100,
                                             line=dict(width=2, color='green'), mode="lines",
                                             name="10yr average", showlegend=False),secondary_y=False, row=1, col=2)

    fig_secular_trends.add_trace(
        go.Scatter(x=pcec96.index.to_list(), y=pcec96._6m_smoothing_growth / 100, name="6m growth average",
                   mode="lines", line=dict(width=2, color='white'), showlegend=False),secondary_y=False, row=2, col=1)
    fig_secular_trends.add_trace(go.Scatter(x=(pcec96_10.index.to_list())[len(pcec96_10) - 19:],
                                             y=pcec96_10['10 yr average'][len(pcec96_10) - 19:] / 100,
                                             line=dict(width=2, color='green'), mode="lines",
                                             name="10yr average", showlegend=False),secondary_y=False, row=2, col=1)

    fig_secular_trends.add_trace(
        go.Scatter(x=core_pce.index.to_list(), y=core_pce._6m_smoothing_growth / 100, name="6m growth average",
                   mode="lines", line=dict(width=2, color='white'), showlegend=False), row=2, col=2)
    fig_secular_trends.add_trace(go.Scatter(x=(core_pce_10.index.to_list())[len(core_pce_10) - 19:],
                                             y=core_pce_10['10 yr average'][len(core_pce_10) - 19:] / 100,
                                             line=dict(width=2, color='green'), mode="lines",
                                             name="10yr average", showlegend=False),secondary_y=False, row=2, col=2)
    fig_secular_trends.add_trace(
        go.Scatter(x=shelter_prices.index.to_list(), y=shelter_prices._6m_smoothing_growth / 100, name="6m growth average",
                   mode="lines", line=dict(width=2, color='white'), showlegend=False),secondary_y=False, row=3, col=1)
    fig_secular_trends.add_trace(go.Scatter(x=(shelter_prices_10.index.to_list())[len(shelter_prices_10) - 19:],
                                            y=shelter_prices_10['10 yr average'][len(shelter_prices_10) - 19:] / 100,
                                            line=dict(width=2, color='green'), mode="lines",
                                            name="10yr average", showlegend=False),secondary_y=False, row=3, col=1)


    fig_secular_trends.add_trace(
        go.Scatter(x=employment_level_wage_tracker.index.to_list(), y=employment_level_wage_tracker._6m_smoothing_growth, name="Employment level 6m annualized growth",
                   mode="lines", line=dict(width=2, color='white'), showlegend=True),secondary_y=False, row=3, col=2)
    fig_secular_trends.add_trace( go.Scatter(x=employment_level_wage_tracker.index.to_list(), y=employment_level_wage_tracker.wage_tracker, name="Atlanta Fed wage tracker",
                   mode="lines", line=dict(width=2, color='blue'), showlegend=True),secondary_y=True, row=3, col=2)
    fig_secular_trends.update_layout(template="plotly_dark",
                                      height=1200, width=1500)
    fig_secular_trends.update_layout(  # customize font and legend orientation & position
        yaxis=dict(tickformat=".1%"),
        title_font_family="Arial Black",
        font=dict(
            family="Rockwell",
            size=18)
        )

    fig_secular_trends.layout.yaxis2.tickformat = ".1%"
    fig_secular_trends.layout.yaxis3.tickformat = ".1%"
    fig_secular_trends.layout.yaxis4.tickformat = ".1%"
    fig_secular_trends.layout.yaxis5.tickformat = ".1%"

    fig_ = go.Figure()

    # drop the blank values

    # ploting the data
    #composite_growth_10 = 100 * (composite_growth.iloc[:, 0].rolling(10).mean().pct_change())
    fig_.add_trace(go.Scatter(x=composite_growth.index.to_list(), y=composite_growth._6m_smoothing_growth / 100, name="6m growth average",
                              mode="lines", line=dict(width=2, color='white')))
    fig_.add_trace(go.Scatter(x=composite_growth_10.index.to_list()[len(composite_growth_10) - 19:], y=composite_growth_10['10 yr average'][len(composite_growth_10) - 19:] / 100, name="6m growth average",
                              mode="lines", line=dict(width=2, color='green')))

    for x, y in zip(composite_growth.index[len(composite_growth) - 19:len(composite_growth):3],
                    composite_growth.iloc[len(composite_growth) - 19:len(composite_growth):3,0]/ 100):
        label = "{:.1f}".format(y*100) + " %"
        fig_.add_annotation(x=x, y=y,
                           text=label,
                           showarrow=False, yshift=10)

    for x, y in zip(composite_growth_10.index[len(composite_growth_10) - 19:len(composite_growth_10):3],
                    composite_growth_10.iloc[len(composite_growth_10) - 19:len(composite_growth_10):3,0]/ 100):
        label = "{:.1f}".format(y*100) + " %"
        fig_.add_annotation(x=x, y=y,
                           text=label,
                           showarrow=False, yshift=10)

    fig_brainard = make_subplots(cols=1,rows=2,specs=[[{"secondary_y": True}],[{"secondary_y": True}]],subplot_titles=["Levels","Returns"])

    fig_brainard.add_trace(
        go.Scatter(x=merged_data.index.to_list(), y=merged_data.iloc[:,0] , name="Spread",
                   mode="lines", line=dict(width=2, color='white')),secondary_y=False,col=1,row=1)
    fig_brainard.add_trace(go.Scatter(x=merged_data.index.to_list(), y=merged_data.iloc[:,1] , name="US 5y",
                   mode="lines", line=dict(width=2, color='purple')),secondary_y=False,col=1,row=1)
    fig_brainard.add_trace(go.Scatter(x=merged_data.index.to_list(), y=merged_data.iloc[:,2], name="Cooper prices",
                                      mode="lines", line=dict(width=2, color='orange')),secondary_y=True,col=1,row=1)

    merged_data_spread_var = pd.DataFrame(merged_data.iloc[:, 0].diff())
    merged_data_5y_var = pd.DataFrame(merged_data.iloc[:, 1].diff())
    merged_data_cooper_ret = pd.DataFrame(merged_data.iloc[:, 2].pct_change())
    merged_ = pd.concat([merged_data_spread_var,merged_data_5y_var,merged_data_cooper_ret],axis=1)
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
                   mode="lines", line=dict(width=2, color='white'),showlegend=False), secondary_y=False, col=1, row=2)
    fig_brainard.add_trace(go.Scatter(x=merged_.index.to_list(), y=merged_["5y"], name="US 5y",
                                      mode="lines", line=dict(width=2, color='purple'),showlegend=False), secondary_y=False, col=1,
                           row=2)
    fig_brainard.add_trace(go.Scatter(x=merged_.index.to_list(), y=merged_["cooper"], name="Cooper prices",
                                      mode="lines", line=dict(width=2, color='orange'),showlegend=False), secondary_y=True, col=1, row=2)
    fig_brainard.add_trace(go.Scatter(x=merged_.index.to_list(), y=merged_["dummy_cooper"], name="dummy Up/Down Cooper",
                                      mode="lines", line=dict(width=2, color='red'),showlegend=True), secondary_y=True, col=1, row=1)

    fig_.update_layout(
        template="plotly_dark",
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
    fig_.update_layout(height=650, width=1500)

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
    fig_brainard.update_layout(height=1250, width=1500)


    if tabs == "Macroeconomic Indicators":
        if dropdown == "Growth":
            return html.Div(dash_table.DataTable(score_table_merged.to_dict('records'),
                                     [{"name": i, "id": i} for i in score_table_merged.columns],
                                     sort_action='native',

                                     style_data_conditional=  [
                                         {'if': {'column_id': 'Indicator'},
                                          'width': '20px'},

                                         {'if': {
                                            'filter_query': '{Score} = 0', # comparing columns to each other
                                            'column_id': 'Score'
                                        },
                                        'backgroundColor': 'rgba(255, 36, 71, 1)'
                                    },
                                         {'if': {
                                             'filter_query': '{Score} = 1',  # comparing columns to each other
                                             'column_id': 'Score'
                                         },
                                             'backgroundColor': 'rgba(255, 36, 71, 0.4)'
                                         },

                                         {'if': {
                                             'filter_query': '{Score} = 2',  # comparing columns to each other
                                             'column_id': 'Score'
                                         },
                                             'backgroundColor': 'rgba(138, 255,0, 1)'
                                         },

                                         {'if': {
                                             'filter_query': '{Score} = 3',  # comparing columns to each other
                                             'column_id': 'Score'
                                         },
                                             'backgroundColor': 'rgba(53, 108, 0, 1)'
                                         }
                                     ],
                                     fill_width=False,
                                     style_header={'backgroundColor': 'rgb(30, 30, 30)','color': 'white'},
                                     style_data={'backgroundColor': 'rgb(30, 30, 30)','color': 'white','whiteSpace': 'normal','height': 'auto'}),style={'margin-left':'250px'}),dcc.Graph(figure=fig_),dcc.Graph(figure=fig_cyclical_trends)
        elif dropdown == "Inflation Outlook":

            return html.Div(dash_table.DataTable(score_table_merged_infla.to_dict('records'),
                                     [{"name": i, "id": i} for i in score_table_merged_infla.columns],
                                     sort_action='native',

                                     style_data_conditional=  [
                                         {'if': {'column_id': 'Indicator'},
                                          'width': '20px'},

                                         {'if': {
                                            'filter_query': '{Score} = 0', # comparing columns to each other
                                            'column_id': 'Score'
                                        },
                                        'backgroundColor': 'rgba(255, 36, 71, 1)'
                                    },
                                         {'if': {
                                             'filter_query': '{Score} = 1',  # comparing columns to each other
                                             'column_id': 'Score'
                                         },
                                             'backgroundColor': 'rgba(255, 36, 71, 0.4)'
                                         },

                                         {'if': {
                                             'filter_query': '{Score} = 2',  # comparing columns to each other
                                             'column_id': 'Score'
                                         },
                                             'backgroundColor': 'rgba(138, 255,0, 1)'
                                         },

                                         {'if': {
                                             'filter_query': '{Score} = 3',  # comparing columns to each other
                                             'column_id': 'Score'
                                         },
                                             'backgroundColor': 'rgba(53, 108, 0, 1)'
                                         }
                                     ],
                                     fill_width=False,
                                     style_header={'backgroundColor': 'rgb(30, 30, 30)','color': 'white'},
                                     style_data={'backgroundColor': 'rgb(30, 30, 30)','color': 'white','whiteSpace': 'normal','height': 'auto'}),style={"margin-left":"250px"}),dcc.Graph(figure=fig_secular_trends)
        elif dropdown == "brainard_test":
            return dcc.Graph(figure=fig_brainard)

    elif tabs == "Directional Indicators":

        pass

    elif tabs == "Trend-Momentum Indicators":

        pass

    elif tabs == "Volatility Indicators":

        pass
#   except ValueError:
        #pass
if __name__ == "__main__":
    # host = socket.gethostbyname(socket.gethostname())
    app.run_server(debug=True, port=8080)




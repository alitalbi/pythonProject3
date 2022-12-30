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
    return data_, data_2
def fred_data(ticker):
    date_start = "2017-01-01"
    date_end = "2022-05-27"
    frequency = 'monthly'

    fred = Fred(api_key='f40c3edb57e906557fcac819c8ab6478')
    # get data as an array and transforming it into a dataframe
    return fred.get_series(ticker, observation_start=date_start, observation_end=date_end, freq=frequency)

# Parameters for the request from the FRED Website

date_start_ = "2017-01-01"
date_end_ = "2022-12-10"

frequency_ = 'monthly'
#ticker_ = "INDPRO"






external_stylesheets = [
    {
        "href": "https://fonts.googleapis.com/css2?"
                "family=Lato:wght@400;700&display=swap",
        "rel": "stylesheet",
    },]

app = Dash(__name__, external_stylesheets = external_stylesheets)

app.config.suppress_callback_exceptions = True

timeout = 20

app.layout = html.Div(style = {'backgroundColor':"rgb(255, 255, 255)"},children = [
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

    html.H1("Secular & Cyclical Economic Framework (Talbi & Co)",style={"margin-left":"550px"}),
    html.Br(),
    html.Div([dcc.Input("INDPRO",placeholder="ticker",id="ticker"),
            dcc.Input(date_start_,placeholder="start date",id="date_start"),
            dcc.Input(date_end_,placeholder="end date",id="date_end")],style={'margin-left':'650px'}),
              html.Div(id="graph_indicator",style={'margin-left':'300px'}),

    #html.Div(,style={"margin-top":"10px"}),

        html.Div(dcc.Dropdown( id = 'dropdown',
        options = [
            {'label':'Cyclical Trends (6-18 month view)', 'value':'cyclical_trends' },
            {'label': 'Secular Trends (3-5 year view)', 'value':'secular_trends'},
            ],
        value = 'cyclical_trends'),style={"width":"350px","margin-left":"300px"}),
    html.Div(id="trends_graphs",style={"margin-left":"100px"})
    ])



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


@app.callback(Output("trends_graphs", "children"),
              # Input("interval_comp","n_intervals"),
              [Input("dropdown", "value"),
               Input("tabs","active_tab"),
               Input("date_start","value"),
               Input("date_end","value"),
               ]
              )
def trends(dropdown,tabs,date_start,date_end):
    date_start2 = "2009-01-01"
    print("1")
    #try:

    fred = Fred(api_key='f40c3edb57e906557fcac819c8ab6478')
    # Data importing
    print("2")
    pcec96, pcec96_10 = smooth_data("PCEC96", date_start, date_start2, date_end)
    print("3")
    indpro, indpro_10 = smooth_data("INDPRO", date_start, date_start2, date_end)
    print('4')
    nonfarm, nonfarm_10 = smooth_data("PAYEMS", date_start, date_start2, date_end)
    print("5")
    real_pers, real_pers_10 = smooth_data("W875RX1", date_start, date_start2, date_end)

    retail_sales,retail_sales_10 = smooth_data("MRTSSM44X72USS", date_start, date_start2, date_end)

    employment_level, employment_level_10 = smooth_data("CE16OV", date_start, date_start2, date_end)
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

    fig_secular_trends = make_subplots(rows=2, cols=2)
    fig_secular_trends.add_trace(go.Scatter(), row=1, col=1)
    fig_secular_trends.add_trace(go.Scatter(), row=1, col=2)
    fig_secular_trends.add_trace(go.Scatter(), row=2, col=1)
    fig_secular_trends.add_trace(go.Scatter(), row=2, col=2)


    if tabs == "Macroeconomic Indicators":
        if dropdown == "cyclical_trends":
            return dcc.Graph(figure=fig_cyclical_trends)
        elif dropdown == "secular_trends":

            return dcc.Graph(figure=fig_secular_trends)


    elif tabs == "Directional Indicators":

        pass

    elif tabs == "Trend-Momentum Indicators":

        pass

    elif tabs == "Volatility Indicators":

        pass
#   except ValueError:
        #pass
if __name__ == "__main__":
    #host = socket.gethostbyname(socket.gethostname())
    app.run_server(debug=True,  port=8080)




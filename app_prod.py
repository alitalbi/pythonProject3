import dash
from dash import Dash, dcc, html, Input, Output
import dash_table
import plotly.graph_objects as go
import plotly.express as px
from fredapi import Fred
import pandas as pd
import matplotlib.pyplot as plt
import os
import datetime

def fred_data(ticker):
    date_start = "2017-01-01"
    date_end = "2022-05-27"
    frequency = 'monthly'

    fred = Fred(api_key='f40c3edb57e906557fcac819c8ab6478')
    # get data as an array and transforming it into a dataframe
    return fred.get_series(ticker, observation_start=date_start, observation_end=date_end, freq=frequency)

# Parameters for the request from the FRED Website

date_start_ = "2017-01-01"
date_end_ = "2022-04-10"

frequency_ = 'monthly'
ticker_ = "INDPRO"






external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = Dash(__name__, external_stylesheets = external_stylesheets)

app.config.suppress_callback_exceptions = True

timeout = 20

app.layout = html.Div([
   # dcc.Interval(id="interval_comp",interval=15*1000),

    html.Div([dcc.Input(ticker_,placeholder="ticker",id="ticker"),
            dcc.Input(date_start_,placeholder="start date",id="date_start"),
            dcc.Input(date_end_,placeholder="end date",id="date_end"),
            dcc.Input(frequency_,placeholder="frequency" ,id="frequency")]),
    dcc.Graph(id="graph_indicator")


])


@app.callback(Output("graph_indicator","figure"),
              #Input("interval_comp","n_intervals"),
              [Input("ticker","value"),
               Input("date_start","value"),
               Input("date_end","value"),
               Input("frequency","value")]

)
def smoothed_2(ticker, date_start, date_end, frequency):
    try:

        fred = Fred(api_key='f40c3edb57e906557fcac819c8ab6478')
        date_start2 = "2009-01-01"
        # get data as an array and transforming it into a dataframe
        data_ = pd.DataFrame(
            fred.get_series(ticker, observation_start=date_start, observation_end=date_end, freq=frequency))
        data_2 = pd.DataFrame(
           fred.get_series(ticker, observation_start=date_start2, observation_end=date_end, freq=frequency))

        # creating 6m smoothing growth column
        data_['_6m_smoothing_growth'] = 100 * ((data_.iloc[:, 0][11:] / data_.iloc[:, 0].rolling(12).mean() - 1) * 2)[
                                              len(data_) - 19:]
        data_2['10 yr average'] = 100 * (data_2.iloc[:, 0].rolling(10).mean().pct_change())

        #fig, ax = plt.subplots(1, figsize=(11, 6))

        fig_ = go.Figure()
        # remove spines

        #ax.spines['right'].set_visible(False)
        #ax.spines['top'].set_visible(False)

        # plot characteristics
        # plt.figure(figsize=(11,7))

        # drop the blank values
        data_.dropna(inplace=True)

        # ploting the data
        print(data_.index)
        print(data_)
        fig_.add_trace(go.Scatter(x=data_.index.to_list(),y =data_._6m_smoothing_growth ,name="6m growth average"))
        fig_.add_trace(go.Scatter(x=(data_2.index.to_list())[len(data_2) - 19:],y =data_2['10 yr average'][len(data_2) - 19:],name="10yr average"))

        fig_.update_layout(
            title={
                'text': fred.get_series_info(ticker).title,
                'y': 0.9,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top'})

        # saving the figures in the static file
        #plt.savefig('/Users/talbi/Downloads/microblog/app/static/' + ticker + ".png")
        return fig_
    except ValueError:
        pass
if __name__ == "__main__":
    app.run_server(debug=True)



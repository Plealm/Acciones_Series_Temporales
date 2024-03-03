import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import yfinance as yf
import numpy as np
import datetime as dt
import warnings
import matplotlib.pyplot as plt

ticker_name = 'BC'

data = yf.download(ticker_name, start='2000-01-01', end='2020-01-01')
df = pd.DataFrame()
df["Date"] = pd.to_datetime(data.index).date
df["Close"] = data["Close"].values

fig = go.Figure()
# Suponiendo que ya tienes el DataFrame df

fig.add_trace(go.Scatter(
    x=df["Date"],
    y=df["Close"],
    name="NeuralProphet Forescasting",
    line=dict(color='blue', width=2)
))
fig.update_xaxes(type='category')  # Para que las fechas se muestren correctamente

# Centrar el título
fig.update_layout(title_text='Precio de acciones Bancolombia', title_x=0.5)
fig.update_layout(
    xaxis=dict(
        rangeselector=dict(
            buttons=list([
                dict(count=1,
                     label="1d",
                     step="day",
                     stepmode="backward"),
                dict(count=7,
                     label="1w",
                     step="day",
                     stepmode="backward"),
                dict(count=1,
                     label="1m",
                     step="month",
                     stepmode="backward"),
                dict(count=3,
                     label="3m",
                     step="month",
                     stepmode="backward"),
                dict(count=6,
                     label="6m",
                     step="month",
                     stepmode="backward"),
                dict(count=1,
                     label="1y",
                     step="year",
                     stepmode="backward"),
                dict(step="all")
            ])
        ),
        rangeslider=dict(
            visible=True
        ),
        type="date"
    ),
    yaxis=dict(
        autorange=True,
        type="linear"
    )
)

fig.update_layout(
    xaxis_title="Date",
    yaxis_title="Price",
    legend=dict(
        orientation="v",
        yanchor="top",
        y=1,
        xanchor="left",
        x=-.35
    )
)

dash.register_page(__name__, name="Importación de datos", path="/importacion")


df = px.data.tips()

layout = html.Div(
        [
        dcc.Markdown('''
        ## Importación de datos       
    '''
    , style={'text-align': 'center', 'margin-bottom': '20px'}
    ),
            
        dcc.Markdown('''
                     Los datos utilizados en este análisis fueron 
                     adquiridos mediante la herramienta **yfinance**, con el fin de obtener 
                     el precio de cierre de las acciones de Bancolombia.
                     ''', style={'text-align': 'center', 'margin-bottom': '20px', 'max-width': '800px', 'margin-left': 'auto', 'margin-right': 'auto'}),
        dcc.Graph(figure=fig)
    ]
)


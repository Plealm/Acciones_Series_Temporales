import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
from scipy.stats import boxcox
import yfinance as yf
from scipy import stats
from PIL import Image

ticker_name = 'BC'
data = yf.download(ticker_name, start='2000-01-01', end='2020-01-01')
df = pd.DataFrame()
df["Date"] = pd.to_datetime(data.index).date
df["Close"] = data["Close"].values

# Análisis Box-Cox
df['BoxCox_Close'], lambda_value = boxcox(df['Close'])

fig = go.Figure()
# Suponiendo que ya tienes el DataFrame df

fig.add_trace(go.Scatter(
    x=df["Date"],
    y=df["Close"],
    name="Original",
    line=dict(color='blue', width=2)
))
fig.add_trace(go.Scatter(
    x=df["Date"],
    y=df["BoxCox_Close"],
    name="Box-Cox",
    line=dict(color='green', width=2)
))
fig.update_xaxes(type='category')  # Para que las fechas se muestren correctamente

# Centrar el título
fig.update_layout(title=dict(
        text = 'Transformación Box-Cox ' # Ajustar el tamaño del título
    ) , title_x=0.75)
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
    ),
    margin=dict(l=0, r=20, t=100, b=50)
)



lambda_plot = Image.open("./img/lambda_box_cox.png")


dash.register_page(__name__, name="1. Estabilización de la varianza", path="/varianza")

df = px.data.tips()

layout = html.Div(
    [
        dcc.Markdown('''
        # 1. Estabilización de la varianza       
    '''
    , style={'text-align': 'center', 'margin-bottom': '20px'}),
        dcc.Markdown(
            '''
            En esta sección se buscará una herramienta para estabilizar la varianza, 

            ### Familia de transformaciones Box-Cox:

            En ocasiones la serie presenta varianza marginal no constante a lo largo del tiempo, lo cual hace necesario tener en cuenta tal característica. En este caso, se sugiere hacer una transformación de potencia para estabilizar la varianza. Esta familia de transformaciones se llaman transformaciones Box-Cox.

            $$
            f_{\\lambda}(u_{t})= \\begin{cases}
                \\lambda^{-1}(u^{\\lambda}_{t}-1), & \\text{si $u_{t} \\geq 0$, para $\\lambda>0$,}\\\\
                \\ln(u_{t}), & \\text{si $u_{t}>0$, para $\\lambda=0$}.
            \\end{cases}
            $$

            Note que la familia de funciones dependen del $\lambda$ escogido.
            ''',mathjax=True, style={'text-align': 'center', 'margin-bottom': '20px', 'max-width': '800px', 'margin-left': 'auto', 'margin-right': 'auto'},
            dangerously_allow_html=True
        ),
        html.Img(src=lambda_plot, style={ 'height': '100%','width': 'auto', 'margin-left': '350px'}),
        dcc.Graph(figure=fig)
    ]
)


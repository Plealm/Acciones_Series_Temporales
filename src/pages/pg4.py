import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy.ndimage import gaussian_filter1d
import statsmodels.api as sm
from plotly.subplots import make_subplots
import warnings
from statsmodels.nonparametric.smoothers_lowess import lowess
from scipy.interpolate import UnivariateSpline
from scipy.stats import boxcox
from statsmodels.tsa.seasonal import STL
import yfinance as yf
import matplotlib.pyplot as plt
import base64
from PIL import Image


warnings.filterwarnings("ignore")

# Importar datos
ticker_name = 'BC'
data = yf.download(ticker_name, start='2000-01-01', end='2020-01-01')
df = pd.DataFrame()
df["Date"] = pd.to_datetime(data.index).date
df["Close"] = data["Close"].values

# Análisis Box-Cox
df['BoxCox_Close'], lambda_value = boxcox(df['Close'])  # Agregar 1 para evitar problemas con valores no positivos



#---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------
#------------------------------- Lowess --------------------------------
#---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=df["Date"],
    y=df["BoxCox_Close"],
    name="Box-Cox",
    line=dict(color='green', width=2)
))

frac = 0.075  # Puedes seleccionar el valor de frac que desees
smoothed_values = lowess(df['BoxCox_Close'], df.index, frac=frac, it=0)

fig.add_trace(go.Scatter(
    x=df["Date"],
    y=smoothed_values[:, 1],
    name=f"Smoothed (frac={frac})",
    line=dict(width=2)
))

trend_removed = df['BoxCox_Close'] - smoothed_values[:, 1]

fig.add_trace(go.Scatter(
    x=df["Date"],
    y=trend_removed,
    name="Trend Removed",
    line=dict(width=2)
))

# Configuración del diseño del gráfico
fig.update_xaxes(type='category')  # Para que las fechas se muestren correctamente
fig.update_layout(
    title_text=f'Eliminación de tendencia Lowess con frac = {frac}',
    title_x=0.5,
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
    ),
    xaxis_title="Date",
    yaxis_title="Values",
    legend=dict(
        orientation="v",
        yanchor="top",
        y=1,
        xanchor="left",
        x=-.35
    )
)

#---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------
#------------------------------Diferencia Ordinaria ---------------------------------
#---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------

df['Close_diff'] = df['BoxCox_Close'].diff()

fig6 = go.Figure()

fig6.add_trace(go.Scatter(
    x=df["Date"],
    y=df["BoxCox_Close"],
    name="Box-Cox Original",
    line=dict(color='green', width=2)
))
fig6.add_trace(go.Scatter(
    x=df["Date"],
    y=df["Close_diff"],
    name="Diferencia Ordinaria",
    line=dict(color='orange', width=2)
))

fig6.update_xaxes(type='category')  # Para que las fechas se muestren correctamente

# Centrar el título
fig6.update_layout(title_text='Diferencia Ordinaria', title_x=0.5)
fig6.update_layout(
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

fig6.update_layout(
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


#---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------
#-------------------------- Diagrama de Retardos ---------------------------------
#---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------


lag_plot = Image.open("./img/lag1_plot.png")



dash.register_page(__name__, name="3. Eliminación de tendencia y analísis", path="/analisis")

df = px.data.tips()

layout = html.Div(
    [
        dcc.Markdown(
            ''' 
            # Eliminación de Tendencia
            
            ''',mathjax=True, style={'text-align': 'center', 'margin-bottom': '20px', 'max-width': '800px', 'margin-left': 'auto', 'margin-right': 'auto'},
            dangerously_allow_html=True
        ),
        dcc.Markdown(
            ''' 
            ## 3.1 Loess o Lowess

            Otro enfoque para suavizar un gráfico de tiempo es la regresión del vecino más cercano.
            La técnica se basa en la regresión de k vecinos más cercanos, en la que uno usa solo 
            los datos $\{x_{t−k/2}, ..., x_t, ..., x_{t+k/2}\}$ para predecir $x_t$ mediante regresión, y luego establece 
            $m_t = \hat{x}_t$. Primero, una cierta proporción de vecinos más cercanos a $x_t$ se incluyen en un esquema de 
            ponderación; los valores más cercanos a $x_t$ en el tiempo obtienen más peso. Luego, se utiliza una regresión ponderada 
            robusta para predecir $x_t$ y obtener los valores suavizados $m_t$. Cuanto mayor sea la fracción de vecinos más cercanos
            incluidos, más suave será el ajuste. 
            ''',mathjax=True, style={'text-align': 'center', 'margin-bottom': '20px', 'max-width': '800px', 'margin-left': 'auto', 'margin-right': 'auto'},
            dangerously_allow_html=True
        ),
         dcc.Graph(figure=fig),
         

         
        dcc.Markdown(
            ''' 
            ## 3.2 Diferencia ordinaria

            Apliquemos una diferencia ordinaria de orden 1 a la serie
            
            $$
            \\nabla^1 Y_t=(1-B)^1 Y_t=Y_t-Y_{t-1}
            $$
            ''',mathjax=True, style={'text-align': 'center', 'margin-bottom': '20px', 'max-width': '800px', 'margin-left': 'auto', 'margin-right': 'auto'},
            dangerously_allow_html=True
        ),
         dcc.Graph(figure=fig6),
         
         
        dcc.Markdown(
            ''' 
            ## 3.3 Diagramas de dispersión para los retardos

            Vamos a hacer gráficos de dispersión para chequear que tipos de relaciones hay entre los retardos 
            de la variable interés. Esto permite chequear si hay posibles relaciones no-lineales.
            
            ''',mathjax=True, style={'text-align': 'center', 'margin-bottom': '20px', 'max-width': '800px', 'margin-left': 'auto', 'margin-right': 'auto'},
            dangerously_allow_html=True
        ),
        html.Img(src=lag_plot, style={'width': '60%', 'height': 'auto', 'margin-left': '250px'}),
         
         
        
    ]
)


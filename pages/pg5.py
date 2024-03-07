import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy.ndimage import gaussian_filter1d
import statsmodels.api as sm
import calendar
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
frac = 0.075  # Puedes seleccionar el valor de frac que desees
smoothed_values = lowess(df['BoxCox_Close'], df.index, frac=frac, it=0)
trend_removed = df['BoxCox_Close'] - smoothed_values[:, 1]
df['Trend_Removed'] = trend_removed
df['Month_Name'] = df['Date'].dt.month_name()

# Crear el box plot utilizando seaborn


# Crear una nueva columna para el nombre del mes
df['Month_Name'] = df['Date'].dt.month_name()

# Crear el box plot utilizando plotly.express
fig = px.box(df, x='Month_Name', y='Trend_Removed', title='Box Plot de trend_removed por Mes', labels={'Trend_Removed': 'Trend Removed'})
fig.update_layout(title_text='Box Plot por Mes', title_x=0.5)


dash.register_page(__name__, name="4. Estimación de Ciclos y Estacionalidad", path="/estacionalidad")

df = px.data.tips()

layout = html.Div(
    [
        dcc.Markdown(
            ''' 
            ## 3.1 Serie Mensual
            
            ''',mathjax=True, style={'text-align': 'center', 'margin-bottom': '20px', 'max-width': '800px', 'margin-left': 'auto', 'margin-right': 'auto'},
            dangerously_allow_html=True
        ),
         dcc.Graph(figure=fig),
         
        dcc.Markdown(
            ''' 
            ## 3.1 Anual

            El periodograma es útil para identificar patrones de periodicidad o ciclos en una serie temporal. 
            Al observar el periodograma, puedes identificar picos o patrones distintivos que indican la 
            presencia de frecuencias dominantes en la serie temporal. Esto puede ser útil en la detección
            de ciclos estacionales, tendencias periódicas u otras estructuras de frecuencia en los datos.
            
            ''',mathjax=True, style={'text-align': 'center', 'margin-bottom': '20px', 'max-width': '800px', 'margin-left': 'auto', 'margin-right': 'auto'},
            dangerously_allow_html=True
        ),
         dcc.Graph(figure=fig),
         
        dcc.Markdown(
            ''' 
            ## 3.1 Periodograma

            El periodograma es útil para identificar patrones de periodicidad o ciclos en una serie temporal. 
            Al observar el periodograma, puedes identificar picos o patrones distintivos que indican la 
            presencia de frecuencias dominantes en la serie temporal. Esto puede ser útil en la detección
            de ciclos estacionales, tendencias periódicas u otras estructuras de frecuencia en los datos.
            
            ''',mathjax=True, style={'text-align': 'center', 'margin-bottom': '20px', 'max-width': '800px', 'margin-left': 'auto', 'margin-right': 'auto'},
            dangerously_allow_html=True
        ),
         dcc.Graph(figure=fig),
                  
         
        
    ]
)


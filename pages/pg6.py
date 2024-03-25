import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy.ndimage import gaussian_filter1d
import statsmodels.api as sm
from scipy.stats import gaussian_kde
from plotly.subplots import make_subplots
from statsmodels.nonparametric.smoothers_lowess import lowess
from scipy.interpolate import UnivariateSpline
from scipy.stats import boxcox
from statsmodels.tsa.seasonal import STL
import yfinance as yf
import matplotlib.pyplot as plt
import base64
from PIL import Image


# Importar datos
# Importar datos
ticker_name = 'BC'
data = yf.download(ticker_name, start='2000-01-01', end='2020-01-01')
df = pd.DataFrame()
df["Date"] = data.index
df["Close"] = data["Close"].values

# Análisis Box-Cox
df['BoxCox_Close'], lambda_value = boxcox(df['Close'])
frac = 0.075
smoothed_values = lowess(df['BoxCox_Close'], df.index, frac=frac, it=0)
trend_removed = df['BoxCox_Close'] - smoothed_values[:, 1]

df['Trend_Removed'] = trend_removed







dash.register_page(__name__, name="5. Modelado", path="/modelado")

df = px.data.tips()

layout = html.Div(
    [
        dcc.Markdown(
            ''' 
            # 5. Modelado           
            
            En esta sección se presentan los resultados obtenidos al modelar la serie de tiempo por suavisamiento exponencial y arboles de decisión.
            
            ''',mathjax=True, style={'text-align': 'center', 'margin-bottom': '20px', 'max-width': '800px', 'margin-left': 'auto', 'margin-right': 'auto'},
            dangerously_allow_html=True
        ),
        
         
                  
         
        
    ]
)


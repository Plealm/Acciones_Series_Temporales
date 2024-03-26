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

#---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------
#-------------------------- Box Plot Día-Semana ----------------------------------
#---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------


# Crear una nueva columna para el nombre del día de la semana
df['Day_Name'] = df['Date'].dt.day_name()

# Crear el box plot utilizando plotly.express
fig_dia_sem = px.box(df, x='Day_Name', y='Trend_Removed', title='Box Plot de trend_removed por Día de la Semana', labels={'Trend_Removed': 'Trend Removed'})
fig_dia_sem.update_layout(title_text='Box Plot por Día de la Semana', title_x=0.5)



#---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------
#-------------------------- Diagrama Día-Mes ----------------------------------
#---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------

# Calcular la mediana y el intervalo de confianza por día del mes

df['Day_Of_Month'] = df['Date'].dt.day
median_ci_by_day = df.groupby('Day_Of_Month')['Trend_Removed'].quantile([0.025, 0.5, 0.975]).unstack().reset_index()
median_ci_by_day.columns = ['Day_Of_Month', 'lower', 'median', 'upper']

# Crear el gráfico de la mediana y el intervalo de confianza
fig_median = go.Figure()
fig_median.add_trace(go.Scatter(x=median_ci_by_day['Day_Of_Month'], y=median_ci_by_day['median'], mode='lines', name='Median'))
fig_median.add_trace(go.Scatter(x=median_ci_by_day['Day_Of_Month'], y=median_ci_by_day['lower'], mode='lines', line=dict(dash='dash')))
fig_median.add_trace(go.Scatter(x=median_ci_by_day['Day_Of_Month'], y=median_ci_by_day['upper'], mode='lines', line=dict(dash='dash')))
fig_median.update_layout(
    title=dict(
        text='Mediana por Día del Mes',
        x=0.5
    ),
    xaxis_title='Día del Mes'
)

#---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------
#-------------------------- Diagrama Día-Año ----------------------------------
#---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------


df['Day_Of_Year'] = df['Date'].dt.dayofyear

# Calcular la mediana y el intervalo de confianza por día del año
median_year = df.groupby('Day_Of_Year')['Trend_Removed'].quantile([0.025, 0.5, 0.975]).unstack().reset_index()
median_year.columns = ['Day_Of_Year', 'lower', 'median', 'upper']

# Crear el gráfico de la mediana y el intervalo de confianza
fig_median_year = go.Figure()
fig_median_year.add_trace(go.Scatter(x=median_year['Day_Of_Year'], y=median_year['median'], mode='lines', name='Median'))
fig_median_year.add_trace(go.Scatter(x=median_year['Day_Of_Year'], y=median_year['lower'], mode='lines', line=dict(dash='dash')))
fig_median_year.add_trace(go.Scatter(x=median_year['Day_Of_Year'], y=median_year['upper'], mode='lines', line=dict(dash='dash')))
fig_median_year.update_layout(
    title=dict(
        text='Mediana por Día del Año',
        x=0.5
    ),
    xaxis_title='Día del Año'
)

#---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------
#-------------------------- Box Plot Mes-Año ----------------------------------
#---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------



df['Month_Name'] = df['Date'].dt.month_name()

# Crear el box plot utilizando plotly.express
fig_month = px.box(df, x='Month_Name', y='Trend_Removed', title='Box Plot de trend_removed por Mes', labels={'Trend_Removed': 'Trend Removed'})
fig_month.update_layout(title_text='Box Plot por Mes', title_x=0.5)



#---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------
#-------------------------- Periodograma -----------------------------------------
#---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------


perio = Image.open("./img/periodograma.png")





dash.register_page(__name__, name="4. Estimación de Ciclos y Estacionalidad", path="/estacionalidad")

df = px.data.tips()

layout = html.Div(
    [
        dcc.Markdown(
            ''' 
            # 4. Estacionalidad             
            
            Para el presente analisis se utilizará unicamente la serie de tiempo a la que se le removio la tendencia por **Lowess**.
            ''',mathjax=True, style={'text-align': 'center', 'margin-bottom': '20px', 'max-width': '800px', 'margin-left': 'auto', 'margin-right': 'auto'},
            dangerously_allow_html=True
        ),
        
        dcc.Markdown(
            ''' 
            ## 4.1 Box plot día de la semana
            
            Como se puede observar en la gráfica inferior, no se determina un cambio considerable 
            de valores, por lo que se duda la existencia de estacionalidad.
            
            ''',mathjax=True, style={'text-align': 'center', 'margin-bottom': '20px', 'max-width': '800px', 'margin-left': 'auto', 'margin-right': 'auto'},
            dangerously_allow_html=True
        ),
         dcc.Graph(figure=fig_dia_sem),
         
        
        dcc.Markdown(
            ''' 
            ## 4.2 Diagrama Día-Mes 
            
            Para calcular el intervalo de confianza para las gráficas de media, se utilizó un enfoque que implica calcular los cuantiles de la distribución
            de datos. Se empleó la función quantile de Pandas para calcular los cuantiles específicos que definen el intervalo de confianza. Estos cuantiles 
            incluyen el límite inferior del intervalo de confianza (por lo general, el cuantil 2.5%), la mediana (cuantil 50%) y el límite superior del 
            intervalo de confianza (por lo general, el cuantil 97.5%).
            
            De igual manera se observa que el cambio de la gráfica es muy pequeño, por lo que se duda la existencia de estacionalidad.
            
            ''',mathjax=True, style={'text-align': 'center', 'margin-bottom': '20px', 'max-width': '800px', 'margin-left': 'auto', 'margin-right': 'auto'},
            dangerously_allow_html=True
        ),
         dcc.Graph(figure=fig_median),
         
        
        dcc.Markdown(
            ''' 
            ## 4.3 Diagrama Día-Año
            
            Aunque se observa mayor variabilidad en la mediana, observe que su valor maximo es de $\\approx 1.1$ y su valor minimo es de $\\approx -1.0$. 
            Lo que no representa un cambio significativo en la serie de tiempo, por lo que se duda la existencia de estacionalidad.
            
            ''',mathjax=True, style={'text-align': 'center', 'margin-bottom': '20px', 'max-width': '800px', 'margin-left': 'auto', 'margin-right': 'auto'},
            dangerously_allow_html=True
        ),
         dcc.Graph(figure=fig_median_year),
         
        dcc.Markdown(
            ''' 
            ## 4.3 Box-plot Mes-Año
            
            Por ultimo, se intenta ver algun cambio significativo en la mediana tomando los valores de cada mes del año,
            sin embargo, no se observa un cambio significativo en la mediana, por lo que con las consideraciones presentadas anteriormente
            se puede esperar que **no** exista estacionalidad en la serie de tiempo.
            
            ''',mathjax=True, style={'text-align': 'center', 'margin-bottom': '20px', 'max-width': '800px', 'margin-left': 'auto', 'margin-right': 'auto'},
            dangerously_allow_html=True
        ),
         dcc.Graph(figure=fig_month),
         
         
        dcc.Markdown(
            ''' 
            ## 4.5 Periodograma

            El periodograma es útil para identificar patrones de periodicidad o ciclos en una serie temporal. 
            Al observar el periodograma, puedes identificar picos o patrones distintivos que indican la 
            presencia de frecuencias dominantes en la serie temporal. Esto puede ser útil en la detección
            de ciclos estacionales, tendencias periódicas u otras estructuras de frecuencia en los datos.
            
            ''',mathjax=True, style={'text-align': 'center', 'margin-bottom': '20px', 'max-width': '800px', 'margin-left': 'auto', 'margin-right': 'auto'},
            dangerously_allow_html=True
        ),html.Img(src=perio,  style={'display': 'block', 'margin': 'auto', 'width': '60%', 'height': 'auto'}),
        
        
         
                  
         
        
    ]
)


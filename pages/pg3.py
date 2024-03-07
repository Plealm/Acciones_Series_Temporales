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

# Agregar columnas de promedios móviles (SMA) con ventanas de 10, 50, 100, 200, 500 y 1000 días
df['SMA_10'] = df['BoxCox_Close'].rolling(window=10).mean()
df['SMA_50'] = df['BoxCox_Close'].rolling(window=50).mean()
df['SMA_100'] = df['BoxCox_Close'].rolling(window=100).mean()
df['SMA_200'] = df['BoxCox_Close'].rolling(window=200).mean()
df['SMA_500'] = df['BoxCox_Close'].rolling(window=500).mean()
df['SMA_1000'] = df['BoxCox_Close'].rolling(window=1000).mean()




# Crear gráfico
fig = go.Figure()

# Suponiendo que ya tienes el DataFrame df

fig.add_trace(go.Scatter(
    x=df["Date"],
    y=df["BoxCox_Close"],
    name="Box-Cox",
    line=dict(color='green', width=2)
))

fig.add_trace(go.Scatter(
    x=df["Date"],
    y=df["SMA_10"],
    name="SMA (10 days)",
    line=dict(color='orange', width=2)
))

fig.add_trace(go.Scatter(
    x=df["Date"],
    y=df["SMA_50"],
    name="SMA (50 days)",
    line=dict(color='red', width=2)
))

fig.add_trace(go.Scatter(
    x=df["Date"],
    y=df["SMA_100"],
    name="SMA (100 days)",
    line=dict(color='purple', width=2)
))

fig.add_trace(go.Scatter(
    x=df["Date"],
    y=df["SMA_200"],
    name="SMA (200 days)",
    line=dict(color='blue', width=2)
))

fig.add_trace(go.Scatter(
    x=df["Date"],
    y=df["SMA_500"],
    name="SMA (500 days)",
    line=dict(color='yellow', width=2)
))

fig.add_trace(go.Scatter(
    x=df["Date"],
    y=df["SMA_1000"],
    name="SMA (1000 days)",
    line=dict(color='cyan', width=2)
))

fig.update_xaxes(type='category')  # Para que las fechas se muestren correctamente

# Centrar el título
fig.update_layout(title_text='Precio de acciones Bancolombia con SMAs', title_x=0.5)
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
#---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------
#------------------------------- Kernel Smoothing --------------------------------
#---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------




df['Smoothed_BoxCox_1'] = gaussian_filter1d(df['BoxCox_Close'], sigma=1)
df['Smoothed_BoxCox_10'] = gaussian_filter1d(df['BoxCox_Close'], sigma=10)
df['Smoothed_BoxCox_100'] = gaussian_filter1d(df['BoxCox_Close'], sigma=100)
df['Smoothed_BoxCox_200'] = gaussian_filter1d(df['BoxCox_Close'], sigma=200)

# Crear gráfico
fig1 = go.Figure()

# Suponiendo que ya tienes el DataFrame df

fig1.add_trace(go.Scatter(
    x=df["Date"],
    y=df["BoxCox_Close"],
    name="Box-Cox",
    line=dict(color='green', width=2)
))

fig1.add_trace(go.Scatter(
    x=df["Date"],
    y=df["Smoothed_BoxCox_1"],
    name="Kernel (σ=1)",
    line=dict(color='orange', width=2)
))
fig1.add_trace(go.Scatter(
    x=df["Date"],
    y=df["Smoothed_BoxCox_10"],
    name="Kernel (σ=10)",
    line=dict(color='red', width=2)
))
fig1.add_trace(go.Scatter(
    x=df["Date"],
    y=df["Smoothed_BoxCox_100"],
    name="Kernel (σ=100)",
    line=dict(color='purple', width=2)
))
fig1.add_trace(go.Scatter(
    x=df["Date"],
    y=df["Smoothed_BoxCox_200"],
    name="Kernel (σ=200)",
    line=dict(color='blue', width=2)
))


fig1.update_xaxes(type='category')  # Para que las fechas se muestren correctamente

# Centrar el título
fig1.update_layout(title_text='Precio de acciones Bancolombia con Suavisamiento Kernel Gaussiano', title_x=0.5)
fig1.update_layout(
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

fig1.update_layout(
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
#------------------------------- Lowess --------------------------------
#---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------

# Crear gráfico con diferentes valores de frac
fig2 = go.Figure()

# Suponiendo que ya tienes el DataFrame df

fig2.add_trace(go.Scatter(
    x=df["Date"],
    y=df["BoxCox_Close"],
    name="Box-Cox",
    line=dict(color='green', width=2)
))

frac_values = [0.05, 0.1, 0.2, 0.3]  # Diferentes valores de frac

for frac in frac_values:
    smoothed_values = lowess(df['BoxCox_Close'], df.index, frac=frac, it=0)
    fig2.add_trace(go.Scatter(
        x=df["Date"],
        y=smoothed_values[:, 1],
        name=f"Smoothed (frac={frac})",
        line=dict(width=2)
    ))

fig2.update_xaxes(type='category')  # Para que las fechas se muestren correctamente

# Centrar el título
fig2.update_layout(title_text='Precio de acciones Bancolombia con Suavizado lowess para diferentes frac', title_x=0.5)
fig2.update_layout(
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

fig2.update_layout(
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
#------------------------------- Spline --------------------------------
#---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------





fig3 = go.Figure()

fig3.add_trace(go.Scatter(
    x=df["Date"],
    y=df["BoxCox_Close"],
    name="Box-Cox Original",
    line=dict(color='green', width=2)
))

fig3.update_xaxes(type='category')  # Para que las fechas se muestren correctamente

# Centrar el título
fig3.update_layout(title_text='Precio de acciones Bancolombia - Box-Cox Original', title_x=0.5)
fig3.update_layout(
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

fig3.update_layout(
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

# Aplicar suavizado con splines para diferentes valores de k
k_values = [1, 2, 3, 4, 5]
colors = ['blue', 'orange', 'red', 'purple', 'cyan']

for k, color in zip(k_values, colors):
    spl = UnivariateSpline(df.index, df['BoxCox_Close'], k=k)
    df[f'Spline_Close_{k}'] = spl(df.index)

    fig3.add_trace(go.Scatter(
        x=df["Date"],
        y=df[f"Spline_Close_{k}"],
        name=f"Spline k={k}",
        line=dict(color=color, width=2)
    ))


#---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------
#---------------------------- Descomposición Promedios Moviles--------------------
#---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------


result = sm.tsa.seasonal_decompose(df['BoxCox_Close'], period=252)  # 252 días en un año bursátil

# Crear una figura con subgráficas
fig4 = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.05)

# Subgráfica 1: Serie original
fig4.add_trace(go.Scatter(
    x=df["Date"],
    y=df["BoxCox_Close"],
    name="Box-Cox Original",
    line=dict(color='green', width=2)
), row=1, col=1)

# Subgráfica 2: Tendencia
fig4.add_trace(go.Scatter(
    x=df["Date"],
    y=result.trend,
    name="Tendencia",
    line=dict(color='blue', width=2)
), row=2, col=1)

# Subgráfica 3: Estacionalidad
fig4.add_trace(go.Scatter(
    x=df["Date"],
    y=result.seasonal,
    name="Estacionalidad",
    line=dict(color='red', width=2)
), row=3, col=1)

# Subgráfica 4: Residuos
fig4.add_trace(go.Scatter(
    x=df["Date"],
    y=result.resid,
    name="Residuos",
    line=dict(color='purple', width=2)
), row=4, col=1)

# Ajustes de diseño
fig4.update_layout(title_text='Descomposición de acciones Bancolombia por Promedios Moviles', title_x=0.5)

fig4.update_layout(
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
        type="date"
    ),
    yaxis=dict(
        autorange=True,
        type="linear"
    )
)

# Configurar un rango selector único para todos los subgráficos
fig4.update_xaxes(rangeslider=dict(visible=True), row=4, col=1)



#---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------
#---------------------------- Descomposición STL ---------------------------------
#---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------
# Descomposición usando STL
stl = STL(df['BoxCox_Close'], seasonal=13, period=252)
result = stl.fit()

# Crear una figura con subgráficas
fig5 = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.05)

# Subgráfica 1: Serie original
fig5.add_trace(go.Scatter(
    x=df["Date"],
    y=df["Close"],
    name="Original",
    line=dict(color='green', width=2)
), row=1, col=1)

# Subgráfica 2: Tendencia
fig5.add_trace(go.Scatter(
    x=df["Date"],
    y=result.trend,
    name="Tendencia",
    line=dict(color='blue', width=2)
), row=2, col=1)

# Subgráfica 3: Estacionalidad
fig5.add_trace(go.Scatter(
    x=df["Date"],
    y=result.seasonal,
    name="Estacionalidad",
    line=dict(color='red', width=2)
), row=3, col=1)

# Subgráfica 4: Residuos
fig5.add_trace(go.Scatter(
    x=df["Date"],
    y=result.resid,
    name="Residuos",
    line=dict(color='purple', width=2)
), row=4, col=1)

# Ajustes de diseño
fig5.update_layout(title_text='Descomposición de acciones Bancolombia con STL', title_x=0.5)

# Configurar un rango selector único para todos los subgráficos
fig5.update_xaxes(rangeslider=dict(visible=True), row=4, col=1)


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
fig6.update_layout(title_text='Precio de acciones Bancolombia - Box-Cox Original', title_x=0.5)
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




dash.register_page(__name__, name="2. Estimación de la tendencia", path="/Tendencia")

df = px.data.tips()

layout = html.Div(
    [
        dcc.Markdown('''
        ## 2. Análisis de la tendencia y eliminación      
    '''
    , style={'text-align': 'center', 'margin-bottom': '20px'}),
        dcc.Markdown(
            '''
            Los procedimientos que permiten estimar y extraer los componentes de tendencia y/o estacionalidad se conoce como **suavizamiento**.
            
            ### 2.1. Promedio móvil :
            
            El promedio móvil es un método útil para descubrir ciertos rasgos en una serie de tiempo, como **tendencias a largo plazo y componentes estacionales**. En particular, si $x_t$ representa las observaciones, entonces una forma de predecir o estimar la tendencia de la serie es:
            
            $$m_t=\\sum_{j=-k}^{k}a_jx_{t-j},$$ 
            
            donde si $a_j=a_{-j}\\geq 0$ y $\\sum_{j=-k}^{k}a_j=1$ se conoce como el promedio móvil simétrico de los datos.
            ''',mathjax=True, style={'text-align': 'center', 'margin-bottom': '20px', 'max-width': '800px', 'margin-left': 'auto', 'margin-right': 'auto'},
            dangerously_allow_html=True
        ),
        dcc.Graph(figure=fig),
        dcc.Markdown(
            ''' 
            ## 2.2 Suavisamiento Kernel

            El suavizamiento kernel es un suavizador de promedio móvil que utiliza una función de ponderación, o kernel, para promediar las observaciones.
            Veamos ahora como queda el promedio móvil:
            
            $$m_t= \\sum_{i=1}^n \\omega_i(t)x_i$$
            
            donde, 
            
            $$w_i(t)=K(\\frac{t-i}{b})/\\sum_{j=1}^{n}K(\\frac{t-j}{b})$$
            
            son los pesos y $K(\\cdot)$ es una función kernel.
            ''',mathjax=True, style={'text-align': 'center', 'margin-bottom': '20px', 'max-width': '800px', 'margin-left': 'auto', 'margin-right': 'auto'},
            dangerously_allow_html=True
        ),
         dcc.Graph(figure=fig1),
         
        dcc.Markdown(
            ''' 
            ## 2.3 Loess o Lowess

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
         dcc.Graph(figure=fig2),
         
         
        dcc.Markdown(
            ''' 
            ##  2.4 Suavizamiento Splines

            Una forma obvia de suavizar los datos sería ajustar una regresión polinomial en términos del tiempo. Por ejemplo,
            un polinomio cúbico tendría $x_t = m_t + w_t$ donde $m_t =\\beta_0 + \\beta_1t + \\beta_2t^2 + \\beta_3t^3$.
            Entonces podríamos ajustar $m_t$ mediante mínimos cuadrados ordinarios.
            
            Una extensión de la regresión polinomial es dividir primero el tiempo $t = 1,. . . , n$, en 
            k intervalos, $[t_0 = 1, t_1]$, $[t_1 + 1, t_2],\\cdots,$ $[t_{k − 1} + 1, t_k = n]$; los valores $t_0$, $t_1$, …, $t_k$ 
            se llaman nodos. Luego, en cada intervalo, se ajusta una regresión polinomial, normalmente de orden 3, y esto se 
            llama splines cúbicos. Un método relacionado es suavizar splines, que minimiza el compromiso entre el ajuste y
            el grado de suavidad dado por
            
            $$\\sum_{t=1}^{n}[x_t-m_t]^2+\\lambda\\int(m_t'')^2 dt,$$
            
            donde $m_t$ es un spline cúbico con nodos en cada tiempo t y el grado de suavidad es controlado por $\\lambda>0.$ 
            ''',mathjax=True, style={'text-align': 'center', 'margin-bottom': '20px', 'max-width': '800px', 'margin-left': 'auto', 'margin-right': 'auto'},
            dangerously_allow_html=True
        ),
         dcc.Graph(figure=fig3),
         
         
        dcc.Markdown(
            ''' 
            ##  2.6 descomposición por Promedios Moviles
            ''',mathjax=True, style={'text-align': 'center', 'margin-bottom': '20px', 'max-width': '800px', 'margin-left': 'auto', 'margin-right': 'auto'},
            dangerously_allow_html=True
        ),
         dcc.Graph(figure=fig4),
         
         
        dcc.Markdown(
            ''' 
            ##  2.7  Descomposición STL

            STL son las iniciales de "Seasonal and Trend decomposition using Loess",el cual fue
            desarrollado por R. B. Cleveland et al. (1990).

            Note que se obliga a extraer un componente estacional, 
            sin embargo puede que está componente en verdad no exista, 
            por eso se debe verificar que en efecto hay.
                
            ''',mathjax=True, style={'text-align': 'center', 'margin-bottom': '20px', 'max-width': '800px', 'margin-left': 'auto', 'margin-right': 'auto'},
            dangerously_allow_html=True
        ),
         dcc.Graph(figure=fig5),
         
                  
         
         
        
    ]
)


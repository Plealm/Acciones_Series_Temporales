import dash
from dash import dcc, html
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import yfinance as yf
import matplotlib.pyplot as plt
from PIL import Image
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, Holt


# Descargar datos
ticker_name = 'BC'
data = yf.download(ticker_name, start='2000-01-01', end='2020-01-01')

# Crear DataFrame
df = pd.DataFrame()
df["Date"] = data.index
df["Close"] = data["Close"].values

# Dividir en conjunto de entrenamiento y prueba
train_size = int(len(df) * 0.85)  # 80% para entrenamiento
train, test = df[:train_size], df[train_size:]


#---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------
#-------------------------- Simple Exponential Smoothing -------------------------
#---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------



# Definir diferentes valores de suavizamiento (alpha)
alphas = [0.2, 0.5, 0.8]

# Predicciones para datos de entrenamiento y prueba con diferentes valores de suavizamiento
preds_train = {}
preds_test = {}

for alpha in alphas:
    # Modelo de suavizado exponencial simple
    model = SimpleExpSmoothing(train["Close"]).fit(smoothing_level=alpha, optimized=False)
    
    # Predicciones para datos de entrenamiento
    preds_train[alpha] = model.fittedvalues
    
    # Predicciones para datos de prueba
    preds_test[alpha] = model.forecast(len(test))

# Visualizar las predicciones y los datos reales con Plotly
fig_simple = go.Figure()

# Datos de entrenamiento
for alpha, preds in preds_train.items():
    fig_simple.add_trace(go.Scatter(x=train["Date"], y=preds, mode='lines', name=f'Train (alpha={alpha})'))

# Datos de prueba
for alpha, preds in preds_test.items():
    fig_simple.add_trace(go.Scatter(x=test["Date"], y=preds, mode='lines', name=f'Test (alpha={alpha})'))

fig_simple.add_trace(go.Scatter(x=train["Date"], y=train["Close"], mode='lines', name='Train', line=dict(color='black')))
fig_simple.add_trace(go.Scatter(x=test["Date"], y=test["Close"], mode='lines', name='Test', line=dict(color='red')))

fig_simple.update_layout(title="Predicciones con Suavizado Exponencial Simple (diferentes alphas)",
                  xaxis_title="Fecha",
                  yaxis_title="Precio de cierre")
fig_simple.update_layout(
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
                dict(count=2,
                     label="2y",
                     step="year",
                     stepmode="backward"),
                dict(count=3,
                     label="3y",
                     step="year",
                     stepmode="backward"),
                dict(count=5,
                     label="5y",
                     step="year",
                     stepmode="backward"),
                dict(count=10,
                     label="10y",
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

#---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------
#--------------------------------- Holt ------------------------------------------
#---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------


fit1 = Holt(train["Close"], initialization_method="estimated").fit(smoothing_level=0.8, smoothing_trend=0.2, optimized=False)
fcast1 = fit1.forecast(len(test) + 5).rename("Holt's linear trend")

fit2 = Holt(train["Close"], exponential=True, initialization_method="estimated").fit(smoothing_level=0.8, smoothing_trend=0.2, optimized=False)
fcast2 = fit2.forecast(len(test) + 5).rename("Exponential trend")

fit3 = Holt(train["Close"], damped_trend=True, initialization_method="estimated").fit(smoothing_level=0.8, smoothing_trend=0.2)
fcast3 = fit3.forecast(len(test) + 5).rename("Additive damped trend")

# Visualizar las predicciones y los datos reales con Plotly
fig_Holt = go.Figure()

# Datos de entrenamiento
fig_Holt.add_trace(go.Scatter(x=train["Date"], y=fit1.fittedvalues, mode='lines', name="Holt's linear trend"))
fig_Holt.add_trace(go.Scatter(x=train["Date"], y=fit2.fittedvalues, mode='lines', name="Exponential trend"))
fig_Holt.add_trace(go.Scatter(x=train["Date"], y=fit3.fittedvalues, mode='lines', name="Additive damped trend"))

# Predicciones a partir de la fecha de prueba
test_dates = pd.date_range(start=test["Date"].iloc[0], periods=len(test) + 5)

for i, (fcast, method) in enumerate(zip([fcast1, fcast2, fcast3], ["Holt's linear trend", "Exponential trend", "Additive damped trend"]), start=1):
    fig_Holt.add_trace(go.Scatter(x=test_dates, y=fcast, mode='lines', name=f"Forecast {i} ({method})"))

fig_Holt.add_trace(go.Scatter(x=train["Date"], y=train["Close"], mode='lines', name='Train', line=dict(color='black')))
fig_Holt.add_trace(go.Scatter(x=test["Date"], y=test["Close"], mode='lines', name='Test', line=dict(color='red')))

fig_Holt.update_layout(title="Predicciones con Holt (diferentes configuraciones)",
                  xaxis_title="Fecha",
                  yaxis_title="Precio de cierre")
fig_Holt.update_layout(
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
                dict(count=2,
                     label="2y",
                     step="year",
                     stepmode="backward"),
                dict(count=3,
                     label="3y",
                     step="year",
                     stepmode="backward"),
                dict(count=5,
                     label="5y",
                     step="year",
                     stepmode="backward"),
                dict(count=10,
                     label="10y",
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



#---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------
#--------------------------------- Tabla Final -----------------------------------
#---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------


# Ajustar modelos
fit1 = SimpleExpSmoothing(df["Close"], initialization_method="estimated").fit()
fit2 = Holt(df["Close"], initialization_method="estimated").fit()
fit3 = Holt(df["Close"], exponential=True, initialization_method="estimated").fit()
fit4 = Holt(df["Close"], damped_trend=True, initialization_method="estimated", ).fit(damping_trend=0.98)
fit5 = Holt(df["Close"], exponential=True, damped_trend=True, initialization_method="estimated").fit()


# Crear la tabla de resultados
results = pd.DataFrame(
    index=[r"$\alpha$", r"$\beta$", r"$\phi$", r"$l_0$", "$b_0$", "MSE"],
    columns=["SES", "Holt's", "Exponential", "Additive", "Multiplicative"],
)
params = ['smoothing_level', 'smoothing_trend', 'damping_trend', 'initial_level', 'initial_trend']

results["SES"] =            [round(fit1.params[p], 3) for p in params] + [round(fit1.sse, 3)]
results["Holt's"] =         [round(fit2.params[p], 3) for p in params] + [round(fit2.sse, 3)]
results["Exponential"] =    [round(fit3.params[p], 3) for p in params] + [round(fit3.sse, 3)]
results["Additive"] =       [round(fit4.params[p], 3) for p in params] + [round(fit4.sse, 3)]
results["Multiplicative"] = [round(fit5.params[p], 3) for p in params] + [round(fit5.sse, 3)]



#---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------
#--------------------------------- Árbol de Decisión -----------------------------
#---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------


"""
Creación de variables rezagadas
"""

df1 = pd.DataFrame()
for i in range(10, 0, -1):
    shifted_values = df['Close'].shift(i)
    df1['t-' + str(i)] = shifted_values  # Selecciona solo las filas sin NaN

# Ajustar el índice a las filas seleccionadas
df1.index = df['Date']
df1['t'] = df['Close'].values
df1 = df1.dropna()

print(df1.head(5))

"""
Dividir los datos
"""

split = df1.values
# split into lagged variables and original time series
X1= split[:, 0:-1]  # slice all rows and start with column 0 and go up to but not including the last column
Y1 =split[:,-1] 

traintarget_size = int(len(Y1) * 0.70) 
valtarget_size = int(len(Y1) * 0.10)+1# Set split
testtarget_size = int(len(Y1) * 0.20)# Set split
train_target, val_target,test_target = Y1[0:traintarget_size],Y1[(traintarget_size):(traintarget_size+valtarget_size)] ,Y1[(traintarget_size+valtarget_size):len(Y1)]


trainfeature_size = int(len(X1) * 0.70)
valfeature_size = int(len(X1) * 0.10)+1# Set split
testfeature_size = int(len(X1) * 0.20)# Set split
train_feature, val_feature,test_feature = X1[0:traintarget_size],X1[(traintarget_size):(traintarget_size+valtarget_size)] ,X1[(traintarget_size+valtarget_size):len(Y1)]







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
        
        dcc.Markdown(
            ''' 
            ## 5.1. Suavisamiento Exponencial         
            
            En esta sección se presentan los resultados obtenidos al modelar la serie de tiempo por suavisamiento exponencial y arboles de decisión.
            
            ''',mathjax=True, style={'text-align': 'center', 'margin-bottom': '20px', 'max-width': '800px', 'margin-left': 'auto', 'margin-right': 'auto'},
            dangerously_allow_html=True
        ),
        
        dcc.Markdown(
            ''' 
            ### 5.1.1 Simple Exponential Smoothing (Suavizado Exponencial Simple)

            El código proporcionado implementa el método de Suavizado Exponencial Simple (SES) para predecir el precio de cierre de una acción. El SES es un método
            de pronóstico de series temporales que asigna pesos exponenciales decrecientes a las observaciones pasadas. Esto significa que las observaciones más recientes 
            tienen más influencia en la predicción que las observaciones más antiguas. El modelo SES se puede describir con la siguiente ecuación de actualización y 
            la ecuación de predicción:
            
            **Ecuación de actualización:**  
            $[ \\hat{y}_{t+1} = \\alpha \\cdot y_t + (1 - \\alpha) \\cdot \\hat{y}_t ]$            
            **Ecuación de predicción:**  
            $[ \\hat{y}_{t+h|t} = \\alpha \\cdot y_t + (1 - \\alpha)^2 \\cdot y_{t-1} + \\ldots + (1 - \\alpha)^h \\cdot \\hat{y}_t ]$
            
            donde:  
            $( \\hat{y}_{t+1})$ es la predicción para el siguiente período.
            $( \\hat{y}_t)$ es la predicción actual.
            $( y_t)$ es la observación actual.
            $( \\alpha)$ es el factor de suavizado, que controla la influencia de las observaciones pasadas en la predicción.
            $( h)$ es el número de períodos en el futuro para los que se está haciendo la predicción.
            
            El código ajusta el modelo SES para diferentes valores de $( \\alpha )$ y realiza predicciones tanto para los datos de entrenamiento como para los datos de prueba. Luego, visualiza las predicciones junto con los datos reales en un gráfico interactivo utilizando Plotly.

            ''',mathjax=True, style={'text-align': 'center', 'margin-bottom': '20px', 'max-width': '800px', 'margin-left': 'auto', 'margin-right': 'auto'},
            dangerously_allow_html=True
        ),
        dcc.Graph(figure=fig_simple),
        
        dcc.Markdown(
            ''' 
            ### 5.1.2 Suavisado exponencial Doble (Holt)
            
            El suavizado exponencial doble, también conocido como método de Holt, es una técnica de pronóstico utilizada para series de tiempo que exhiben tanto tendencia como estacionalidad. Fue propuesto por Charles C. Holt en 1957.

            **Descripción**
            El método de Holt extiende el suavizado exponencial simple (SES) al introducir un término adicional para modelar la tendencia. A diferencia de SES, que solo utiliza un parámetro de suavizado para la nivelación, Holt utiliza dos parámetros: uno para la nivelación y otro para la tendencia. Esto permite capturar tanto la pendiente como el nivel cambiante de la serie temporal.
            
            La ecuación de nivel y la ecuación de tendencia en el método de Holt son las siguientes:
            
            **Ecuación de nivel**:
            
            $$L_t = \\alpha y_t + (1 - \\alpha) (L_{t-1} +T_{t-1})$$
            
            $L_t$ es el nivel estimado en el tiempo $t$.
            $y_t$ es la observación actual.
            $L_{t-1}$ es el nivel estimado en el tiempo anterior.
            $T_{t-1}$ es la tendencia estimada en el tiempo anterior.
            
            **Ecuación de Tendencia**:
            
            $$T_t = \\beta (L_t - L_{t-1}) + (1 - \\beta) T_{t-1}$$
            
            $T_t$ es la tendencia estimada en el tiempo $t$.
            $\\beta$ es el factor de suavizado para la tendencia.
            $L_t$ es el nivel estimado en el tiempo $t$.
            $L_{t-1}$ es el nivel estimado en el tiempo anterior.
            
            El código proporcionado ajusta el modelo de Holt para diferentes configuraciones y 
            realiza predicciones tanto para los datos de entrenamiento como para los datos de prueba. Luego, 
            visualiza las predicciones junto con los datos reales en un gráfico interactivo utilizando Plotly.
            
            
            
            ''',mathjax=True, style={'text-align': 'center', 'margin-bottom': '20px', 'max-width': '800px', 'margin-left': 'auto', 'margin-right': 'auto'},
            dangerously_allow_html=True
        ),
        dcc.Graph(figure=fig_Holt),
         
           dcc.Markdown(
            ''' 
            ### 5.1.3 Comparación de modelos por SSE
            

            En este análisis, se compararon varios modelos de suavizado exponencial y Holt para ajustarse a una serie temporal de datos. A continuación se describen brevemente cada uno de los modelos utilizados:
            
            1.**Suavizado Exponencial Simple (SES):**
            
            - Este modelo aplica un solo parámetro de suavizado a la serie temporal.
            - Se ajusta a la serie temporal original sin considerar tendencias ni estacionalidad.
            - Útil cuando la serie es relativamente estable sin tendencias ni patrones estacionales discernibles.
            
            2.**Holt's Linear Trend:**
            
            - Extiende el SES al introducir un término adicional para modelar la tendencia lineal.
            - Adecuado para series temporales con una tendencia lineal clara.
            - No considera efectos de estacionalidad.
            
            3.**Holt's Exponential Trend:**
            
            - Similar al modelo anterior, pero la tendencia se modela de manera exponencial en lugar de lineal.
            - Útil cuando la serie muestra una tendencia que está creciendo o disminuyendo exponencialmente.
            
            4.**Holt's Additive Damped Trend:**
            
            - Introduce un término de amortiguación para reducir gradualmente el efecto de la tendencia con el tiempo.
            - Útil cuando se espera que la tendencia cambie de forma gradual en lugar de persistir de manera lineal o exponencial.
            
            5.**Holt's Exponential Trend with Damped Trend:**
            
            - Combina una tendencia exponencial con un término de amortiguación.
            - Útil cuando se espera una tendencia exponencial con un cambio gradual en el tiempo.
            
            **Conclusiones**
            
            Los modelos Holt's Exponential Trend y Holt's Exponential Trend with Damped Trend muestran un MSE ligeramente más bajo en comparación con los otros modelos.
            
            ''',mathjax=True, style={'text-align': 'center', 'margin-bottom': '20px', 'max-width': '800px', 'margin-left': 'auto', 'margin-right': 'auto'},
            dangerously_allow_html=True
        ),       
         
        html.Div(
        [
            # HTML Table
            html.Table(
                [html.Tr([html.Th("")] + [html.Th(col) for col in results.columns], style={'background-color': 'lightblue'})] +
                [html.Tr([html.Td(dcc.Markdown(results.index[i], mathjax=True))] + 
                         [html.Td(dcc.Markdown(str(results.iloc[i][col]), mathjax=True), style={'text-align': 'center'}) for col in results.columns]) for i in range(len(results))],
                style={'margin': '0 auto', 'max-width': '800px', 'text-align': 'center', 'border-spacing': '10px'}
            )
        ],
        style={'text-align': 'center'}
    )
])
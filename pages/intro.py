import dash
from dash import dcc, html
import plotly.express as px

dash.register_page(__name__, name='Introducción', path='/')

layout = html.Div([
    dcc.Markdown('''
        ## Introducción       
    '''
    , style={'text-align': 'center', 'margin-bottom': '60px'}
    ),

    dcc.Markdown('''
        Este análisis se centra en el **precio de cierre** vs **tiempo** de las acciones de Bancolombia durante el periodo comprendido entre el 2020-01-01 y el 2020-01-01. La elección de este periodo se fundamenta en evitar la inclusión de datos durante o después de la pandemia. Dada la naturaleza de la serie temporal de acciones, se anticipa un comportamiento de tipo caminata aleatoria. Además, es importante destacar que esta serie temporal presenta datos no uniformemente distribuidos en el tiempo, ya que los sábados, domingos y días festivos no hay actividad en la bolsa.

    '''
    ,  style={'text-align': 'center', 'margin-bottom': '70px', 'max-width': '800px', 'margin-left': 'auto', 'margin-right': 'auto'}
    ),

    dcc.Markdown('''
        **Autores:** Pedro Leal, Luis Mantilla, Bryam Bustos.
    '''
    , style={'text-align': 'center', 'margin-bottom': '20px'}
    ),
])







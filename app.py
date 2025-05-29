import dash
from dash import dcc, html, dash_table, Output, Input
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
import joblib
import os

# Cargar datos y modelos precomputados
df = pd.read_pickle('data_subset.pkl')
model_results = joblib.load('model_results.pkl')
le = joblib.load('label_encoder.pkl')

# Dividir datos como en el script de precomputación para obtener y_val
X_pre = df.drop(columns=["income"])
y_pre = df["income"]
X_train, X_temp, y_train, y_temp = train_test_split(X_pre, y_pre, test_size=0.30, stratify=y_pre, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42)

# Encontrar el mejor modelo
best_model_name = max(model_results.keys(), key=lambda x: model_results[x]['f1_score'])
best_results = model_results[best_model_name]

# Configurar la app Dash
app = dash.Dash(__name__, external_stylesheets=['https://codepen.io/chriddyp/pen/bWLwgP.css'])
server = app.server  # Para Render/Gunicorn

app.layout = html.Div([
    html.H1("Dashboard: Análisis del Dataset Adult", style={'textAlign': 'center', 'color': '#2c3e50'}),
    
    # Análisis Exploratorio
    html.H2("Análisis Exploratorio", style={'color': '#34495e'}),
    html.Div([
        html.P(f"Dimensiones: {df.shape[0]} filas × {df.shape[1]} columnas"),
        html.P(f"Variables numéricas: {len(df.select_dtypes(include=['number']).columns)}"),
        html.P(f"Variables categóricas: {len(df.select_dtypes(include=['object']).columns)}"),
    ], style={'backgroundColor': '#ecf0f1', 'padding': '15px'}),
    html.P("Histograma de Income omitido para optimizar memoria"),
    html.P("Mapa de correlaciones omitido para optimizar memoria"),
    
    # Visualización del Dataset (primeras 5 filas)
    html.H2("Visualización del Dataset", style={'color': '#34495e'}),
    dash_table.DataTable(
        data=df.head().to_dict('records'),
        columns=[{'name': col, 'id': col} for col in df.columns],
        style_table={'overflowX': 'auto'},
        style_cell={'textAlign': 'left'},
        style_header={'backgroundColor': '#3498db', 'color': 'white'}
    ),
    
    # Pipelines
    html.H2("Pipelines de Modelos", style={'color': '#34495e'}),
    dash_table.DataTable(
        data=[
            {
                'Modelo': modelo,
                'Clasificador': str(resultados['pipeline'].named_steps['classifier']).split('(')[0]
            }
            for modelo, resultados in model_results.items()
        ],
        columns=[{'name': 'Modelo', 'id': 'Modelo'}, {'name': 'Clasificador', 'id': 'Clasificador'}],
        style_cell={'textAlign': 'left'},
        style_header={'backgroundColor': '#3498db', 'color': 'white'}
    ),
    
    # Mejor Modelo
    html.H2(f"Mejor Modelo: {best_model_name}", style={'color': '#27ae60'}),
    html.Div([
        html.P(f"Accuracy: {best_results['accuracy']:.4f}"),
        html.P(f"F1-Score: {best_results['f1_score']:.4f}"),
    ], style={'backgroundColor': '#d5f4e6', 'padding': '15px'}),
    
    dcc.Graph(
        figure=px.imshow(
            best_results['confusion_matrix'],
            labels=dict(x="Predicción", y="Actual", color="Cantidad"),
            x=['≤50k', '>50k'],
            y=['≤50k', '>50k'],
            color_continuous_scale='Blues'
        ).update_layout(title=f"Matriz de Confusión - {best_model_name}", width=400, height=400)
    ),
    
    dcc.Graph(id='roc-curve-best')
])

@app.callback(
    Output('roc-curve-best', 'figure'),
    Input('roc-curve-best', 'id')
)
def update_roc_curve(_):
    try:
        if best_results['y_proba'] is None:
            return go.Figure().add_annotation(text="Curva ROC no disponible", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        fpr, tpr, _ = roc_curve(y_val, best_results['y_proba'])
        roc_auc = auc(fpr, tpr)
        fig = go.Figure([
            go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC (AUC = {roc_auc:.2f})', line=dict(color='darkorange')),
            go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random', line=dict(color='navy', dash='dash'))
        ])
        fig.update_layout(title='Curva ROC', xaxis_title='False Positive Rate', yaxis_title='True Positive Rate', width=400, height=400)
        return fig
    except Exception as e:
        print(f"Error en el callback de la curva ROC: {e}")
        return go.Figure().add_annotation(text=f"Error al generar la curva ROC: {str(e)}", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=False)
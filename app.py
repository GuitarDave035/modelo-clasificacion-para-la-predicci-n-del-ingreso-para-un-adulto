import dash
import os
from dash import dcc, html, dash_table, Input, Output, callback
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, 
    confusion_matrix, roc_curve, auc, precision_score, recall_score
)
from imblearn.over_sampling import RandomOverSampler
from ucimlrepo import fetch_ucirepo
from sklearn.base import clone
import warnings
warnings.filterwarnings('ignore')

# ===== CARGA Y PREPROCESAMIENTO DE DATOS =====
def load_and_preprocess_data():
    """Carga y preprocesa el dataset Adult"""
    # Cargar dataset
    adult = fetch_ucirepo(id=2)
    X = adult.data.features
    y = adult.data.targets
    
    # Concatenar features y target
    df = pd.concat([X, y], axis=1)
    
    # Limpiar datos
    df = df[df["income"].notnull()].reset_index(drop=True)
    df = df.drop_duplicates().reset_index(drop=True)
    
    # Normalizar strings
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].str.strip().str.lower()
    
    # Filtrar solo clases válidas
    valores_esperados = ['<=50k', '>50k']
    df = df[df["income"].isin(valores_esperados)].reset_index(drop=True)
    
    # Eliminar columna redundante
    df = df.drop(columns=["education"])
    
    # Codificar variable objetivo
    le = LabelEncoder()
    df["income"] = le.fit_transform(df["income"])
    
    return df, le

def split_data(df):
    """Divide los datos en conjuntos de entrenamiento, validación y prueba"""
    X_pre = df.drop(columns=["income"])
    y_pre = df["income"]
    
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_pre, y_pre, test_size=0.30, stratify=y_pre, random_state=42
    )
    
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def train_models(X_train, X_val, y_train, y_val):
    """Entrena múltiples modelos y retorna resultados"""
    # Identificar columnas
    numeric_cols = X_train.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
    
    # Preprocesamiento
    preprocessor = ColumnTransformer(transformers=[
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler())
        ]), numeric_cols),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ]), categorical_cols)
    ])
    
    # Modelos
    modelos = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "K-Nearest Neighbors": KNeighborsClassifier(),
        "Naive Bayes": GaussianNB(),
        "SVM Linear": SVC(kernel='linear', probability=True),
        "SVM RBF": SVC(kernel='rbf', probability=True),
        "MLP Classifier": MLPClassifier(max_iter=500, random_state=42)
    }
    
    resultados = {}
    
    for nombre, modelo in modelos.items():
        pipeline = Pipeline(steps=[
            ("preprocessing", preprocessor),
            ("classifier", modelo)
        ])
        
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_val)
        y_proba = pipeline.predict_proba(X_val)[:, 1] if hasattr(pipeline, "predict_proba") else None
        
        acc = accuracy_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred, average="macro")
        cm = confusion_matrix(y_val, y_pred)
        
        resultados[nombre] = {
            'accuracy': acc,
            'f1_score': f1,
            'confusion_matrix': cm,
            'y_pred': y_pred,
            'y_proba': y_proba,
            'pipeline': pipeline
        }
    
    return resultados

# ===== CARGA DE DATOS INICIAL =====
df, le = load_and_preprocess_data()
X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)
model_results = train_models(X_train, X_val, y_train, y_val)

# ===== CONFIGURACIÓN DE LA APLICACIÓN DASH =====
app = dash.Dash(__name__)

# Estilos CSS personalizados
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div([
    html.H1("Dashboard: Análisis del Dataset Adult", 
            style={'textAlign': 'center', 'marginBottom': 30, 'color': '#2c3e50'}),
    
    # Tabs principales
    dcc.Tabs(id='main-tabs', value='tab-eda', children=[
        dcc.Tab(label='📊 Análisis Exploratorio', value='tab-eda'),
        dcc.Tab(label='🤖 Modelos ML', value='tab-models'),
        dcc.Tab(label='🎯 Mejor Modelo', value='tab-best-model'),
        dcc.Tab(label='📈 Métricas Finales', value='tab-final-metrics')
    ]),
    
    html.Div(id='tabs-content')
])

@app.callback(Output('tabs-content', 'children'),
              Input('main-tabs', 'value'))
def render_content(tab):
    if tab == 'tab-eda':
        return html.Div([
            html.H2("Análisis Exploratorio de Datos", style={'color': '#34495e'}),
            
            # Información del dataset
            html.Div([
                html.H3("Información General del Dataset"),
                html.P(f"Dimensiones: {df.shape[0]} filas × {df.shape[1]} columnas"),
                html.P(f"Variables numéricas: {len(df.select_dtypes(include=['number']).columns)}"),
                html.P(f"Variables categóricas: {len(df.select_dtypes(include=['object']).columns)}"),
            ], style={'backgroundColor': '#ecf0f1', 'padding': '15px', 'borderRadius': '5px', 'marginBottom': '20px'}),
            
            # Distribución de la variable objetivo
            html.Div([
                html.H3("Distribución de la Variable Objetivo (Income)"),
                dcc.Graph(
                    figure=px.histogram(
                        df, x='income', 
                        title="Distribución de Income (0: ≤50k, 1: >50k)",
                        labels={'income': 'Income', 'count': 'Cantidad'},
                        color_discrete_sequence=['#3498db']
                    ).update_layout(showlegend=False)
                )
            ]),
            
            # División de conjuntos
            html.Div([
                html.H3("División de Conjuntos de Datos"),
                dcc.Graph(
                    figure=go.Figure(data=[
                        go.Bar(x=['Entrenamiento', 'Validación', 'Prueba'], 
                               y=[len(X_train), len(X_val), len(X_test)],
                               text=[f'{len(X_train)/len(df)*100:.1f}%', 
                                     f'{len(X_val)/len(df)*100:.1f}%', 
                                     f'{len(X_test)/len(df)*100:.1f}%'],
                               textposition='auto',
                               marker_color=['#e74c3c', '#f39c12', '#27ae60'])
                    ]).update_layout(
                        title="Tamaño de Conjuntos de Datos",
                        xaxis_title="Conjunto",
                        yaxis_title="Número de Registros"
                    )
                )
            ]),
            
            # Correlaciones
            html.Div([
                html.H3("Mapa de Correlaciones (Variables Numéricas)"),
                dcc.Graph(
                    figure=px.imshow(
                        df.select_dtypes(include=['number']).corr(),
                        text_auto=True,
                        aspect="auto",
                        title="Correlaciones entre Variables Numéricas",
                        color_continuous_scale='RdBu_r'
                    )
                )
            ]),
            
            # Distribuciones por categorías
            html.Div([
                html.H3("Distribuciones de Variables Categóricas"),
                html.P("Selecciona una variable categórica:"),
                dcc.Dropdown(
                    id='categorical-dropdown',
                    options=[{'label': col, 'value': col} 
                            for col in df.select_dtypes(include=['object']).columns],
                    value=df.select_dtypes(include=['object']).columns[0] if len(df.select_dtypes(include=['object']).columns) > 0 else None
                ),
                dcc.Graph(id='categorical-plot')
            ])
        ])
    
    elif tab == 'tab-models':
        return html.Div([
            html.H2("Comparación de Modelos de Machine Learning", style={'color': '#34495e'}),
            
            # Tabla de resultados
            html.Div([
                html.H3("Métricas de Rendimiento"),
                dash_table.DataTable(
                    data=[
                        {
                            'Modelo': modelo,
                            'Accuracy': f"{resultados['accuracy']:.4f}",
                            'F1-Score': f"{resultados['f1_score']:.4f}"
                        }
                        for modelo, resultados in model_results.items()
                    ],
                    columns=[
                        {'name': 'Modelo', 'id': 'Modelo'},
                        {'name': 'Accuracy', 'id': 'Accuracy'},
                        {'name': 'F1-Score', 'id': 'F1-Score'}
                    ],
                    style_cell={'textAlign': 'center'},
                    style_header={'backgroundColor': '#3498db', 'color': 'white'},
                    sort_action="native"
                )
            ], style={'marginBottom': '30px'}),
            
            # Gráfico de comparación
            html.Div([
                html.H3("Comparación Visual de Métricas"),
                dcc.Graph(
                    figure=go.Figure(data=[
                        go.Bar(name='Accuracy', 
                               x=list(model_results.keys()), 
                               y=[res['accuracy'] for res in model_results.values()],
                               marker_color='#3498db'),
                        go.Bar(name='F1-Score', 
                               x=list(model_results.keys()), 
                               y=[res['f1_score'] for res in model_results.values()],
                               marker_color='#e74c3c')
                    ]).update_layout(
                        title="Comparación de Métricas por Modelo",
                        xaxis_title="Modelos",
                        yaxis_title="Puntuación",
                        barmode='group',
                        xaxis_tickangle=-45
                    )
                )
            ]),
            
            # Selector de modelo para matriz de confusión
            html.Div([
                html.H3("Matriz de Confusión por Modelo"),
                html.P("Selecciona un modelo:"),
                dcc.Dropdown(
                    id='model-dropdown',
                    options=[{'label': modelo, 'value': modelo} 
                            for modelo in model_results.keys()],
                    value=list(model_results.keys())[0]
                ),
                dcc.Graph(id='confusion-matrix-plot')
            ])
        ])
    
    elif tab == 'tab-best-model':
        # Encontrar el mejor modelo
        best_model_name = max(model_results.keys(), 
                             key=lambda x: model_results[x]['f1_score'])
        best_results = model_results[best_model_name]
        
        return html.Div([
            html.H2(f"Mejor Modelo: {best_model_name}", style={'color': '#27ae60'}),
            
            html.Div([
                html.H3("Métricas del Mejor Modelo"),
                html.P(f"Accuracy: {best_results['accuracy']:.4f}"),
                html.P(f"F1-Score: {best_results['f1_score']:.4f}"),
            ], style={'backgroundColor': '#d5f4e6', 'padding': '15px', 'borderRadius': '5px', 'marginBottom': '20px'}),
            
            # Matriz de confusión del mejor modelo
            html.Div([
                html.H3("Matriz de Confusión"),
                dcc.Graph(
                    figure=px.imshow(
                        best_results['confusion_matrix'],
                        text_auto=True,
                        aspect="auto",
                        title=f"Matriz de Confusión - {best_model_name}",
                        labels=dict(x="Predicción", y="Actual", color="Cantidad"),
                        x=['≤50k', '>50k'],
                        y=['≤50k', '>50k']
                    )
                )
            ]),
            
            # Curva ROC si está disponible
            html.Div([
                html.H3("Curva ROC"),
                dcc.Graph(id='roc-curve-best')
            ]) if best_results['y_proba'] is not None else html.Div()
        ])
    
    elif tab == 'tab-final-metrics':
        return html.Div([
            html.H2("Métricas Finales y Evaluación", style={'color': '#8e44ad'}),
            
            html.Div([
                html.H3("Resumen de Todos los Modelos"),
                dcc.Graph(
                    figure=make_subplots(
                        rows=1, cols=2,
                        subplot_titles=('Accuracy por Modelo', 'F1-Score por Modelo')
                    ).add_trace(
                        go.Bar(x=list(model_results.keys()),
                               y=[res['accuracy'] for res in model_results.values()],
                               name='Accuracy',
                               marker_color='#3498db'),
                        row=1, col=1
                    ).add_trace(
                        go.Bar(x=list(model_results.keys()),
                               y=[res['f1_score'] for res in model_results.values()],
                               name='F1-Score',
                               marker_color='#e74c3c'),
                        row=1, col=2
                    ).update_layout(
                        title_text="Métricas Finales de Todos los Modelos",
                        showlegend=False
                    ).update_xaxes(tickangle=-45)
                )
            ]),
            
            # Tabla resumen completa
            html.Div([
                html.H3("Tabla Resumen Completa"),
                dash_table.DataTable(
                    data=[
                        {
                            'Modelo': modelo,
                            'Accuracy': f"{resultados['accuracy']:.4f}",
                            'F1-Score': f"{resultados['f1_score']:.4f}",
                            'TP': int(resultados['confusion_matrix'][1,1]),
                            'TN': int(resultados['confusion_matrix'][0,0]),
                            'FP': int(resultados['confusion_matrix'][0,1]),
                            'FN': int(resultados['confusion_matrix'][1,0])
                        }
                        for modelo, resultados in model_results.items()
                    ],
                    columns=[
                        {'name': 'Modelo', 'id': 'Modelo'},
                        {'name': 'Accuracy', 'id': 'Accuracy'},
                        {'name': 'F1-Score', 'id': 'F1-Score'},
                        {'name': 'TP', 'id': 'TP'},
                        {'name': 'TN', 'id': 'TN'},
                        {'name': 'FP', 'id': 'FP'},
                        {'name': 'FN', 'id': 'FN'}
                    ],
                    style_cell={'textAlign': 'center'},
                    style_header={'backgroundColor': '#8e44ad', 'color': 'white'},
                    sort_action="native"
                )
            ])
        ])

@app.callback(
    Output('categorical-plot', 'figure'),
    Input('categorical-dropdown', 'value')
)
def update_categorical_plot(selected_col):
    if selected_col is None:
        return go.Figure()
    
    fig = px.histogram(
        df, y=selected_col, 
        title=f"Distribución de {selected_col}",
        orientation='h'
    )
    fig.update_layout(height=400)
    return fig

@app.callback(
    Output('confusion-matrix-plot', 'figure'),
    Input('model-dropdown', 'value')
)
def update_confusion_matrix(selected_model):
    if selected_model is None:
        return go.Figure()
    
    cm = model_results[selected_model]['confusion_matrix']
    fig = px.imshow(
        cm,
        text_auto=True,
        aspect="auto",
        title=f"Matriz de Confusión - {selected_model}",
        labels=dict(x="Predicción", y="Actual", color="Cantidad"),
        x=['≤50k', '>50k'],
        y=['≤50k', '>50k']
    )
    return fig

@app.callback(
    Output('roc-curve-best', 'figure'),
    Input('main-tabs', 'value')
)
def update_roc_curve(tab):
    if tab != 'tab-best-model':
        return go.Figure()
    
    best_model_name = max(model_results.keys(), 
                         key=lambda x: model_results[x]['f1_score'])
    best_results = model_results[best_model_name]
    
    if best_results['y_proba'] is None:
        return go.Figure().add_annotation(
            text="Curva ROC no disponible para este modelo",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
    
    fpr, tpr, _ = roc_curve(y_val, best_results['y_proba'])
    roc_auc = auc(fpr, tpr)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr,
        mode='lines',
        name=f'ROC Curve (AUC = {roc_auc:.2f})',
        line=dict(color='darkorange', width=2)
    ))
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        name='Random Classifier',
        line=dict(color='navy', width=1, dash='dash')
    ))
    
    fig.update_layout(
        title='Curva ROC',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        width=600, height=500
    )
    
    return fig

if __name__ == '__main__':
    import os
    print("🚀 Iniciando Dashboard del Dataset Adult...")
    print("📊 Datos cargados y modelos entrenados exitosamente!")
    port = int(os.environ.get("PORT", 8050))
    print(f"🌐 Abriendo dashboard en http://0.0.0.0:{port}/")
    app.run(host="0.0.0.0", port=port, debug=False)
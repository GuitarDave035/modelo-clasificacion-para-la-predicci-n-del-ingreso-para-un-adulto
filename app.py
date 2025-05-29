import dash
from dash import dcc, html, dash_table, Input, Output
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
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
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_curve, auc
from ucimlrepo import fetch_ucirepo
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

# ===== CARGA Y PREPROCESAMIENTO DE DATOS (CON PKL) =====
def load_and_preprocess_data():
    """Carga y preprocesa el dataset Adult, guarda en pickle si no existe"""
    pkl_file = 'adult_data.pkl'
    if os.path.exists(pkl_file):
        with open(pkl_file, 'rb') as f:
            df, le = pickle.load(f)
        return df, le

    adult = fetch_ucirepo(id=2)
    X = adult.data.features
    y = adult.data.targets
    df = pd.concat([X, y], axis=1)
    df = df[df["income"].notnull()].reset_index(drop=True)
    df = df.drop_duplicates().reset_index(drop=True)
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].str.strip().str.lower()
    valores_esperados = ['<=50k', '>50k']
    df = df[df["income"].isin(valores_esperados)].reset_index(drop=True)
    df = df.drop(columns=["education"])
    le = LabelEncoder()
    df["income"] = le.fit_transform(df["income"])
    
    with open(pkl_file, 'wb') as f:
        pickle.dump((df, le), f)
    return df, le

def split_data(df):
    """Divide los datos en conjuntos de entrenamiento, validación y prueba"""
    pkl_file = 'split_data.pkl'
    if os.path.exists(pkl_file):
        with open(pkl_file, 'rb') as f:
            return pickle.load(f)

    X_pre = df.drop(columns=["income"])
    y_pre = df["income"]
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_pre, y_pre, test_size=0.30, stratify=y_pre, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42
    )
    
    with open(pkl_file, 'wb') as f:
        pickle.dump((X_train, X_val, X_test, y_train, y_val, y_test), f)
    return X_train, X_val, X_test, y_train, y_val, y_test

def train_models(X_train, X_val, y_train, y_val):
    """Entrena modelos y guarda resultados en pickle"""
    pkl_file = 'model_results.pkl'
    if os.path.exists(pkl_file):
        with open(pkl_file, 'rb') as f:
            return pickle.load(f)

    numeric_cols = X_train.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = X_train.select_dtypes(include=["object"]).columns.tolist()
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
    
    with open(pkl_file, 'wb') as f:
        pickle.dump(resultados, f)
    return resultados

# ===== CARGA DE DATOS Y ENTRENAMIENTO =====
df, le = load_and_preprocess_data()
X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)
model_results = train_models(X_train, X_val, y_train, y_val)

# Encontrar el mejor modelo
best_model_name = max(model_results.keys(), key=lambda x: model_results[x]['f1_score'])
best_results = model_results[best_model_name]

# ===== CONFIGURACIÓN DE LA APLICACIÓN DASH =====
app = dash.Dash(__name__, external_stylesheets=['https://codepen.io/chriddyp/pen/bWLwgP.css'])

# Layout con título arriba y estructura de izquierda a derecha para las secciones
app.layout = html.Div([
    # Título principal en un contenedor separado
    html.Div([
        html.H1("Dashboard: Análisis del Dataset Adult", 
                style={'textAlign': 'center', 'marginBottom': 20, 'color': '#2c3e50', 'padding': '10px'})
    ], style={'width': '100%'}),

    # Contenedor principal con las tres secciones en flexbox
    html.Div([
        # Sección 1: Análisis Exploratorio (Izquierda)
        html.Div([
            html.H2("Análisis Exploratorio", style={'color': '#34495e', 'marginBottom': 15}),
            
            # Información del dataset
            html.Div([
                html.H3("Información General"),
                html.P(f"Dimensiones: {df.shape[0]} filas × {df.shape[1]} columnas"),
                html.P(f"Numéricas: {len(df.select_dtypes(include=['number']).columns)}"),
                html.P(f"Categóricas: {len(df.select_dtypes(include=['object']).columns)}"),
            ], style={'backgroundColor': '#ecf0f1', 'padding': '10px', 'borderRadius': '5px', 'marginBottom': '15px'}),
            
            # Subplots para distribuciones y mapa de calor
            html.Div([
                html.H3("Distribuciones y Correlaciones"),
                dcc.Graph(
                    id='eda-subplots',
                    figure=make_subplots(
                        rows=2, cols=2,
                        subplot_titles=(
                            "Distribución de Income",
                            "Tamaño de Conjuntos",
                            "Mapa de Correlaciones",
                            "Variable Categórica vs Income"
                        ),
                        specs=[[{"type": "histogram"}, {"type": "bar"}],
                               [{"type": "heatmap"}, {"type": "histogram"}]],
                        vertical_spacing=0.15
                    ).add_trace(
                        go.Histogram(
                            x=df['income'],
                            marker_color='#3498db',
                            name='Income'
                        ), row=1, col=1
                    ).add_trace(
                        go.Bar(
                            x=['Entrenamiento', 'Validación', 'Prueba'],
                            y=[len(X_train), len(X_val), len(X_test)],
                            text=[f'{len(X_train)/len(df)*100:.1f}%', 
                                  f'{len(X_val)/len(df)*100:.1f}%', 
                                  f'{len(X_test)/len(df)*100:.1f}%'],
                            textposition='auto',
                            marker_color=['#e74c3c', '#f39c12', '#27ae60'],
                            name='Conjuntos'
                        ), row=1, col=2
                    ).add_trace(
                        go.Heatmap(
                            z=df.select_dtypes(include=['number']).corr(),
                            x=df.select_dtypes(include=['number']).columns,
                            y=df.select_dtypes(include=['number']).columns,
                            colorscale='RdBu_r',
                            text=df.select_dtypes(include=['number']).corr().round(2).values,
                            texttemplate="%{text}",
                            showscale=True
                        ), row=2, col=1
                    ).add_trace(
                        go.Histogram(
                            y=df[df.select_dtypes(include=['object']).columns[0]],
                            x=df['income'],
                            histfunc='count',
                            marker_color='#2ecc71',
                            name='Categórica vs Income'
                        ), row=2, col=2
                    ).update_layout(
                        height=800,
                        showlegend=False,
                        title_text="Análisis Exploratorio de Datos",
                        title_x=0.5
                    ).update_xaxes(title_text="Income (0: ≤50k, 1: >50k)", row=1, col=1)
                    .update_xaxes(title_text="Conjunto", row=1, col=2)
                    .update_xaxes(title_text="Predicción", row=2, col=2)
                    .update_yaxes(title_text="Cantidad", row=1, col=1)
                    .update_yaxes(title_text="Registros", row=1, col=2)
                    .update_yaxes(title_text="Variable", row=2, col=2)
                )
            ]),
            
            # Dropdown para variables categóricas
            html.Div([
                html.P("Variable Categórica:"),
                dcc.Dropdown(
                    id='categorical-dropdown',
                    options=[{'label': col, 'value': col} 
                             for col in df.select_dtypes(include=['object']).columns],
                    value=df.select_dtypes(include=['object']).columns[0],
                    style={'width': '100%'}
                ),
                dcc.Graph(id='categorical-plot')
            ], style={'marginTop': '15px'})
        ], style={'width': '30%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '10px'}),
        
        # Sección 2: Matrices de Confusión (Centro)
        html.Div([
            html.H2("Matrices de Confusión por Modelo", style={'color': '#34495e', 'marginBottom': 15}),
            
            # Dropdown para seleccionar modelo
            html.P("Seleccionar Modelo:"),
            dcc.Dropdown(
                id='model-dropdown',
                options=[{'label': modelo, 'value': modelo} for modelo in model_results.keys()],
                value=list(model_results.keys())[0],
                style={'width': '100%'}
            ),
            
            # Matriz de confusión
            dcc.Graph(id='confusion-matrix-plot'),
            
            # Tabla de métricas
            html.H3("Métricas de Modelos"),
            dash_table.DataTable(
                id='metrics-table',
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
                sort_action="native",
                style_table={'marginTop': '15px'}
            )
        ], style={'width': '30%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '10px'}),
        
        # Sección 3: Mejor Modelo (Derecha)
        html.Div([
            html.H2(f"Mejor Modelo: {best_model_name}", style={'color': '#27ae60', 'marginBottom': 15}),
            
            # Métricas del mejor modelo
            html.Div([
                html.H3("Métricas"),
                html.P(f"Accuracy: {best_results['accuracy']:.4f}"),
                html.P(f"F1-Score: {best_results['f1_score']:.4f}"),
            ], style={'backgroundColor': '#d5f4e6', 'padding': '10px', 'borderRadius': '5px', 'marginBottom': '15px'}),
            
            # Subplots para matriz de confusión, curva ROC y residuos
            html.Div([
                html.H3("Análisis del Mejor Modelo"),
                dcc.Graph(
                    id='best-model-subplots',
                    figure=make_subplots(
                        rows=1, cols=3,
                        subplot_titles=(
                            f"Matriz de Confusión - {best_model_name}",
                            "Curva ROC",
                            "Residuos"
                        ),
                        specs=[[{"type": "heatmap"}, {"type": "scatter"}, {"type": "histogram"}]]
                    ).add_trace(
                        go.Heatmap(
                            z=best_results['confusion_matrix'],
                            x=['≤50k', '>50k'],
                            y=['≤50k', '>50k'],
                            text=best_results['confusion_matrix'],
                            texttemplate="%{text}",
                            colorscale='Blues',
                            showscale=True
                        ), row=1, col=1
                    ).add_traces(
                        [
                            go.Scatter(
                                x=roc_curve(y_val, best_results['y_proba'])[0],
                                y=roc_curve(y_val, best_results['y_proba'])[1],
                                mode='lines',
                                name=f'ROC (AUC = {auc(*roc_curve(y_val, best_results["y_proba"])[:2]):.2f})',
                                line=dict(color='darkorange', width=2)
                            ),
                            go.Scatter(
                                x=[0, 1], y=[0, 1],
                                mode='lines',
                                name='Random',
                                line=dict(color='navy', width=1, dash='dash')
                            )
                        ], rows=[1, 1], cols=[2, 2]
                    ).add_trace(
                        go.Histogram(
                            x=best_results['y_pred'] - y_val,
                            nbinsx=20,
                            marker_color='#e74c3c',
                            name='Residuos'
                        ), row=1, col=3
                    ).update_layout(
                        height=400,
                        showlegend=True,
                        title_text=f"Análisis Detallado - {best_model_name}",
                        title_x=0.5
                    ).update_xaxes(title_text="Predicción", row=1, col=1)
                    .update_xaxes(title_text="False Positive Rate", row=1, col=2)
                    .update_xaxes(title_text="Residuos (Pred - Actual)", row=1, col=3)
                    .update_yaxes(title_text="Actual", row=1, col=1)
                    .update_yaxes(title_text="True Positive Rate", row=1, col=2)
                    .update_yaxes(title_text="Frecuencia", row=1, col=3)
                )
            ])
        ], style={'width': '30%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '10px'})
    ], style={'display': 'flex', 'justifyContent': 'space-between', 'padding': '20px', 'minHeight': '900px'})
])

# ===== CALLBACKS =====
@app.callback(
    Output('categorical-plot', 'figure'),
    Input('categorical-dropdown', 'value')
)
def update_categorical_plot(selected_col):
    if selected_col is None:
        return go.Figure()
    return px.histogram(
        df, y=selected_col, x='income', histfunc='count',
        title=f"{selected_col} vs Income",
        color_discrete_sequence=['#2ecc71'],
        height=400
    ).update_layout(
        xaxis_title="Income (0: ≤50k, 1: >50k)",
        yaxis_title=selected_col,
        showlegend=False
    )

@app.callback(
    Output('confusion-matrix-plot', 'figure'),
    Input('model-dropdown', 'value')
)
def update_confusion_matrix(selected_model):
    if selected_model is None:
        return go.Figure()
    cm = model_results[selected_model]['confusion_matrix']
    return px.imshow(
        cm,
        text_auto=True,
        aspect="auto",
        title=f"Matriz de Confusión - {selected_model}",
        labels=dict(x="Predicción", y="Actual", color="Cantidad"),
        x=['≤50k', '>50k'],
        y=['≤50k', '>50k'],
        color_continuous_scale='Blues'
    ).update_layout(height=400)

if __name__ == '__main__':
    print("🚀 Iniciando Dashboard del Dataset Adult...")
    print("📊 Datos y modelos cargados desde pickles!")
    port = int(os.environ.get("PORT", 8050))
    print(f"🌐 Abriendo dashboard en http://0.0.0.0:{port}/")
    app.run(host="0.0.0.0", port=port, debug=False)
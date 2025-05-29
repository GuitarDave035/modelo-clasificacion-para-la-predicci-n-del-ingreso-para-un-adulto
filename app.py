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
    return df, le

def split_data(df):
    """Divide los datos en conjuntos de entrenamiento, validación y prueba"""
    X_pre = df.drop(columns=["income"])
    y_pre = df["income"]
    X_train, X_temp, y_train, y_temp = train_test_split(X_pre, y_pre, test_size=0.30, stratify=y_pre, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42)
    return X_train, X_val, X_test, y_train, y_val, y_test

def train_models(X_train, X_val, y_train, y_val):
    """Entrena múltiples modelos y retorna resultados"""
    numeric_cols = X_train.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
    preprocessor = ColumnTransformer(transformers=[
        ("num", Pipeline([("imputer", SimpleImputer(strategy="mean")), ("scaler", StandardScaler())]), numeric_cols),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), categorical_cols)
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
        pipeline = Pipeline(steps=[("preprocessing", preprocessor), ("classifier", modelo)])
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_val)
        y_proba = pipeline.predict_proba(X_val)[:, 1] if hasattr(pipeline, "predict_proba") else None
        acc = accuracy_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred, average="macro")
        cm = confusion_matrix(y_val, y_pred)
        resultados[nombre] = {'accuracy': acc, 'f1_score': f1, 'confusion_matrix': cm, 'y_pred': y_pred, 'y_proba': y_proba, 'pipeline': pipeline}
    return resultados

# ===== CARGA DE DATOS INICIAL =====
df, le = load_and_preprocess_data()
X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)
model_results = train_models(X_train, X_val, y_train, y_val)

# ===== CONFIGURACIÓN DE LA APLICACIÓN DASH =====
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div([
    html.H1("Dashboard: Análisis del Dataset Adult", style={'textAlign': 'center', 'marginBottom': 30, 'color': '#2c3e50'}),
    dcc.Tabs(id='main-tabs', value='tab-eda', children=[
        dcc.Tab(label='📊 Análisis Exploratorio', value='tab-eda'),
        dcc.Tab(label='🤖 Modelos ML', value='tab-models'),
        dcc.Tab(label='🎯 Mejor Modelo', value='tab-best-model'),
        dcc.Tab(label='📈 Métricas Finales', value='tab-final-metrics')
    ]),
    html.Div(id='tabs-content')
])

# Resto de los callbacks (render_content, update_categorical_plot, etc.)...

# Exponer el servidor WSGI
server = app.server

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))
    print(f"Running on port {port}")
    app.run_server(host="0.0.0.0", port=port, debug=False)
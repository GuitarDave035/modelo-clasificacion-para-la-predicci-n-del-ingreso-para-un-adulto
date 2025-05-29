import dash
from dash import html

# Crear la aplicación Dash
app = dash.Dash(__name__)

# Exponer el servidor WSGI para gunicorn
server = app.server

# Layout básico para pruebas
app.layout = html.Div([
    html.H1("Prueba de Despliegue en Render"),
    html.P("Si ves este mensaje, el servidor Dash se ha iniciado correctamente.")
])

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8050))
    print(f"🌐 Iniciando servidor en http://0.0.0.0:{port}/")
    app.run(host="0.0.0.0", port=port, debug=False)
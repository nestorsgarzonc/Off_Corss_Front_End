import dash
from dash.dependencies import Input, Output
import dash_html_components as html
from dash.exceptions import PreventUpdate
import dash_core_components as dcc
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd
import base64
import json
import dash_table
#--------------------------------------------
df = pd.read_csv('data/superstore.csv', parse_dates=['Order Date', 'Ship Date'])
df['Order_Month'] = pd.to_datetime(df['Order Date'].map(lambda x: "{}-{}".format(x.year, x.month)))
with open('data/us.json') as geo:
    geojson = json.loads(geo.read())
states=['California', 'Texas','New York']
ddf=df[df['State'].isin(states)]
ddf=ddf.groupby(['State','Order_Month']).sum().reset_index()
lineplot=px.box(ddf,x="Order_Month",y="Sales", color="State")
scatter=px.scatter(df, x="Sales", y="Profit", color="Category", hover_data=['State','Sub-Category','Order ID','Product Name'])  

app = dash.Dash(__name__,external_stylesheets=[dbc.themes.BOOTSTRAP])

# the style arguments for the sidebar. We use position:fixed and a fixed width
SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "16rem",
    "padding": "2rem 1rem",
    "background-color": "#f8f9fa",
}

# the styles for the main content position it to the right of the sidebar and
# add some padding.
CONTENT_STYLE = {
    "margin-left": "18rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem",
}

sidebar = html.Div(
    [
        html.H2("Barra Test", className="display-4"),
        html.Hr(),
        html.P(
            "Una barra de test", className="lead"
        ),
        dbc.Nav(
            [
                dbc.NavLink("Mapas de calor por trafico en tienda", href="/page-1", id="page-1-link"),
                dbc.NavLink("Metricas de ingresos por tienda", href="/page-2", id="page-2-link"),
                dbc.NavLink("Procesar video", href="/page-3", id="page-3-link"),
                dbc.NavLink("About", href="/page-4", id="page-4-link"),
            ],
            vertical=True,
            pills=True,
        ),
    ],
    style=SIDEBAR_STYLE,
)

content = html.Div(id="page-content", style=CONTENT_STYLE)

#----------------------------------------
from urllib.request import urlopen
with urlopen('https://gist.githubusercontent.com/john-guerra/43c7656821069d00dcbc/raw/be6a6e239cd5b5b803c6e7c2ec405b793a9064dd/Colombia.geo.json') as response:
    counties = json.load(response)
    
locs = ['ANTIOQUIA', 'ATLANTICO', 'BOLIVAR']
for loc in counties['features']:
    loc['id'] = loc['properties']['NOMBRE_DPT']
#----------------------------------------
colors = {
    'background': '#111111',
    'text': '#7FDBFF'
}

image_filename = 'Images/offlogo.png' # replace with your own image
encoded_image = base64.b64encode(open(image_filename, 'rb').read())

app.layout = html.Div([
    html.Div([dcc.Location(id="url"), sidebar, content]),
    html.Div(
        className="app-header",
        style={
            'textAlign': 'center',
            'color': colors['text']},
        children=[
            html.Div('Off Corss CV Team 69', className="app-header--title")
        ]
    ),
    html.Div(className="center",
            style={'backgroundColor': colors['background'],
                  'color': colors['text']},
            children=html.Div(
             [
            html.Img(src='data:image/png;base64,{}'.format(encoded_image.decode()), className="center"),
            html.H1('Overview', style={
            'textAlign': 'center'}),
            html.Div('''
                Esto es solo un Test
            ''', style={
            'textAlign': 'center'})
        ])
    ),
    html.Div([
    html.H2("Esto sigue siendo un test", id='title',style={'textAlign': 'center'}), 
    ], className='jumbotron'),
    
    
    dbc.Row([
        dbc.Col([dcc.Graph(figure=scatter, id='scatter'),]),
        dbc.Col([dcc.Graph(figure=lineplot, id='lineplot'),]),
            ]),
    
    dcc.Slider(min=0,max=1,marks={1:'Col Map', 0:'Scatter Plot'},value=0,id='fig-slider',),
    
    html.Div(children = [
        html.Video(
            controls = True,
            id = 'movie_player',
            src = "https://www.w3schools.com/html/mov_bbb.mp4",
            autoPlay=True
           , className="center"
        ),
    ]),
    
    
    dash_table.DataTable(id='table',
                         columns=[{'name':i, 'id':i} for i in df.columns],
                         data=df.head(5).to_dict('records'),
    )
    

])


@app.callback(Output('table','data'), [Input('scatter', 'clickData')])
def changeTable(clickData):
    print(clickData)
    if clickData is None:
        raise PreventUpdate
    OrderID=clickData['points'][0]['customdata'][2]
    ddf=df[df['Order ID']==OrderID]
    return ddf.head(5).to_dict('records')
@app.callback(Output('scatter', 'figure'), [Input('fig-slider','value' )])
def changeFigure(value):
    if value==1:
        USMAP = go.Figure(go.Choroplethmapbox(
                    geojson=counties,
                    locations=locs,
                    z=[1, 2, 3],
                    colorscale='Viridis',
                    colorbar_title="Aún es ejemplo"))
        USMAP.update_layout(mapbox_style="carto-positron",
                        mapbox_zoom=3,
                        mapbox_center = {"lat": 4.570868, "lon": -74.2973328})
        return USMAP
    if value==0:
        return scatter
# Main
if __name__ == '__main__':
    app.run_server(host='0.0.0.0',port='8050',debug=True)

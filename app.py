""" First Front End Team 69 """

# Dash Libraries
import dash
import dash_table
from dash import no_update
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State

# Flask libraries
from flask import Flask, send_from_directory

# Python Libraries
import io
import re
import cv2
import json
import base64
import requests
import numpy as np
import pandas as pd
from PIL import Image
from skimage import io
import matplotlib.cm as cm
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from datetime import date, datetime
from scipy.ndimage.filters import gaussian_filter
# SQL Alchemy
import sqlalchemy
from sqlalchemy import create_engine, text

# ///////////////////////////////////////////////////////////////////////////////////////////////////
# EC2 Credentials
url = "http://ec2-3-82-23-253.compute-1.amazonaws.com"
# ///////////////////////////////////////////////////////////////////////////////////////////////////
# ///////////////////////////////////////////////////////////////////////////////////////////////////
# DataBase Credentials
host = 'team-cv.cfsx82z4jthl.us-east-2.rds.amazonaws.com'
port = 5432
user = 'ds4a_69'
password = 'DS4A!2020'
database = 'postgres'
# ///////////////////////////////////////////////////////////////////////////////////////////////////
# Image Sources
image_sidebar = 'Images/side_bar_logo3.png'  # replace with your own image
encoded_image = base64.b64encode(open(image_sidebar, 'rb').read())
# Cover Page images
cover_child = 'Images/portada_nignos.jpg'
cover_img = base64.b64encode(open(cover_child, 'rb').read())
# ---
metrics_images = "Images/Metrics_icon.png"
metrics_img = base64.b64encode(open(metrics_images, 'rb').read())
# ---
cloud_images = "Images/upload_vector.jpg"
cloud_img = base64.b64encode(open(cloud_images, 'rb').read())
# Crew
crew_1 = 'Images/Crew1.jpg'
crew_2 = 'Images/Crew2.jpg'
crew_3 = 'Images/Crew3.jpg'
crew_4 = 'Images/Crew4.jpg'
crew_5 = 'Images/Crew5.jpg'
crew_6 = 'Images/Crew6.jpg'
crew_7 = 'Images/Crew7.jpg'

encoded_crew1 = base64.b64encode(open(crew_1, 'rb').read())
encoded_crew2 = base64.b64encode(open(crew_2, 'rb').read())
encoded_crew3 = base64.b64encode(open(crew_3, 'rb').read())
encoded_crew4 = base64.b64encode(open(crew_4, 'rb').read())
encoded_crew5 = base64.b64encode(open(crew_5, 'rb').read())
encoded_crew6 = base64.b64encode(open(crew_6, 'rb').read())
encoded_crew7 = base64.b64encode(open(crew_7, 'rb').read())
# Team logo
team_l = "Images/Team.png"
encoded_logo = base64.b64encode(open(team_l, 'rb').read())

# Store Maps
san_diego_store = 'Images/Planos_San_Diego_1.jpg'
san_diego_blue = base64.b64encode(open(san_diego_store, 'rb').read())
# ///////////////////////////////////////////////////////////////////////////////////////////////////
# Upload directory definition
UPLOAD_DIRECTORY = "Uploaded_Videos"
# ///////////////////////////////////////////////////////////////////////////////////////////////////
# Button definition
button_groups = html.Div(
    [
        dbc.ButtonGroup(
            [dbc.Button("Upload"), dbc.Button(
                "Process"), dbc.Button("Cancel")],
            size="lg",
            className="mr-1")
    ]
)

button = dbc.Button("Upload", color="primary", block=True)
# ///////////////////////////////////////////////////////////////////////////////////////////////////
# JumboTron for the Home Page
card_content_cover = [
    dbc.CardImg(
        src='data:image/jpg;base64,{}'.format(cover_img.decode()), top=True),
]

card_cover = dbc.Jumbotron(
    className="jumboclass"
)

card_int_1 = [
    dbc.CardHeader("Explore the traffic metrics in the Stores"),
    dbc.CardBody(
        [
            dbc.CardImg(src='data:image/png;base64,{}'.format(metrics_img.decode()),
                        top=True, className="card-img-top"),
            html.Hr(className="my-2"),
            dbc.Button("Take me there!", href="/page-4",
                       color="primary", className="metrics_button"),
        ], className="home_card"
    ),
]

card_int_2 = [
    dbc.CardHeader("Make quick heat analysis of multiple videos!"),
    dbc.CardBody(
        [
            dbc.CardImg(
                src='data:image/jpg;base64,{}'.format(cloud_img.decode()), top=True),
            html.Hr(className="my-2"),
            dbc.Button("Take me there!", href="/page-2",
                       color="primary", className="metrics_button"),
        ], className="home_card"
    ),
]

card_int_3 = [
    dbc.CardHeader("Meet our Company"
                   " and Learn more about OffCorss"),
    dbc.CardBody(
        [
            dbc.CardImg(
                src='data:image/png;base64,{}'.format(encoded_image.decode()), top=True),
            html.Hr(className="my-2"),
            dbc.Button("Take me there!", href="https://www.offcorss.com/",
                       color="primary", className="metrics_button"),
        ], className="home_card"
    ),
]

cards_home = dbc.CardColumns(
    [
        dbc.Card(card_int_1, body=True),
        dbc.Card(card_int_2, body=True),
        dbc.Card(card_int_3, body=True),
    ]
)

jumbotron_links = html.Div(
    [
        dbc.Button("", href="https://www.mintic.gov.co/portal/inicio/",
                   color="primary", className="mintic_button"),
        dbc.Button("", href="https://www.correlation-one.com/ds4a-latam",
                   color="primary", className="correl_button"),
        dbc.Button("", href="https://dash.plotly.com/",
                   color="primary", className="ploty_button"),
    ],
    className="pretty_container"
)


jumbotron_int = dbc.Jumbotron(
    [
        html.P(
            "Welcome to the CV APP "
            ", we hope you to enjoy the Deep Learning Ride",
            className="lead",
        ),
        html.P(
            cards_home
        ),
        html.P(
            dbc.CardImg(src='data:image/png;base64,{}'.format(encoded_logo.decode()),
                        top=True, className="radius_img"),
        ),
        jumbotron_links
    ],
    className="welcome_jumbo"
)

# ///////////////////////////////////////////////////////////////////////////////////////////////////
# About Us cards
card_content_1 = [
    dbc.CardHeader("Crew Mate 1", className="crew_card"),
    dbc.CardBody(
        [
            dbc.CardImg(src='data:image/jpg;base64,{}'.format(encoded_crew1.decode()),
                        top=True, className="radius_img"),
            html.H5("Alekcei Hernández", className="center-text"),
            html.Div(
                [
                    dbc.Button("", href="https://www.linkedin.com/in/alekceinosach/",
                               color="primary", className="linked_button"),
                    dbc.Button("", href="https://www.correlation-one.com/ds4a-latam",
                               color="primary", className="email_button"),
                ],
            )
        ], className="radius_card"
    ),
]

card_content_2 = [
    dbc.CardHeader("Crew Mate 2", className="crew_card"),
    dbc.CardBody(
        [
            dbc.CardImg(src='data:image/jpg;base64,{}'.format(encoded_crew2.decode()),
                        top=True, className="radius_img"),
            html.H5("Alexander Sandoval", className="center-text"),
            html.Div(
                [
                    dbc.Button("", href="https://www.linkedin.com/in/alexander-sandoval-202003/",
                               color="primary", className="linked_button"),
                    dbc.Button("", href="https://www.correlation-one.com/ds4a-latam",
                               color="primary", className="email_button"),
                ],
            )
        ], className="radius_card"
    ),
]

card_content_3 = [
    dbc.CardHeader("Crew Mate 3", className="crew_card"),
    dbc.CardBody(
        [
            dbc.CardImg(src='data:image/jpg;base64,{}'.format(encoded_crew3.decode()),
                        top=True, className="radius_img"),
            html.H5("David Henriquez", className="center-text"),
            html.Div(
                [
                    dbc.Button("", href="https://www.linkedin.com/in/david-henriquez-bernal/",
                               color="primary", className="linked_button"),
                    dbc.Button("", href="https://www.correlation-one.com/ds4a-latam",
                               color="primary", className="email_button"),
                ],
            )
        ], className="radius_card"
    ),
]

card_content_4 = [
    dbc.CardHeader("Crew Mate 4", className="crew_card"),
    dbc.CardBody(
        [
            dbc.CardImg(src='data:image/jpg;base64,{}'.format(encoded_crew4.decode()),
                        top=True, className="radius_img"),
            html.H5("Guillermo Valencia", className="center-text"),
            html.Div(
                [
                    dbc.Button("", href="https://www.linkedin.com/in/guillermo-valencia-53a150a4/",
                               color="primary", className="linked_button"),
                    dbc.Button("", href="https://www.correlation-one.com/ds4a-latam",
                               color="primary", className="email_button"),
                ],
            )
        ], className="radius_card"
    ),
]

card_content_5 = [
    dbc.CardHeader("Crew Mate 5", className="crew_card"),
    dbc.CardBody(
        [
            dbc.CardImg(src='data:image/jpg;base64,{}'.format(encoded_crew5.decode()),
                        top=True, className="radius_img"),
            html.H5("Harold García", className="center-text"),
            html.Div(
                [
                    dbc.Button("", href="https://www.linkedin.com/in/haroldgiovannygarciar",
                               color="primary", className="linked_button"),
                    dbc.Button("", href="https://www.correlation-one.com/ds4a-latam",
                               color="primary", className="email_button"),
                ],
            )
        ], className="radius_card"
    ),
]

card_content_6 = [
    dbc.CardHeader("Crew Mate 6", className="crew_card"),
    dbc.CardBody(
        [
            dbc.CardImg(src='data:image/jpg;base64,{}'.format(encoded_crew6.decode()),
                        top=True, className="radius_img"),
            html.H5("Sebastian Garzón", className="center-text"),
            html.Div(
                [
                    dbc.Button("", href="https://www.linkedin.com/in/sebastiangarzonc/",
                               color="primary", className="linked_button"),
                    dbc.Button("", href="https://www.correlation-one.com/ds4a-latam",
                               color="primary", className="email_button"),
                ],
            )
        ], className="radius_card"
    ),
]

card_content_7 = [
    dbc.CardHeader("Crew Mate 7", className="crew_card"),
    dbc.CardBody(
        [
            dbc.CardImg(src='data:image/jpg;base64,{}'.format(encoded_crew7.decode()),
                        top=True, className="radius_img"),
            html.H5("Nicolás González", className="center-text"),
            html.Div(
                [
                    dbc.Button("", href="https://www.linkedin.com/in/jose-nicol%C3%A1s-gonz%C3%A1lez-guatibonza-41205076/",
                               color="primary", className="linked_button"),
                    dbc.Button("", href="https://www.correlation-one.com/ds4a-latam",
                               color="primary", className="email_button"),
                ],
            )
        ], className="radius_card"
    ),
]

cards = dbc.CardColumns(
    [
        dbc.Card(card_content_1, color="warning", inverse=True),
        dbc.Card(card_content_2, color="warning", inverse=True),
        dbc.Card(card_content_3, color="warning", inverse=True),
    ]
)

cards_2 = dbc.CardColumns(
    [
        dbc.Card(card_content_4, color="warning", inverse=True),
        dbc.Card(card_content_5, color="warning", inverse=True),
        dbc.Card(card_content_6, color="warning", inverse=True),
    ]
)

cards_3 = dbc.CardColumns(
    [
        dbc.Card("", inverse=True, className="card_black"),
        dbc.Card(card_content_7, color="warning", inverse=True),
        dbc.Card("", inverse=True, className="card_black"),
    ]
)
# ///////////////////////////////////////////////////////////////////////////////////////////////////
# Form for the date, hour and cam picking I
# 1. Form for stores
dropdown_stores = dbc.FormGroup(
    [
        dbc.Label("Stores", html_for="dropdown"),
        dcc.Dropdown(
            id="dropdown_store",
            options=[
                {"label": "San Diego Store", "value": 'san diego'},
                {"label": "Santa Fe Store", "value": 'santa fe'},
            ],
            value='san diego'
        ),
        html.Div(id='output-dropdown-stores')
    ],  className="form_style"
)

# 2. Form for the Hours
hour_picker_start = dbc.Card([
    dbc.Label("Start_Hour"),
    dcc.Input(
        id="Start Hour", type="number",
        debounce=True, placeholder="Hour", min=0, max=23
    ),
    dcc.Input(
        id="Start Minute", type="number",
        debounce=True, placeholder="Minute", min=0, max=59
    ),
    html.Div(id="number-out"),
], className="form_style")

hour_picker_end = dbc.Card([
    dbc.Label("End_Hour"),
    dcc.Input(
        id="End Hour", type="number",
        debounce=True, placeholder="Hour", min=0, max=23
    ),
    dcc.Input(
        id="End Minute", type="number",
        debounce=True, placeholder="Minute", min=0, max=59
    ),
    html.Div(id="number-out-end"),
], className="form_style")

# 3. Date Picking Form
date_picker = dbc.Card([
    dbc.Label("Date"),
    dcc.DatePickerSingle(
        id='my-date-picker-single',
        min_date_allowed=date(2020, 10, 1),
        max_date_allowed=date(2020, 10, 19),
        initial_visible_month=date(2020, 10, 1),
        date=date(2020, 10, 1),
        className="date_pick_style"
    ),
    html.Div(id='output-container-date-picker-single')
], className="form_style_nd")

# 4. Cam Picking Form
Cam_picker = dbc.FormGroup(
    [
        dbc.Label("Cameras", html_for="dropdown"),
        dcc.Dropdown(
            id="dropdown_cams",
            options=[
                {"label": "Cam 1", "value": 'Cam 1'},
                {"label": "Cam 2", "value": 'Cam 2'},
                {"label": "Cam 3", "value": 'Cam 3'},
                {"label": "Cam 4", "value": 'Cam 4'},
                {"label": "Cam 5", "value": 'Cam 5'},
                {"label": "All", "value": 'All'},
            ],
            value='All'
        ),
        html.Div(id='output-dropdown-cameras')
    ], className="form_style"
)

# 5. Final form
form_store_pick = html.Div(
    [
        dbc.Row(
            [dbc.Col(
                dbc.FormGroup([dropdown_stores, date_picker])

            ),
                dbc.Col(
                dbc.FormGroup([hour_picker_start, hour_picker_end])
            ),
                dbc.Col(
                dbc.FormGroup([Cam_picker,
                               html.Button('Launch!', id='submit-val', n_clicks=0, className="ico_submit")])
            )
            ])
    ])
# ///////////////////////////////////////////////////////////////////////////////////////////////////
card_date = dbc.Card(
    [
        dbc.CardHeader("Traffic Processing Module",
                       className="title_video_picker"),
        dbc.CardBody(
            [
                form_store_pick
            ]
        ),
        dbc.CardFooter("Please select a Date, Start and End Hour and the Cam to start the video analysis",
                       className="foot_video_picker"),
    ],
    className="big_form_style"
)
# ///////////////////////////////////////////////////////////////////////////////////////////////////
# Form for the date, hour and cam picking II
# 1. Form for stores
dropdown_stores_II = dbc.FormGroup(
    [
        dbc.Label("Stores", html_for="dropdown"),
        dcc.Dropdown(
            id="dropdown_store_II",
            options=[
                {"label": "San Diego Store", "value": 'san diego'},
                {"label": "Santa Fe Store", "value": 'santa fe'},
            ],
            value='san diego'
        ),
        html.Div(id='output-dropdown-stores-II')
    ],  className="form_style"
)

# 2. Form for the Hours
hour_picker_start_II = dbc.Card([
    dbc.Label("Start_Hour"),
    dcc.Input(
        id="Start Hour II", type="number",
        debounce=True, placeholder="Hour", min=0, max=23
    ),
    dcc.Input(
        id="Start Minute II", type="number",
        debounce=True, placeholder="Minute", min=0, max=59
    ),
    html.Div(id="number-out-II"),
], className="form_style_II")

hour_picker_end_II = dbc.Card([
    dbc.Label("End_Hour"),
    dcc.Input(
        id="End Hour II", type="number",
        debounce=True, placeholder="Hour", min=0, max=23
    ),
    dcc.Input(
        id="End Minute II", type="number",
        debounce=True, placeholder="Minute", min=0, max=59
    ),
    html.Div(id="number-out-end-II"),
], className="form_style_II")

# 3. Date Picking Form
date_picker_II = dbc.Card([
    dbc.Label("Date"),
    dcc.DatePickerRange(
        id='my-date-picker-single-II',
        start_date_placeholder_text="Start Period",
        end_date_placeholder_text="End Period",
        calendar_orientation='vertical',
        min_date_allowed=date(2020, 10, 1),
        max_date_allowed=date(2020, 10, 19),
        initial_visible_month=date(2020, 10, 1),
        className="date_pick_style"
    ),
    html.Div(id='output-container-date-picker-single-II')
], className="form_style_nd")

# 4. Cam Picking Form
Cam_picker_II = dbc.FormGroup(
    [
        dbc.Label("Cameras", html_for="dropdown"),
        dcc.Dropdown(
            id="dropdown_cams_II",
            options=[
                {"label": "Cam 1", "value": 'Cam 1'},
                {"label": "Cam 2", "value": 'Cam 2'},
                {"label": "Cam 3", "value": 'Cam 3'},
                {"label": "Cam 4", "value": 'Cam 4'},
                {"label": "Cam 5", "value": 'Cam 5'},
                {"label": "All", "value": 'All'},
            ],
            value='All'
        ),
        html.Div(id='output-dropdown-cameras-II')
    ], className="form_style"
)

# 5. Final form
form_store_pick_II = html.Div(
    [
        dbc.Row(
            [dbc.Col(
                dbc.FormGroup([dropdown_stores_II, date_picker_II])

            ),
                dbc.Col(
                dbc.FormGroup([hour_picker_start_II, hour_picker_end_II])
            ),
                dbc.Col(
                dbc.FormGroup([Cam_picker_II,
                               html.Button('Launch!', id='submit-val-II', n_clicks=0, className="ico_submit")])
            )
            ])
    ])

# ///////////////////////////////////////////////////////////////////////////////////////////////////
card_trends = dbc.Card(
    [
        dbc.CardHeader("Trends Module", className="title_video_picker"),
        dbc.CardBody(
            [
                form_store_pick_II,
            ]
        ),
        dbc.CardFooter("Please select a Date, Start and End Hour and the Cam to start the video analysis",
                       className="foot_video_picker"),
    ],
    className="big_form_style_II"
)
# ///////////////////////////////////////////////////////////////////////////////////////////////////
# Database Functions


def get_db():
    """ Function to conect to data base in postgres """
    # Parameters
    host = "team-cv.cfsx82z4jthl.us-east-2.rds.amazonaws.com"
    user = "ds4a_69"
    port = "5432"
    password = "DS4A!2020"
    database = "postgres"

    # Create the engine with the db credentials
    engine = sqlalchemy.create_engine(
        f'postgresql://{user}:{password}@{host}:{port}/{database}', max_overflow=20)
    return engine


def filter_df(engine, store, date, start_hour, start_min, end_hour, end_min, cam):
    """ Function to generate fig, count people per second """
    # Filter dates
    start_date = datetime.strptime(
        date + ' ' + start_hour.zfill(2) + ':' + start_min.zfill(2) + ':00', '%d/%m/%Y %H:%M:%S')
    end_date = datetime.strptime(
        date + ' ' + end_hour.zfill(2) + ':' + end_min.zfill(2) + ':00', '%d/%m/%Y %H:%M:%S')

    connection = engine.connect()

    query = '''
            SELECT * from tracker 
            WHERE "Store_name"= :group 
            AND "current_datetime" BETWEEN :A AND :B 
            AND "Camera"= :C
            AND "Object"= :O
            '''

    tracker_conn = connection.execute(text(
        query), group='san diego', A=start_date, B=end_date, C=int(cam[-1]), O='person').fetchall()
    columns = ['Store_name', 'Start_date', 'End_date', 'current_datetime', 'Camera', 'Object', 'Id', 'X_center_original', 'Y_center_original',
               'X_center_perspective', 'Y_center_perspective', 'X_min', 'Y_min', 'X_max', 'Y_max', 'Frame']
    tracker = pd.DataFrame(tracker_conn, columns=columns)

    return tracker


def visual_count(tracker):
    """ Function to generate count og people per sec """
    # df_plot = tracker.groupby(['Store_name','Current_date','Frame']).count()['Second'].reset_index()
    df_plot = tracker.groupby(['current_datetime', 'Frame']).count()[
        'Store_name'].reset_index()
    # df_plot = df_plot.groupby(['Store_name','Current_date']).mean().reset_index()
    df_plot['hour'] = df_plot['current_datetime'].dt.hour
    df_plot['minute'] = df_plot['current_datetime'].dt.minute
    df_plot['second'] = df_plot['current_datetime'].dt.second
    df_plot = df_plot.groupby(['hour', 'minute']).mean().reset_index()
    df_plot['hour_minute'] = df_plot['hour'].astype(
        'str') + ':' + df_plot['minute'].astype('str')

    df_plot['People'] = np.round(df_plot['Store_name'])
    # fig = px.line(df_plot, x = "Current_date",  y = "People", color = 'Store_name', title = 'Number of persons per second of video')
    fig = px.line(df_plot, x='hour_minute',  y="People",
                  title='Number of persons per minute of video')
    fig.update_layout(yaxis=dict(range=[0, max(df_plot['People']) + 1]))
    fig.update_layout(width=900, height=600)
    return fig


def print_path(tracker, plane):
    """ Function to print path inside a plane """
    tracker = tracker[(tracker['X_center_perspective'] != 0)
                      & (tracker['Y_center_perspective'] != 0)]
    img = plane
    fig = px.imshow(img)
    fig.update_traces(hoverinfo='skip')
    fig.add_trace(go.Scatter(x=tracker['X_center_perspective'],
                             y=tracker['Y_center_perspective'],
                             marker_color=tracker['Id'],
                             name='Id',
                             mode='markers'))
    fig.update_layout(coloraxis_showscale=False)
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    fig.update_layout(autosize=False, width=900, height=600, margin=go.layout.Margin(
        l=0,  # left margin
        r=0,  # right margin
        b=0,  # bottom margin
        t=0  # top margin
    ))
    fig.update_yaxes(automargin=True)
    return fig


def print_hot_spots(df, plane, name_temp, sigma=4):
    # Get the data to plot
    df = df[(df['X_center_perspective'] != 0) &
            (df['Y_center_perspective'] != 0)]
    x = list(df['X_center_perspective'])
    y = list(df['Y_center_perspective'])

    # Create imagen and save temp
    base_image = plane
    fig, ax = plt.subplots(1, 1, figsize=(20, 16))
    heatmap, xedges, yedges = np.histogram2d(
        x, y, bins=200, range=[[0, 1280], [0, 720]])
    heatmap = gaussian_filter(heatmap, sigma=sigma)
    extent = [xedges[0], xedges[-1], yedges[-1], yedges[0]]
    img = heatmap.T
    ax.imshow(img, extent=extent, cmap=cm.jet)
    ax.imshow(base_image, alpha=0.5)
    ax.set_xlim(0, 1280)
    ax.set_ylim(720, 0)
    ax.axes.get_yaxis().set_visible(False)
    ax.axes.get_xaxis().set_visible(False)
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
    plt.savefig(name_temp, bbox_inches='tight')
    plt.close(fig)

    # Upload imagen in plotly
    img = io.imread(name_temp)
    fig = px.imshow(img)
    fig.update_traces(hoverinfo='skip')
    fig.update_layout(coloraxis_showscale=False)
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    fig.update_layout(autosize=False, width=900, height=600, margin=go.layout.Margin(
        l=0,  # left margin
        r=0,  # right margin
        b=0,  # bottom margin
        t=0  # top margin
    ))
    fig.update_yaxes(automargin=True)
    return fig


# **************************************************************************************
store = 'san diego'
date = '01/10/2020'
start_hour = '18'
start_min = '57'
end_hour = '23'
end_min = '59'
cam = 'Cam 1'

imgx = Image.open('./Images/Planos_San_Diego_1.jpg')
imgx = imgx.convert("RGBA")
engine = get_db()
df = filter_df(engine, store, date, start_hour,
               start_min, end_hour, end_min, cam)
lineplot = visual_count(df)
pathx = print_path(df, imgx)
# ---------------------------------------
heat_map = print_hot_spots(df, imgx, 'temp_hot_spot.jpeg')
# ---------------------------------------
Graphs_tracker = html.Div([
    html.P("People Tracker Map"),
    dcc.Graph(figure=pathx, id="map_traffic"),
    html.Hr(),
    html.P("Store Traffic Heat Map"),
    dcc.Graph(figure=heat_map, id="heat_traffic"),
    html.Hr(),
    html.P("Traffic Lineplot"),
    dcc.Graph(figure=lineplot, id="lin_traffic"),
], id='graph_tracker')

# ///////////////////////////////////////////////////////////////////////////////////////////////////


def counter_df(engine, store, date_start, date_end, start_hour, start_min, end_hour, end_min, cam):
    """ Function to generate fig, count people per second """

    # Filter dates
    start_date = datetime.strptime(date_start + ' ' + start_hour.zfill(
        2) + ':' + start_min.zfill(2) + ':00', '%d/%m/%Y %H:%M:%S')
    end_date = datetime.strptime(
        date_end + ' ' + end_hour.zfill(2) + ':' + end_min.zfill(2) + ':00', '%d/%m/%Y %H:%M:%S')

    connection = engine.connect()
    query = '''
            SELECT * from counts 
            WHERE "Store_name"= :group 
            AND "Start_date" >= :A
            AND "End_date" <= :B 
            AND "Camera"= :C
            '''

    counter_conn = connection.execute(text(
        query), group=store, A=start_date, B=end_date, C=int(cam[-1]), O='person').fetchall()
    columns = ['Store_name', 'Start_date', 'End_date',
               'Camera', 'Count', 'inout', 'name_video']
    counter = pd.DataFrame(counter_conn, columns=columns)
    return counter
# -------------------------------------------------------------------------------


def count_plot(counts_dt, filter_1='day_of_week_name', filter_2='hour'):
    vars_plot = {'day_of_week_name': 'Days of the Week',
                 'hour': 'Hours',
                 'minute': 'Minutes',
                 'week': 'Week',
                 'weekofyear': 'Week of Year',
                 'month': 'Month',
                 'day': 'Day',
                 'year': 'Year'}
    vars_select = [filter_1, filter_2]
    var_0 = vars_plot[vars_select[0]]
    var_1 = vars_plot[vars_select[1]]
    into = counts_dt.loc[counts_dt.inout == "In", :]
    dw_mapping = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday',
                  4: 'Friday', 5: 'Saturday', 6: 'Sunday'}
    into.loc[:, 'day_of_week_name'] = into.Start_date.dt.weekday.map(
        dw_mapping)
    into.loc[:, 'hour'] = into.Start_date.dt.hour
    into.loc[:, 'minute'] = into.Start_date.dt.minute
    into.loc[:, 'week'] = into.Start_date.dt.week
    into.loc[:, 'weekofyear'] = into.Start_date.dt.weekofyear
    into.loc[:, 'year'] = into.Start_date.dt.year
    into.loc[:, 'month'] = into.Start_date.dt.month
    into.loc[:, 'day'] = into.Start_date.dt.day
    df_plot = into.groupby([vars_select[0], vars_select[1]]).sum()[
        'Count'].reset_index()
    fig = px.bar(df_plot, y='Count', x=vars_select[1], facet_col=vars_select[0],
                 title='Counter of People per {} and {}'.format(var_0, var_1))
    total_in = np.sum(into['Count'])
    total_out = np.sum(counts_dt.loc[counts_dt.inout == "Out", 'Count'])
    return {'fig': fig, 'in': total_in, 'out': total_out}


# **************************************************************************************
engine = get_db()
store = 'san diego'
date_start = '01/10/2020'
date_end = '03/10/2020'
start_hour = '00'
start_min = '00'
end_hour = '23'
end_min = '59'
cam = 'Cam 1'

counts_dt = counter_df(engine, store, date_start, date_end,
                       start_hour, start_min, end_hour, end_min, cam)
plot = count_plot(counts_dt, filter_1='day_of_week_name', filter_2='hour')
plotx = plot['fig']

in_pep = plot['in']
out_pep = plot['out']

Graphs_counter = html.Div([html.P("People Counter Graph"),
                           dcc.Graph(figure=plotx, id="count_week"),
                           ], id='graph_counter')

card_count_1 = [
    dbc.CardHeader("Total count of customer in-traffic:",
                   className="crew_card"),
    dbc.CardBody(
        [
            html.Div(in_pep, id='count_in'),
        ], className="radius_card_2"
    ),
]

card_count_2 = [
    dbc.CardHeader("Total count of customer out-traffic:",
                   className="crew_card"),
    dbc.CardBody(
        [
            html.Div(out_pep, id='count_out'),
        ], className="radius_card_2"
    ),
]

# ///////////////////////////////////////////////////////////////////////////////////////////////////
# Heat analysis
video_library = dbc.FormGroup(
    [
        dbc.Label("Video Library", html_for="dropdown"),
        dcc.Dropdown(
            id="dropdown_video_lib",
            options=[
                {"label": "Outlet_Americas_1",
                 "value": 'Outlet_Americas_1'},
                {"label": "Outlet_Americas_2",
                 "value": 'Outlet_Americas_2'},
                {"label": "Outlet_Galerias_1",
                 "value": 'Outlet_Galerias_1'},
            ],
            value='Outlet_Americas_1'
        ),
        html.Button('Launch!', id='submit-val-III',
                    n_clicks=0, className="ico_submit"),
    ], className="form_style"
)

card_video_trail = dbc.Card(
    [
        dbc.CardHeader("Quick Heat Trail Analysis",
                       className="title_video_picker"),
        dbc.CardBody(
            [
                video_library
            ]
        ),
        dbc.CardFooter("Please select a video available from the library",
                       className="foot_video_picker"),
    ],
    className="big_form_style_III"
)

final_video_lib = html.Div([

    dbc.Row([
        dbc.Col(dbc.FormGroup([card_video_trail,
                               html.Hr(),
                               html.P("Video Sample"),
                               html.Hr(),
                               html.Div(id='video_player',
                                        className="center_embed"),
                               html.Hr(),
                               html.P("Quick Heat Trail results"),
                               html.Hr(),
                               html.Div(id='image_heat')
                               ]))
    ])
])

video_upload = html.Div([
    dcc.Upload(
        id='upload-video',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select Files')
        ]),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
        # Allow multiple files to be uploaded
        multiple=True
    ),
    html.Div(id='output-video-upload'),
])


# Building Blocks
# ------------------------------------------
# Meta Tags
# ------------------------------------------
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.CYBORG],
    # These meta_tags ensure content is scaled correctly on different devices. Don't Delete!!
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1"}
    ],
    suppress_callback_exceptions=True
)

server = app.server

app.title = 'CV Team-69 DS4A!'

# ------------------------------------------
# Sidebar Component
# ------------------------------------------
# we use the Row and Col components to construct the sidebar header
# it consists of a title, and a toggle, the latter is hidden on large screens
sidebar_header = dbc.Row(
    [
        dbc.Col(dbc.CardImg(
            src='data:image/png;base64,{}'.format(encoded_image.decode()), className="display-4")),
        dbc.Col(
            [
                html.Button(
                    # use the Bootstrap navbar-toggler classes to style
                    html.Span(className="navbar-toggler-icon"),
                    className="navbar-toggler",
                    # the navbar-toggler classes don't set color
                    style={
                        "color": "rgba(0,0,0,.5)",
                        "border-color": "rgba(0,0,0,.1)",
                    },
                    id="navbar-toggle",
                ),
                html.Button(
                    # use the Bootstrap navbar-toggler classes to style
                    html.Span(className="navbar-toggler-icon"),
                    className="navbar-toggler",
                    # the navbar-toggler classes don't set color
                    style={
                        "color": "rgba(0,0,0,.5)",
                        "border-color": "rgba(0,0,0,.1)",
                    },
                    id="sidebar-toggle",
                ),
            ],
            # the column containing the toggle will be only as wide as the
            # toggle, resulting in the toggle being right aligned
            width="auto",
            # vertically align the toggle in the center
            align="center",
        ),
    ]
)

sidebar = html.Div(
    [
        sidebar_header,
        # we wrap the horizontal rule and short blurb in a div that can be
        # hidden on a small screen
        html.Div(
            [
                html.Hr(),
                html.P(
                    "Welcome to the OffCorss CV app!",
                    className="lead",
                ),
            ],
            id="blurb",
        ),
        # use the Collapse component to animate hiding / revealing links
        dbc.Collapse(
            dbc.Nav(
                [
                    dbc.NavLink("Home", href="/page-1",
                                id="page-1-link", className="ico_home"),
                    dbc.NavLink("Quick heat video analysis", href="/page-2",
                                id="page-2-link", className="ico_upload"),
                    # ----------------------------------------------------------------------------------------
                    dbc.Button(
                        "Store's Metrics",
                        className="ico_store",
                        block=True,
                        color="light",
                        id="collapse-button",
                    ),
                    # -------------------------------------------
                    dbc.Collapse(dbc.Nav(
                        [dbc.NavLink("Trends Analysis", href="/page-3", id="page-3-link", className="collap_menu"),
                         dbc.NavLink("Video Analysis results", href="/page-4", id="page-4-link", className="collap_menu"), ],
                        vertical=True,
                        pills=True,
                    ),
                        id="collapse_2",
                    ),
                    # ---------------------------------------r-------------------------------------------------
                    dbc.NavLink("About Us", href="/page-5",
                                id="page-5-link", className="ico_about_"),
                ],
                vertical=True,
                pills=True,
            ),
            id="collapse",
        ),
    ],
    id="sidebar",
)

content = html.Div(id="page-content")
app.layout = html.Div([dcc.Location(id="url"), sidebar, content])

# this callback uses the current pathname to set the active state of the
# corresponding nav link to true, allowing users to tell see page they are on


@app.callback(
    [Output(f"page-{i}-link", "active") for i in range(1, 6)],
    [Input("url", "pathname")],
)
def toggle_active_links(pathname):
    if pathname == "/":
        # Treat page 1 as the homepage / index
        return True, False, False, False, False
    return [pathname == f"/page-{i}" for i in range(1, 6)]


@app.callback(Output("page-content", "children"), [Input("url", "pathname")])
def render_page_content(pathname):
    if pathname in ["/", "/page-1"]:
        return html.Div([card_cover,
                         jumbotron_int
                         ])
    elif pathname == "/page-2":
        return html.Div([
            final_video_lib
            #  video_upload
        ])
    elif pathname == "/page-3":
        return html.Div([card_trends,
                         html.Hr(),
                         dbc.CardColumns(
                             [
                                 dbc.Card(card_count_1,
                                          className="center_cardc"),
                                 dbc.Card(card_count_2,
                                          className="center_cardc")
                             ]
                         ),
                         Graphs_counter])
    elif pathname == "/page-4":
        return html.Div([card_date,
                         html.Hr(),
                         Graphs_tracker])
    elif pathname == "/page-5":
        return html.Div([cards,
                         cards_2,
                         cards_3])
    # If the user tries to reach a different page, return a 404 message
    return dbc.Jumbotron(
        [
            html.H1("404: Not found", className="text-danger"),
            html.Hr(),
            html.P(f"The pathname {pathname} was not recognised..."),
        ]
    )


@app.callback(
    Output("sidebar", "className"),
    [Input("sidebar-toggle", "n_clicks")],
    [State("sidebar", "className")],
)
def toggle_classname(n, classname):
    if n and classname == "":
        return "collapsed"
    return ""


@app.callback(
    Output("collapse", "is_open"),
    [Input("navbar-toggle", "n_clicks")],
    [State("collapse", "is_open")],
)
def toggle_collapse(n, is_open):
    if n:
        return not is_open
    return is_open


@app.callback(
    Output("collapse_2", "is_open"),
    [Input("collapse-button", "n_clicks")],
    [State("collapse_2", "is_open")],
)
def toggle_collapse(n, is_open):
    if n:
        return not is_open
    return is_open


# Hour and minutes callback
# Start
@app.callback(
    Output("number-out", "children"),
    [Input("Start Hour", "value"), Input("Start Minute", "value")],
)
def number_render(dhour, dminute):
    return "Start Hour: {}, Start Minute: {}".format(dhour, dminute)

# End


@app.callback(
    Output("number-out-end", "children"),
    [Input("End Hour", "value"), Input("End Minute", "value")],
)
def number_render(dhour, dminute):
    return "End Hour: {}, End Minute: {}".format(dhour, dminute)

# Date Callback


@app.callback(
    Output('output-container-date-picker-single', 'children'),
    [Input('my-date-picker-single', 'date')])
def update_output(date_value):
    string_prefix = 'You have selected: '
    if date_value is not None:
        date_object = datetime.strptime(date_value.split(' ')[0], '%Y-%m-%d')
        date_string = date_object.strftime('%d, %B, %Y')
        return string_prefix + date_string

# Store Callback


@app.callback(
    Output('output-dropdown-stores', 'children'),
    [Input('dropdown_store', 'value')]
)
def update_output_store(value):
    string_prefix = 'You have selected: '
    if value is not None:
        return 'You have selected "{}"'.format(value)

# Camera Callback


@app.callback(
    Output('output-dropdown-cameras', 'children'),
    [Input('dropdown_cams', 'value')]
)
def update_output_camera(value):
    string_prefix = 'You have selected: '
    if value is not None:
        return 'You have selected "{}"'.format(value)

# Video heat analysis Callback


def parse_contents(contents):
    return html.Div([
        html.Video(src=contents, autoPlay=True, controls=True),
        html.Hr()
    ])


@app.callback([Output('image_heat', 'children'),
               Output('video_player', 'children')],
              [Input('dropdown_video_lib', 'value'),
               Input('submit-val-III', 'n_clicks')])
def update_output(video_name, submit):
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    if 'submit-val-III' in changed_id and video_name is not None:
        # Build paths
        video_name = str(video_name)
        video_file = 'Sample_videos/' + video_name + '.mp4'
        video_file_x = zip(video_file)
        # Play the video
        if video_name == "Outlet_Americas_1":
            video_player = html.Iframe(
                width="560", height="315", src="https://www.youtube.com/embed/fJo9A3OhNfM")
        elif video_name == "Outlet_Americas_2":
            video_player = html.Iframe(
                width="560", height="315", src="https://www.youtube.com/embed/BnH67JXD36k")
        else:
            video_player = html.Iframe(
                width="560", height="315", src="https://www.youtube.com/embed/wBh-mp83nak")
        # Build the analysis
        payload = {}
        files = [('file', open(video_file, 'rb'))]
        headers = {}
        response = requests.request(
            "POST", url+'/uploadHeatmapVideo', headers=headers, data=payload, files=files)
        file = open("sample_image.png", "wb")
        file.write(response.content)
        file.close()
        back_result = 'sample_image.png'
        encoded_result = base64.b64encode(open(back_result, 'rb').read())
        image_heat = dbc.CardImg(
            src='data:image/png;base64,{}'.format(encoded_result.decode()), className="display-4")
        return image_heat, video_player
    else:
        return (no_update, no_update)


# Video analysis results callback
@app.callback([Output('map_traffic', 'figure'),
               Output('lin_traffic', 'figure'),
               Output('heat_traffic', 'figure')],
              [Input("Start Hour", "value"),
               Input("Start Minute", "value"),
               Input("End Hour", "value"),
               Input("End Minute", "value"),
               Input('dropdown_cams', 'value'),
               Input('my-date-picker-single', 'date'),
               Input('submit-val', 'n_clicks')])
def process_form(start_hour, start_minute, end_hour, end_min, cam, date, submit):
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    if 'submit-val' in changed_id:
        store = 'san diego'
        start_hour = str(start_hour)
        start_minute = str(start_minute)
        end_hour = str(end_hour)
        end_min = str(end_min)
        cam = str(cam)
        date = datetime.strptime(date.split(' ')[0], '%Y-%m-%d')
        date = date.strftime('%d/%m/%Y')
        # ----------------------------------------
        imgx = Image.open('Images/Planos_San_Diego_1.jpg')
        imgx = imgx.convert("RGBA")
        # ----------------------------------------
        engine = get_db()
        df = filter_df(engine, store, date, start_hour,
                       start_minute, end_hour, end_min, cam)
        # ------------------------------------
        lin_traffic = visual_count(df)
        map_traffic = print_path(df, imgx)
        heat_traffic = print_hot_spots(df, imgx, 'temp_hot_spot.jpeg')
        return map_traffic, lin_traffic, heat_traffic
    else:
        return (no_update, no_update, no_update)

# Counter analysis results callback


@app.callback([Output('count_week', 'figure'),
               Output('count_in', 'children'),
               Output('count_out', 'children')],
              [Input("Start Hour II", "value"),
               Input("Start Minute II", "value"),
               Input("End Hour II", "value"),
               Input("End Minute II", "value"),
               Input('dropdown_cams_II', 'value'),
               Input('my-date-picker-single-II', 'start_date'),
               Input('my-date-picker-single-II', 'end_date'),
               Input('submit-val-II', 'n_clicks')])
def process_form_II(start_hour, start_minute, end_hour, end_min, cam, start_date, end_date, submit):
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    if 'submit-val-II' in changed_id:
        store = 'san diego'
        # ---------------------------
        start_date = datetime.strptime(start_date.split(' ')[0], '%Y-%m-%d')
        start_date = start_date.strftime('%d/%m/%Y')
        # ---------------------------
        end_date = datetime.strptime(end_date.split(' ')[0], '%Y-%m-%d')
        end_date = end_date.strftime('%d/%m/%Y')
        # ---------------------------
        start_hour = str(start_hour)
        start_minute = str(start_minute)
        end_hour = str(end_hour)
        end_min = str(end_min)
        cam = str(cam)
        # ------------------------------------
        engine = get_db()
        counts_dt = counter_df(engine, store, start_date, end_date,
                               start_hour, start_minute, end_hour, end_min, cam)
        count_week_org = count_plot(
            counts_dt, filter_1='day_of_week_name', filter_2='hour')
        count_week = count_week_org['fig']
        count_in = count_week_org['in']
        count_out = count_week_org['out']
        return count_week, count_in, count_out

    else:
        return (no_update)


if __name__ == "__main__":
    app.run_server(host='0.0.0.0', port='8050', debug=True)

from dash import dcc, html
from plotly import graph_objects as go

from dashboard.application.components.contact_fig import contact_fig
from dashboard.application.components.contact_matrix_figure import contact_matrix_figure
from dashboard.application.components.contact_scatter import contact_scatter
from dashboard.application.components.params import params
from dashboard.application.components.r_eff_plot import fig
from dashboard.application.data.results import model_storage


layout = html.Div(children=[
    html.H1(children='R_eff estimation dashboard'),
    html.Div(
        id='outer_container',
        children=[
            html.Button(
                'Parameter settings',  # filter button
                id='param-settings',
                style={
                    'text-align': 'center', "width": "100%", 'padding': '10px'
                }
            ),
            dcc.Dropdown(
                id='model-selector',
                options=[
                    {'label': k, 'value': k}
                    for k, v in model_storage.items()
                ],
                style={'display': 'block'},
                multi=True,
                value=[]
            ),
            params,
            html.Div(
                id="output-container",
                children=[
                    html.Div(id='infected', style=dict()),
                    html.Div(
                        dcc.Graph(
                            id='r_eff_plot',
                            figure=fig
                        ),
                        style={'display': "inline-block", 'width': '60%', 'zIndex': -1}
                    ),
                    html.Div(
                        dcc.Graph(
                            id='contact_matrix',
                            figure=contact_matrix_figure
                        ),
                        style={'display': "inline-block", 'width': '40%', 'zIndex': -1}
                    ),
                    html.Div(
                        dcc.Graph(
                            id='contact_numbers',
                            figure=contact_fig
                        ),
                        style={'display': 'inline-block', 'width': '57.5%', 'zIndex': -1}
                    ),
                    html.Div(
                        dcc.Graph(
                            id='contact_scatter',
                            figure=contact_scatter
                        ),
                        style={'display': 'inline-block', 'width': '42.5%', 'zIndex': -1}
                    ),
                    html.Div(
                        dcc.Graph(
                            id='recovered',
                            figure=go.Figure(
                                layout=dict(
                                    xaxis=dict(title='Date'),
                                    yaxis=dict(title='Recovered ratio')
                                )
                            )
                        ),
                        style={'display': 'inline-block', 'width': '57.5%', 'zIndex': -1}
                    ),
                    html.Div(
                        dcc.Graph(
                            id='seasonality-fig',
                            figure=go.Figure(
                                layout=dict(
                                    xaxis=dict(title='Date'),
                                    yaxis=dict(title='Seasonality')
                                )
                            )
                        ),
                        style={'display': 'inline-block', 'width': '42.5%', 'zIndex': -1}
                    )
                ],
                style={'display': 'block', 'zIndex': -1}
            )
        ],
        style={
            'position': 'relative'
        }
    ),

])
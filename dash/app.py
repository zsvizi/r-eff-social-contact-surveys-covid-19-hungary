# source venv/bin/activate
# pip install dash, dash_daq

from copy import deepcopy
from datetime import datetime
import os
import sys

import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_daq as daq
from dash.dependencies import Input, Output, State
import dash_html_components as html
import plotly.express as px
import plotly.graph_objects as go

from matplotlib.colors import to_hex
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

cmap = plt.get_cmap('nipy_spectral')

PROJECT_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(PROJECT_PATH)
from src.simulation import Simulation

sim = Simulation(contact_data_json='dynmatrix_step_1d_window_7d_v15_kid_masked_all.json')
contact_data = pd.DataFrame(sim.data.contact_data_json)
contact_data["avg_contactnum"] = contact_data.avg_actual_outside_proxy + contact_data.avg_family

model_storage = {}

sim = Simulation(contact_data_json='dynmatrix_step_1d_window_7d_v15_kid_reduced_all.json')
sim.date_for_calibration = '2020-09-13'
sim.baseline_cm_date = (sim.date_for_calibration, '2020-09-20')

methods = sim.data.reference_r_eff_data["method"].sort_values().unique()
method_mask = sim.data.reference_r_eff_data["method"] == methods[0]

sample_trace = go.Scatter(
    x=[],
    y=[],
    name="simulated R_eff",
    marker=dict(color='black'),
    line=dict(width=4)
)

fig = go.Figure(
    data=[
        go.Scatter(
            x=sim.data.reference_r_eff_data[method_mask]["datetime"],
            y=sim.data.reference_r_eff_data[method_mask]["r_eff"],
            name="estimated R_eff (FT)",
            mode='markers',
            marker=dict(
                size=5,
                cmax=20,
                cmin=0,
                color=sim.data.reference_r_eff_data[method_mask]["pos"],
                colorbar=dict(
                    title="pos. rate (%)"
                ),
                colorscale="oranges",
                reversescale=True,
            )
        ),
        go.Scatter(
            x=sim.data.reference_r_eff_data[method_mask]["datetime"].tolist() +
            sim.data.reference_r_eff_data[method_mask]["datetime"].tolist()[::-1],
            y=sim.data.reference_r_eff_data[method_mask]["ci_lower"].tolist() +
            sim.data.reference_r_eff_data[method_mask]["ci_upper"].tolist()[::-1],
            name="estimated R_eff (FT) CI",
            hoverinfo="skip",
            fillcolor="gray",
            opacity=0.3,
            fill='toself',
            mode='none'
        )
        # ),
        # go.Scatter(
        #     x = [datetime.fromtimestamp(t) for t in sim.data.contact_data_json["start_ts"]],
        #     y = sim.data.contact_data_json["avg_actual_outside_proxy"] + sim.data.contact_data_json["avg_family"],
        #     name = "contactnum"
        # )
    ],
    layout=dict(
        selectdirection="h",
        legend=dict(
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0
        ),
        xaxis_range=[
            sim.data.reference_r_eff_data[method_mask]["datetime"].min(),
            # sim.data.reference_r_eff_data[method_mask]["datetime"].max()
            "2021-01-15"
        ],
        xaxis=dict(
            title="Date"
        ),
        yaxis=dict(
            title="R_eff"
        ),
    )
)

contact_fig = go.Figure(
    data=[
        go.Scatter(
            x=[datetime.fromtimestamp(t) for t in contact_data.start_ts],
            y=contact_data.avg_contactnum,
            name="Contact numbers",
            mode='lines'
        )
    ],
    layout=dict(
        selectdirection="h",
        legend=dict(
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0
        ),
        xaxis=dict(
            title="Date"
        ),
        yaxis=dict(
            title="Average contactnum"
        ),
        xaxis_range=[
            sim.data.reference_r_eff_data[method_mask]["datetime"].min(),
            # sim.data.reference_r_eff_data[method_mask]["datetime"].max()
            "2021-01-15"
        ]
    )
)

contact_scatter = go.Figure(
    layout=dict(
        xaxis=dict(title='R_eff'),
        yaxis=dict(title='Average contactnum')
    )
)

contact_matrix_figure = px.imshow(
    np.zeros((8, 8)),
    x=["0-4", "5-14", "15-29", "30-44", "45-59", "60-69", "70-79", "80+"],
    y=["0-4", "5-14", "15-29", "30-44", "45-59", "60-69", "70-79", "80+"],
    labels={'x': 'Ego age group', 'y': "Contact age group", 'color': "Number of contacts"},
    color_continuous_scale='blues',
    zmin=np.log10(1e-4),
    zmax=np.log10(25),
    origin='lower'
)
contact_matrix_figure.update_traces(
    hovertemplate='Ego age group: %{x} <br>Contact age group: %{y} <br>Number of contacts: %{text:.2f}')
contact_matrix_figure.update_layout(coloraxis_colorbar=dict(
    title="Number of contacts",
    tickvals=np.linspace(-4, np.log10(25), 7),
    ticktext=list(map(lambda i: "%.3f" % i, np.power(10, np.linspace(-4, np.log10(25), 7))))
))
daterange = pd.date_range(start='2020-03-01', end=str(datetime.now().date()), freq='D').map(
    lambda d: str(d.date())).tolist()

params = html.Div(
    id='filter-elements',
    children=[
        html.P('Date range'),  # date range
        dcc.RangeSlider(
            id="datepicker",
            min=0,
            max=len(daterange) - 1,
            step=1,
            value=[daterange.index("2020-03-31"), daterange.index("2021-01-26")],
            marks={0: daterange[0], len(daterange) - 1: daterange[-1]}
        ),
        html.P('Seasonality'),
        dcc.Slider(
            id='seasonality',
            min=0,
            max=1.0,
            step=0.05,
            value=0.8,
            marks=dict(zip(np.linspace(0, 1.0, 11),
                           np.array(np.round(np.linspace(0, 1.0, 11), 1), dtype='str')))
        ),
        html.P('Include recovered as immune'),
        daq.BooleanSwitch(
            id="is_r_eff_calc",
            on=True
        ),
        html.P('Baseline R_0'),
        dcc.Slider(
            id="baseline_r_0",
            min=1,
            max=2.5,
            value=1.3,
            step=0.1,
            marks=dict(zip(np.linspace(1, 2.5, 16),
                           np.array(np.round(np.linspace(1, 2.5, 16), 1), dtype='str')))
        ),
        html.P('Initial R_0'),
        dcc.Slider(
            id="initial_r_0",
            min=1.5,
            max=3.0,
            value=2.5,
            step=0.1,
            marks=dict(zip(np.linspace(1.5, 3.0, 16),
                           np.array(np.round(np.linspace(1.5, 3.0, 16), 1), dtype='str')))
        ),
        html.P('Initial ratio of recovereds'),
        dcc.Slider(
            id='init_ratio_recovered',
            min=0.01,
            max=0.02,
            step=0.001,
            value=0.011,
            marks=dict(zip(np.linspace(0.01, 0.02, 11),
                           np.array(np.round(np.linspace(0.01, 0.02, 11), 4), dtype='str')))
        ),
        html.P('Seasonality function'),
        dcc.Dropdown(
            id='seas_select',
            options=[
                {'label': 'cosine', 'value': 0},
                {'label': 'piecewise linear', 'value': 1},
                {'label': 'truncated cosine', 'value': 2}
            ],
            value=2
        ),
    ],
    style={
        'display': 'none',
        'position': 'relative',
        'width': '100%',
        'background-color': 'white',
        'zIndex': 100
    }
)

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.FLATLY]
)


@app.callback(
    [
        Output("r_eff_plot", "figure"),
        Output("contact_scatter", "figure"),
        Output("infected", "children"),
        Output('recovered', 'figure'),
        Output('seasonality-fig', 'figure')
    ],
    [
        Input('model-selector', 'value')
    ],
    [
        State("r_eff_plot", "figure"),
        State('contact_scatter', 'figure'),
        State('recovered', 'figure'),
        State('seasonality-fig', 'figure')
    ]
)
def plot_updater(values, fig, cs_fig, r_fig, s_fig):
    # clear figs
    cs_fig["data"] = []
    r_fig["data"] = []
    s_fig["data"] = []
    fig["data"] = fig["data"][0:2]

    latent, infected = None, None
    for i, val in enumerate(values[::-1]):
        print(val)
        sim_h = model_storage[val]

        fig["data"].append(go.Scatter())
        fig["data"][-1]["x"] = [datetime.fromtimestamp(t) for t in sim_h.timestamps]
        fig["data"][-1]["y"] = sim_h.r_eff_plot
        fig["data"][-1]["name"] = val

        fig["data"][-1]["marker"]["color"] = to_hex(cmap(0.5 + i / len(fig["data"]) * 0.5))

        bin_edges = np.array(contact_data.start_ts)
        bin_number = np.digitize(sim_h.timestamps, bin_edges)

        temp_agg_r_eff = pd.DataFrame([sim_h.timestamps, sim_h.r_eff_plot, bin_number]).T
        temp_agg_r_eff.columns = ['ts', 'r_eff', 'binnum']
        temp_agg_r_eff = temp_agg_r_eff.groupby('binnum').agg({'ts': 'min', 'r_eff': 'mean'})
        temp_agg_r_eff["contactnum"] = temp_agg_r_eff.index.map(lambda k: contact_data.loc[k]["avg_contactnum"])
        temp_agg_r_eff['date'] = temp_agg_r_eff['ts'].map(lambda t: str(datetime.fromtimestamp(t).date()))

        cs_fig["data"].append(go.Scatter(mode="markers"))
        cs_fig["data"][-1]["x"] = temp_agg_r_eff.r_eff
        cs_fig["data"][-1]["y"] = temp_agg_r_eff.contactnum
        cs_fig["data"][-1]["text"] = temp_agg_r_eff.date
        cs_fig["data"][-1]["hoverinfo"] = 'text'
        cs_fig["data"][-1]["name"] = val

        r_fig["data"].append(go.Scatter(mode="markers"))
        r_fig["data"][-1]["x"] = [datetime.fromtimestamp(t) for t in sim_h.timestamps]
        r_fig["data"][-1]["y"] = sim_h.rec_ratio

        s_fig["data"].append(go.Scatter(mode="markers"))
        s_fig["data"][-1]["x"] = [datetime.fromtimestamp(t) for t in sim_h.timestamps]
        s_fig["data"][-1]["y"] = sim_h.seasonality_values

        for fig_ in [cs_fig, r_fig, s_fig]:
            fig_["data"][-1]["marker"]["color"] = to_hex(cmap(0.5 + i / len(fig["data"]) * 0.5))
            fig_["data"][-1]["showlegend"] = False

        latent = sim_h.init_latent
        infected = sim_h.init_infected

    return fig, cs_fig, f'Latent + Infected at 2020.09.13.: {latent:.0f} + {infected:.0f}', r_fig, s_fig


@app.callback(
    [
        Output('model-selector', 'options'),
        Output('model-selector', 'value')
    ],
    [
        Input("datepicker", "value"),
        Input('seasonality', 'value'),
        Input('is_r_eff_calc', 'on'),
        Input('baseline_r_0', 'value'),
        Input('initial_r_0', 'value'),
        Input('init_ratio_recovered', 'value'),
        Input('seas_select', 'value')
    ]
)
def select_period(datepicker_range, c, is_r_eff_calc, r0,
                  initial_r0, init_ratio_recovered, seas_select):
    start_time = daterange[datepicker_range[0]]
    end_time = daterange[datepicker_range[1]]

    sim.is_r_eff_calc = is_r_eff_calc
    sim.r0 = r0

    sim.initial_r0 = initial_r0
    sim.init_ratio_recovered = init_ratio_recovered
    sim.seasonality_idx = seas_select

    print("Running simulation...")
    print("\tc", c)
    print("\tis_r_eff_calc", is_r_eff_calc)
    print("\tstart", start_time)
    print("\tend", end_time)
    print("\tR_0", r0)

    sim.simulate(
        start_time=start_time,
        end_time=end_time,
        c=c
    )

    sim_to_store = deepcopy(sim)
    label = "simulated R_eff, R_0=%.1f, c=%.1f, immune %s" % (r0, c, str(is_r_eff_calc))
    if sim_to_store.seasonality_idx == 0:
        seas_str = 'cosine'
    elif sim_to_store.seasonality_idx == 1:
        seas_str = 'piecewise linear'
    else:
        seas_str = 'truncated cosine'
    label += \
        ", initial_r0=%.1f, initial ratio=%.3f, seasonality: %s" \
        % (sim_to_store.initial_r0, sim_to_store.init_ratio_recovered, seas_str)
    model_storage[label] = sim_to_store

    options = [
        {'label': k, 'value': k}
        for k, v in model_storage.items()
    ]
    value = [label]
    return [options, value]


@app.callback(
    Output('contact_matrix', 'figure'),
    [Input('r_eff_plot', 'hoverData')],
    [State('contact_matrix', 'figure')]
)
def display_contact_matrix(hoverdata, cm_fig):
    if hoverdata is not None:
        date = hoverdata['points'][0]['x'].split(' ')[0]
        df = pd.DataFrame(sim.data.contact_data_json).set_index("start_date")
        cm_fig['data'][0]['z'] = np.log10(np.array(df.loc[date]['contact_matrix']) + 1e-4)
        cm_fig['data'][0]['text'] = np.array(df.loc[date]['contact_matrix'])
        cm_fig['layout']['title'] = date
    return cm_fig


@app.callback(
    [Output('filter-elements', 'style'),
     Output('param-settings', 'style')],
    [Input('param-settings', 'n_clicks')],
    [State('filter-elements', 'style'), State('param-settings', 'style')]
)
def show_hide_labels(n_clicks, style, button_style):
    print("Callback\tshow_hide_labels")
    if style is None:
        style = {}
    if n_clicks is None or n_clicks % 2 == 0:
        style.update({'display': 'none'})
        button_style['background-color'] = 'white'
        return style, button_style
    else:
        style.update({'display': 'block'})
        button_style['background-color'] = 'lightgrey'
        return style, button_style


app.layout = html.Div(children=[
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

if __name__ == "__main__":
    app.run_server(host='127.0.0.1', port='8050', debug=True)

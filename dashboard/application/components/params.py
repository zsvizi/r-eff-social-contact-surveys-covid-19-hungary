from dash import dcc, html
import dash_daq as daq
import numpy as np

from dashboard.application.data.results import daterange


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

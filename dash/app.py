import sys

from dash_core_components.Checklist import Checklist
sys.path.insert(0,"/".join(sys.path[0].split("/")[:-1]))

from simulation import Simulation

# source venv/bin/activate
# pip install dash, dash_daq

import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_daq as daq
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
import pandas as pd

from datetime import datetime

sim = Simulation()

sim.time_step = 1
sim.r0 = 2.5
sim.is_r_eff_calc = False
sim.baseline_cm_date = ('2020-08-30', '2020-09-06')

methods = sim.data.reference_r_eff_data["method"].sort_values().unique()

m = methods[0]
method_mask = sim.data.reference_r_eff_data["method"]==m

fig = go.Figure(
    data = [
        go.Scatter(
            x = [],
            y = [],
            name = "simulated R_eff",
            marker = dict(color='black'),
            line = dict(width=4)    
        ),
        go.Scatter(
            x = sim.data.reference_r_eff_data[method_mask]["datetime"],
            y = sim.data.reference_r_eff_data[method_mask]["r_eff"],
            name = "estimated R_eff (FT)",
            mode = 'markers',
            marker = dict(
                size = 5,
                cmax = 20,
                cmin = 0,
                color = sim.data.reference_r_eff_data[method_mask]["pos"],
                colorbar = dict(
                    title = "positivity rate (%)"
                ),
            colorscale = "oranges",
            reversescale = True
            )
        ),
        go.Scatter(
            x = sim.data.reference_r_eff_data[method_mask]["datetime"].tolist() + sim.data.reference_r_eff_data[method_mask]["datetime"].tolist()[::-1],
            y = sim.data.reference_r_eff_data[method_mask]["ci_lower"].tolist() + sim.data.reference_r_eff_data[method_mask]["ci_upper"].tolist()[::-1],
            name = "estimated R_eff (FT) CI",
            hoverinfo = "skip",
            fillcolor = "gray",
            opacity = 0.3,
            fill = 'toself',
            mode = 'none'
        ),
    ]
)

daterange = pd.date_range(start='2020-03-01',end=str(datetime.now().date()),freq='D').map(lambda d: str(d.date())).tolist()

params = html.Div(
    id = 'filter-elements',
    children = [  
        html.P('Date range'), # date range
        dcc.RangeSlider(
            id = "datepicker",
            min = 0,
            max = len(daterange)-1,
            step = 1,
            value = [daterange.index("2020-03-31"),daterange.index("2021-01-26")],
            marks = {0:daterange[0],len(daterange)-1:daterange[-1]}
        ),
        html.P('Seasonality'),
        dcc.Slider(
            id = 'seasonality',
            min = 0,
            max = 0.5,
            step = 0.05,
            value = 0.3
        ),
        html.P('Include recovered as immune'),
        daq.BooleanSwitch(
            id = "is_r_eff_calc",
            on = False
        ),
        html.P('Baseline R_0'),
        dcc.Slider(
            id="baseline_r_0",
            min=2.0,
            max=3.5,
            value=2.5,
            step= 0.1
        )
    ]
)


app = dash.Dash(__name__)

@app.callback(
    Output("example-graph","figure"),
    [
        Input("datepicker","value"),
        Input('seasonality','value'),
        Input('is_r_eff_calc','on'),
        Input('baseline_r_0','value')
    ],
    [State("example-graph","figure")]
)
def select_period(datepicker_range, c, is_r_eff_calc, r0, fig):

    start_time = daterange[datepicker_range[0]]
    end_time = daterange[datepicker_range[1]]

    sim.is_r_eff_calc = is_r_eff_calc
    sim.r0 = r0

    print("Running simulation...")
    print("\tc", c)
    print("\tis_r_eff_calc", is_r_eff_calc)
    print("\tstart", start_time)
    print("\tend", end_time)
    print("\tR_0", r0)

    sim.simulate(
        start_time = start_time,
        end_time = end_time,
        c = c
    )
    print("Done.")


    fig["data"][0]["x"] = [datetime.fromtimestamp(t) for t in sim.timestamps]
    fig["data"][0]["y"] = sim.r_eff_plot

    return fig


app.layout = html.Div(children=[
    html.H1(children='R_eff estimation dashboard'),
    params,
    dcc.Graph(
        id='example-graph',
        figure=fig
    )
])


if __name__ == "__main__":
    app.run_server(host='127.0.0.1', port='8050', debug=True)


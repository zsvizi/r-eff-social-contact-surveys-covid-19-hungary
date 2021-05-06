import sys

from dash_core_components.Checklist import Checklist
from dash_html_components.Div import Div
sys.path.insert(0,"/".join(sys.path[0].split("/")[:-1]))

from simulation import Simulation

# source venv/bin/activate
# pip install dash, dash_daq

import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import dash_daq as daq
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
import plotly.express as px

import pandas as pd
import numpy as np

from datetime import datetime

import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
cmap = plt.get_cmap('Greens')

sim = Simulation(contact_data_json = 'dynmatrix_step_1d_window_7d_v15_kid_reduced_all.json')

sim.time_step = 1
sim.r0 = 2.5
sim.is_r_eff_calc = False
sim.baseline_cm_date = ('2020-08-30', '2020-09-06')

methods = sim.data.reference_r_eff_data["method"].sort_values().unique()

m = methods[0]
method_mask = sim.data.reference_r_eff_data["method"]==m

sample_trace = go.Scatter(
            x = [],
            y = [],
            name = "simulated R_eff",
            marker = dict(color='black'),
            line = dict(width=4)    
        )

fig = go.Figure(
    data = [
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
                    title = "pos. rate (%)"
                ),
            colorscale = "oranges",
            reversescale = True,
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
        )
        # ),
        # go.Scatter(
        #     x = [datetime.fromtimestamp(t) for t in sim.data.contact_data_json["start_ts"]],
        #     y = sim.data.contact_data_json["avg_actual_outside_proxy"] + sim.data.contact_data_json["avg_family"],
        #     name = "contactnum"
        # )
    ],
    layout = dict(
        selectdirection = "h",
        legend = dict(
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0
        )
    )
)

contact_matrix_figure = px.imshow(
    np.zeros((8,8)),
    x = ["0-4","5-14","15-29","30-44","45-59","60-69","70-79","80+"],
    y = ["0-4","5-14","15-29","30-44","45-59","60-69","70-79","80+"],
    labels = {'x': 'Ego age group', 'y' : "Contact age group", 'color': "Number of contacts"},
    color_continuous_scale = 'blues',
    zmin=np.log10(1e-4),
    zmax=np.log10(25),
    origin='lower'
)
contact_matrix_figure.update_traces(hovertemplate='Ego age group: %{x} <br>Contact age group: %{y} <br>Number of contacts: %{text:.2f}')
contact_matrix_figure.update_layout(coloraxis_colorbar=dict(
    title="Number of contacts",
    tickvals=np.linspace(-4,np.log10(25),7),
    ticktext=list(map(lambda i: "%.3f" % i,np.power(10,np.linspace(-4,np.log10(25),7))))
))
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
            max = 1.0,
            step = 0.05,
            value = 0.3,
            marks = dict(zip(np.linspace(0,1.0,11),np.array(np.round(np.linspace(0,1.0,11),1),dtype='str')))
        ),
        html.P('Include recovered as immune'),
        daq.BooleanSwitch(
            id = "is_r_eff_calc",
            on = False
        ),
        html.P('Baseline R_0'),
        dcc.Slider(
            id="baseline_r_0",
            min=2,
            max=3.5,
            value=2.5,
            step= 0.1,
            marks = dict(zip(np.linspace(2,3.5,16),np.array(np.round(np.linspace(2,3.5,16),1),dtype='str')))
        )
    ]
)


app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.FLATLY]
)

@app.callback(
    Output("r_eff_plot","figure"),
    [
        Input("datepicker","value"),
        Input('seasonality','value'),
        Input('is_r_eff_calc','on'),
        Input('baseline_r_0','value')
    ],
    [State("r_eff_plot","figure")]
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


    fig["data"].insert(0,sample_trace)
    fig["data"][0]["x"] = [datetime.fromtimestamp(t) for t in sim.timestamps]
    fig["data"][0]["y"] = sim.r_eff_plot
    fig["data"][0]["name"] = "simulated R_eff, R_0=%.1f, c=%.1f, immune %s" % (r0, c, str(is_r_eff_calc))

    for i,t in enumerate(fig["data"][:-3]):
        fig["data"][i]["marker"]["color"] = to_hex(cmap(0.5+i/len(fig["data"])*0.5))


    return fig

@app.callback(
    Output('contact_matrix','figure'),
    [Input('r_eff_plot','hoverData')],
    [State('contact_matrix','figure')]
)
def display_contact_matrix(hoverdata, cm_fig):
    if hoverdata is not None:
        date = hoverdata['points'][0]['x'].split(' ')[0]
        df = pd.DataFrame(sim.data.contact_data_json).set_index("start_date")
        cm_fig['data'][0]['z'] = np.log10(np.array(df.loc[date]['contact_matrix'])+1e-4)
        cm_fig['data'][0]['text'] = np.array(df.loc[date]['contact_matrix'])
        cm_fig['layout']['title'] = date
    return cm_fig


app.layout = html.Div(children=[
    html.H1(children='R_eff estimation dashboard'),
    params,
    html.Div(
        dcc.Graph(
            id='r_eff_plot',
            figure=fig
        ),
                style = dict(display = "inline-block", width='70%')
    ),
    html.Div(
        dcc.Graph(
            id='contact_matrix',
            figure=contact_matrix_figure
        ),
        style = dict(display = "inline-block", width='30%')
    )
])


if __name__ == "__main__":
    app.run_server(host='127.0.0.1', port='8050', debug=True)


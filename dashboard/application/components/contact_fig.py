from datetime import datetime

from plotly import graph_objects as go

from dashboard.application.data.results import contact_data, method_mask, sim


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

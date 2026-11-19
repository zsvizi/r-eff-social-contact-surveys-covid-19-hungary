from plotly import graph_objects as go

from dashboard.application.data.results import method_mask, sim


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

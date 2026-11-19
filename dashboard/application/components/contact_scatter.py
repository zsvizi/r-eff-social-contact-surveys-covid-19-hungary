from plotly import graph_objects as go


contact_scatter = go.Figure(
    layout=dict(
        xaxis=dict(title='R_eff'),
        yaxis=dict(title='Average contactnum')
    )
)

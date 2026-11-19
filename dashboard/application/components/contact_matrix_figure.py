import numpy as np
from plotly import express as px


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

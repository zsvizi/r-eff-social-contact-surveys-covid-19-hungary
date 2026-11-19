from pathlib import Path
import sys

import dash
import dash_bootstrap_components as dbc

from dashboard.application.main_layout import layout


# Make `python dash/app.py` work from any current working directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

app = dash.Dash(
    __name__,
    assets_folder=str(PROJECT_ROOT / "dashboard" / "application" / "assets"),
    external_stylesheets=[dbc.themes.FLATLY]
)
app.layout = layout

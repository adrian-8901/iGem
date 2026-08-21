import numpy as np
import plotly.graph_objects as go
import trimesh
from plotly.subplots import make_subplots
from dash import Dash, dcc, html, Input, Output

app = Dash(__name__)

# Load your 3D model (STL, OBJ, PLY, etc.)
mesh = trimesh.load("./sample_bead.stl")

# Basic sanity checks - dissolution math needs a valid closed shape
print("Volume:", mesh.volume)
print("Surface area:", mesh.area)
if mesh.volume < 0:
    volume = abs(mesh.volume)

density = 1.05  # TODO replace with actual density


def mesh_mass(volume, density):
    return volume * density


def scaled_mesh_area(
    V_original, A_original, V_new
):  # TODO: Switch to actual erosion computation model
    if V_new <= 0:
        return 0.0
    s = (V_new / V_original) ** (1 / 3)
    # Area scales with the square of linear scale factor
    return A_original * (s**2)


app.layout = html.Div(
    style={
        "backgroundColor": "#000000",
        "color": "#ffffff",
        "padding": "20px",
        "fontFamily": "sans-serif",
        "minHeight": "100vh",
    },
    children=[
        html.H1(
            "Simulation",
            style={"textAlign": "center", "marginBottom": "30px", "color": "#ffffff"},
        ),
        html.Div([dcc.Graph(id="simulation-graphs")]),
    ],
)


@app.callback(
    Output("simulation-graphs", "figure"),
)
def main_loop():
    V0 = volume
    A0 = mesh.area
    M0 = mesh_mass(V0, density)
    t_max = 60
    dt = 0.1

    steps = int(t_max / dt)
    t_history, M_history, A_history, N_history = [], [], [], []

    M = M0
    N = 0

    # Mass vs time
    for step in range(steps + 1):
        k = 0.05
        t = step * dt
        density_bacteria = 1e6  # CFU released per mg hydrogel dissolved #TODO
        r_growth = 0.3 / 3600  # bacterial growth rate (per hour, placeholder) #TODO
        K_capacity = 1e9  # carrying capacity (CFU), placeholder #TODO
        V_current = M / density
        A_current = scaled_mesh_area(V0, A0, V_current)
        t_history.append(t)
        M_history.append(M)
        A_history.append(A_current)

        if M <= 0:
            dM_dt = 0
        else:
            dM_dt = -k * A_current
        M = max(0.0, M + dM_dt * dt)

        def release_rate():
            return density_bacteria * (-dM_dt)

        dN_dt = release_rate() + r_growth * N * (1 - N / K_capacity)
        N = N + dN_dt * dt
        N_history.append(N)

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Pill mass decay over time", "CFU released over time"),
    )

    fig.update_layout(
        template="plotly_dark",
        height=550,
        showlegend=False,
        paper_bgcolor="#000000",
        plot_bgcolor="#000000",
        margin=dict(l=20, r=20, t=40, b=20),
    )

    fig.add_trace(
        go.Scatter(
            x=t_history,
            y=M_history,
            mode="lines",
            name="Mass Decay",
            line=dict(color="royalblue", width=3),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=t_history,
            y=N_history,
            mode="lines",
            name="CFU Released",
            line=dict(color="limegreen", width=3),
        ),
        row=1,
        col=2,
    )

    fig.update_yaxes(row=1, col=1, title_text="Remaining Pill Mass (mg)")
    fig.update_xaxes(row=1, col=1, title_text="Time (s)")
    fig.update_yaxes(row=1, col=2, title_text="CFU Released")
    fig.update_xaxes(row=1, col=2, title_text="Time (s)")

    return fig


if __name__ == "__main__":
    app.run(debug=True)

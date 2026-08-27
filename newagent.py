from concurrent.futures import ProcessPoolExecutor
import base64
import io
from multiprocessing import freeze_support

from dash import Dash, Input, Output, dcc, html
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.ndimage import distance_transform_edt
from skimage import measure
import trimesh

# Constant Configurations
pitch = 0.05
voxel_size = pitch
density = 1.05  # Density in g/cm^3


# Helper Functions
def mesh_mass(vol, dens):
    return vol * dens


def mesh_area(verts, faces):
    area = 0.0
    for tri in faces:
        v0 = verts[tri[0]]
        v1 = verts[tri[1]]
        v2 = verts[tri[2]]
        area += 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0))
    return area


def mass_and_area_at_depth(distance_fld, erosion_depth, v_size, dens):
    remaining_mask = distance_fld > erosion_depth
    vol = remaining_mask.sum() * (v_size**3)

    if remaining_mask.sum() < 10:
        return mesh_mass(vol, dens), 0.0

    verts, faces, _, _ = measure.marching_cubes(
        distance_fld - erosion_depth, level=0.0, spacing=(v_size,) * 3
    )
    return mesh_mass(vol, dens), mesh_area(verts, faces)


# Dash App Initialization
app = Dash(__name__)

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
            style={
                "textAlign": "center",
                "marginBottom": "20px",
                "color": "#ffffff",
            },
        ),
        # Mandatory Drag and Drop Area
        dcc.Upload(
            id="upload-stl",
            children=html.Div(
                [
                    "Drag and Drop STL File Here to Run Simulation or ",
                    html.A(
                        "Browse File",
                        style={"color": "#00ccff", "textDecoration": "underline"},
                    ),
                ]
            ),
            style={
                "width": "100%",
                "height": "80px",
                "lineHeight": "80px",
                "borderWidth": "2px",
                "borderStyle": "dashed",
                "borderRadius": "8px",
                "borderColor": "#00ccff",
                "textAlign": "center",
                "marginBottom": "25px",
                "backgroundColor": "#111111",
                "cursor": "pointer",
                "fontSize": "16px",
            },
            multiple=False,
        ),
        html.Div(
            id="metrics-display",
            style={
                "display": "flex",
                "justifyContent": "space-around",
                "marginBottom": "25px",
                "backgroundColor": "#111111",
                "padding": "15px",
                "borderRadius": "8px",
            },
        ),
        html.Div([dcc.Graph(id="simulation-graphs")]),
    ],
)


@app.callback(
    [Output("simulation-graphs", "figure"), Output("metrics-display", "children")],
    Input("upload-stl", "contents"),
    prevent_initial_call=False,
)
def run_simulation(contents):
    # Prompt state when no file is uploaded yet
    if contents is None:
        fig = go.Figure()
        fig.update_layout(
            template="plotly_dark",
            height=450,
            paper_bgcolor="#000000",
            plot_bgcolor="#000000",
            annotations=[
                {
                    "text": "Please drag and drop an STL file above to generate simulation.",
                    "xref": "paper",
                    "yref": "paper",
                    "showarrow": False,
                    "font": {"size": 18, "color": "#666666"},
                }
            ],
        )

        empty_metrics = [
            html.Div(
                [
                    html.H4(
                        "Initial Mass", style={"margin": "0 0 5px 0", "color": "#aaa"}
                    ),
                    html.H2("-- mg", style={"margin": "0"}),
                ]
            ),
            html.Div(
                [
                    html.H4(
                        "Min Wall Thickness (6h)",
                        style={"margin": "0 0 5px 0", "color": "#aaa"},
                    ),
                    html.H2("-- µm", style={"margin": "0", "color": "#ffaa00"}),
                ]
            ),
            html.Div(
                [
                    html.H4(
                        "Premature Loss (Pre-6h)",
                        style={"margin": "0 0 5px 0", "color": "#aaa"},
                    ),
                    html.H2("-- %", style={"margin": "0"}),
                ]
            ),
            html.Div(
                [
                    html.H4(
                        "Colon Released CFU",
                        style={"margin": "0 0 5px 0", "color": "#aaa"},
                    ),
                    html.H2("-- CFU", style={"margin": "0"}),
                ]
            ),
            html.Div(
                [
                    html.H4(
                        "Colon Delivery Score",
                        style={"margin": "0 0 5px 0", "color": "#aaa"},
                    ),
                    html.H2("--", style={"margin": "0", "color": "#00ccff"}),
                ]
            ),
        ]
        return fig, empty_metrics

    # Parse and compute ONCE per upload
    try:
        content_type, content_string = contents.split(",")
        decoded = base64.b64decode(content_string)
        mesh = trimesh.load(io.BytesIO(decoded), file_type="stl")

        if mesh.is_empty or len(mesh.vertices) == 0:
            raise ValueError("Uploaded STL file is empty or corrupt.")

        volume = abs(mesh.volume)
        voxel_grid = mesh.voxelized(pitch).fill()
        solid_mask = voxel_grid.matrix
        distance_field = distance_transform_edt(solid_mask, sampling=voxel_size)

        max_depth = distance_field.max() * 1.05
        d_samples = np.linspace(0, max_depth, 200)

        with ProcessPoolExecutor() as executor:
            results = list(
                executor.map(
                    mass_and_area_at_depth,
                    [distance_field] * 200,
                    d_samples,
                    [voxel_size] * 200,
                    [density] * 200,
                )
            )

        m_samples = np.array([r[0] for r in results])
        a_samples = np.array([r[1] for r in results])

        m_of_depth = lambda d: np.interp(d, d_samples, m_samples)
        a_of_depth = lambda d: np.interp(d, d_samples, a_samples)

    except Exception as e:
        print(f"Error processing file: {e}")
        return go.Figure(), []

    # Simulation Logic
    v0 = volume
    a0 = mesh.area
    m0 = m_of_depth(0)

    t_max = 14.0
    dt = 0.1
    steps = int(t_max / dt)

    target_t_full = 12.0
    k = 3 * density * v0 / (a0 * target_t_full)

    t_colon = 6.0
    req_wall_thickness_mm = (k / density) * t_colon
    req_wall_thickness_um = req_wall_thickness_mm * 1000.0

    density_bacteria = 1e6
    r_growth = 0.3
    k_capacity = 1e9

    t_history, percentage_history, n_history = [], [], []
    percentage_history_wall, n_history_wall = [], []

    n = 0.0
    n_wall = 0.0
    depth = 0.0

    m_at_t6 = m0
    n_at_t6 = 0.0

    for step in range(steps + 1):
        t = step * dt

        # Without Wall
        m = m_of_depth(depth)
        a_current = a_of_depth(depth)

        if abs(t - t_colon) < (dt / 2.0):
            m_at_t6 = m
            n_at_t6 = n

        dm_dt = -k * a_current if m > 0 else 0.0
        release_rate = density_bacteria * max(0.0, -dm_dt)
        dn_dt = release_rate + r_growth * n * (1.0 - n / k_capacity)
        n = max(0.0, n + dn_dt * dt)

        t_history.append(t)
        n_history.append(n)
        percentage_history.append(((m0 - m) / m0) * 100 if m0 > 0 else 0.0)

        # With Wall
        d_core = max(0.0, depth - req_wall_thickness_mm)
        m_wall = m_of_depth(d_core)
        a_wall = a_of_depth(d_core)

        dm_dt_wall = (
            -k * a_wall if (m_wall > 0 and depth >= req_wall_thickness_mm) else 0.0
        )
        release_rate_wall = density_bacteria * max(0.0, -dm_dt_wall)
        dn_dt_wall = release_rate_wall + r_growth * n_wall * (1.0 - n_wall / k_capacity)
        n_wall = max(0.0, n_wall + dn_dt_wall * dt)

        percentage_history_wall.append(((m0 - m_wall) / m0) * 100 if m0 > 0 else 0.0)
        n_history_wall.append(n_wall)

        depth += (k / density) * dt

    premature_loss_pct = ((m0 - m_at_t6) / m0) * 100 if m0 > 0 else 0.0
    colon_delivered_cfu = n - n_at_t6
    colon_delivery_score = max(0.0, (100.0 - premature_loss_pct) / 100.0) * (
        colon_delivered_cfu / k_capacity
    )

    metrics_html = [
        html.Div(
            [
                html.H4("Initial Mass", style={"margin": "0 0 5px 0", "color": "#aaa"}),
                html.H2(f"{m0:.2f} mg", style={"margin": "0"}),
            ]
        ),
        html.Div(
            [
                html.H4(
                    "Min Wall Thickness (6h)",
                    style={"margin": "0 0 5px 0", "color": "#aaa"},
                ),
                html.H2(
                    f"{req_wall_thickness_um:.2f} µm",
                    style={"margin": "0", "color": "#ffaa00"},
                ),
            ]
        ),
        html.Div(
            [
                html.H4(
                    "Premature Loss (Pre-6h)",
                    style={"margin": "0 0 5px 0", "color": "#aaa"},
                ),
                html.H2(f"{premature_loss_pct:.2f}%", style={"margin": "0"}),
            ]
        ),
        html.Div(
            [
                html.H4(
                    "Colon Released CFU", style={"margin": "0 0 5px 0", "color": "#aaa"}
                ),
                html.H2(f"{colon_delivered_cfu:.2e}", style={"margin": "0"}),
            ]
        ),
        html.Div(
            [
                html.H4(
                    "Colon Delivery Score",
                    style={"margin": "0 0 5px 0", "color": "#aaa"},
                ),
                html.H2(
                    f"{colon_delivery_score:.4f}",
                    style={"margin": "0", "color": "#00ccff"},
                ),
            ]
        ),
    ]

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("% released over time", "CFU released over time"),
    )

    fig.update_layout(
        template="plotly_dark",
        height=550,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        paper_bgcolor="#000000",
        plot_bgcolor="#000000",
        margin=dict(l=20, r=20, t=60, b=20),
    )

    # % Released
    fig.add_trace(
        go.Scatter(
            x=t_history,
            y=percentage_history,
            mode="lines",
            name="No Wall",
            line=dict(color="royalblue", width=3),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=t_history,
            y=percentage_history_wall,
            mode="lines",
            name="With Wall",
            line=dict(color="gold", width=3, dash="dash"),
        ),
        row=1,
        col=1,
    )

    # CFU Released
    fig.add_trace(
        go.Scatter(
            x=t_history,
            y=n_history,
            mode="lines",
            name="No Wall",
            line=dict(color="limegreen", width=3),
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=t_history,
            y=n_history_wall,
            mode="lines",
            name="With Wall",
            line=dict(color="orange", width=3, dash="dash"),
        ),
        row=1,
        col=2,
    )

    fig.add_vline(x=t_colon, line_dash="dash", line_color="red", row=1, col=1)
    fig.add_vline(x=t_colon, line_dash="dash", line_color="red", row=1, col=2)

    fig.update_yaxes(row=1, col=1, title_text="% released")
    fig.update_xaxes(row=1, col=1, title_text="Time (h)")
    fig.update_yaxes(row=1, col=2, title_text="CFU Released")
    fig.update_xaxes(row=1, col=2, title_text="Time (h)")

    return fig, metrics_html


if __name__ == "__main__":
    freeze_support()
    app.run(debug=True, use_reloader=False)

from concurrent.futures import ProcessPoolExecutor
import base64
import io
import os
import time
from multiprocessing import freeze_support

from dash import Dash, Input, Output, State, dcc, html
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.ndimage import distance_transform_edt
from skimage import measure
import trimesh

# Global State & Configurations
file_name = ""
cache_file = f"{file_name}.npz"

pitch = 0.05
voxel_size = pitch
density = 1.05  # Density in g/cm^3


# Global Functions
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


def get_depth_lookup(distance_fld, v_size, dens, n_samples=200):
    if os.path.exists(cache_file):
        print(f"Loading cached depth lookup from {cache_file}...")
        data = np.load(cache_file)
        depths, m_samples, a_samples = (
            data["depths"],
            data["M_samples"],
            data["A_samples"],
        )
    else:
        print("No cache found. Building depth lookup...")
        max_depth = distance_fld.max() * 1.05
        depths = np.linspace(0, max_depth, n_samples)

        with ProcessPoolExecutor() as executor:
            results = list(
                executor.map(
                    mass_and_area_at_depth,
                    [distance_fld] * n_samples,
                    depths,
                    [v_size] * n_samples,
                    [dens] * n_samples,
                )
            )
        m_samples = np.array([r[0] for r in results])
        a_samples = np.array([r[1] for r in results])
        np.savez(cache_file, depths=depths, M_samples=m_samples, A_samples=a_samples)

    return (
        lambda d: np.interp(d, depths, m_samples),
        lambda d: np.interp(d, depths, a_samples),
    )


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
        # Drag and Drop STL Upload Box
        dcc.Upload(
            id="upload-stl",
            children=html.Div(
                [
                    "Drag and Drop STL File Here or ",
                    html.A(
                        "Select File",
                        style={"color": "#00ccff", "textDecoration": "underline"},
                    ),
                ]
            ),
            style={
                "width": "100%",
                "height": "70px",
                "lineHeight": "70px",
                "borderWidth": "2px",
                "borderStyle": "dashed",
                "borderRadius": "8px",
                "borderColor": "#444444",
                "textAlign": "center",
                "marginBottom": "25px",
                "backgroundColor": "#111111",
                "cursor": "pointer",
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
        dcc.Interval(id="startup-trigger", interval=1, n_intervals=0, max_intervals=1),
        html.Div([dcc.Graph(id="simulation-graphs")]),
    ],
)


@app.callback(
    [Output("simulation-graphs", "figure"), Output("metrics-display", "children")],
    [Input("upload-stl", "contents")],
    State("upload-stl", "filename"),
)
def main_loop(contents, filename):
    # Precomputation / File Loading
    mesh = (
        trimesh.load(f"./{file_name}")
        if os.path.exists(f"./{file_name}")
        else trimesh.creation.box(extents=(1, 1, 1))
    )
    volume = abs(mesh.volume)

    voxel_grid = mesh.voxelized(pitch).fill()
    solid_mask = voxel_grid.matrix

    global m_of_depth, a_of_depth
    
    distance_field = distance_transform_edt(solid_mask, sampling=voxel_size)

    # Parse uploaded STL file if dropped into dcc.Upload
    if contents is not None:
        try:
            content_type, content_string = contents.split(",")
            decoded = base64.b64decode(content_string)
            uploaded_mesh = trimesh.load(io.BytesIO(decoded), file_type="stl")

            if not uploaded_mesh.is_empty and len(uploaded_mesh.vertices) > 0:
                mesh = uploaded_mesh
                volume = abs(mesh.volume)
                v_grid = mesh.voxelized(pitch).fill()
                s_mask = v_grid.matrix
                d_field = distance_transform_edt(s_mask, sampling=voxel_size)

                # Recompute lookup functions for the newly dropped mesh
                max_d = d_field.max() * 1.05
                d_samples = np.linspace(0, max_d, 200)

                with ProcessPoolExecutor() as executor:
                    results = list(
                        executor.map(
                            mass_and_area_at_depth,
                            [d_field] * 200,
                            d_samples,
                            [voxel_size] * 200,
                            [density] * 200,
                        )
                    )
                m_s = np.array([r[0] for r in results])
                a_s = np.array([r[1] for r in results])

                m_of_depth = lambda d: np.interp(d, d_samples, m_s)
                a_of_depth = lambda d: np.interp(d, d_samples, a_s)
        except Exception as e:
            print(f"Error processing uploaded STL file: {e}")

    if m_of_depth is None or a_of_depth is None:
        return go.Figure(), []

    v0 = volume
    a0 = mesh.area
    m0 = m_of_depth(0)

    t_max = 14.0
    dt = 0.1
    steps = int(t_max / dt)

    target_t_full = 12.0
    k = 3 * density * v0 / (a0 * target_t_full)

    # Targeted colon release settings (t = 6h)
    t_colon = 6.0
    req_wall_thickness_mm = (k / density) * t_colon
    req_wall_thickness_um = req_wall_thickness_mm * 1000.0

    density_bacteria = 1e6  # CFU released per mg hydrogel dissolved
    r_growth = 0.3  # Bacterial growth rate per hour
    k_capacity = 1e9  # Carrying capacity (CFU)

    t_history, percentage_history, n_history = [], [], []
    percentage_history_wall, n_history_wall = [], []

    n = 0.0
    n_wall = 0.0
    depth = 0.0

    m_at_t6 = m0
    n_at_t6 = 0.0

    for step in range(steps + 1):
        t = step * dt

        # --- WITHOUT WALL ---
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

        # --- WITH PROTECTIVE WALL ---
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

    # Calculation of Colon Delivery Score & premature mass loss
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

    # Subplot 1: % Released
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

    # Subplot 2: CFU Released
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

    # Indicator lines for t = 6h targeted colon entry
    fig.add_vline(x=t_colon, line_dash="dash", line_color="red", row=1, col=1)
    fig.add_vline(x=t_colon, line_dash="dash", line_color="red", row=1, col=2)

    fig.update_yaxes(row=1, col=1, title_text="% released")
    fig.update_xaxes(row=1, col=1, title_text="Time (h)")
    fig.update_yaxes(row=1, col=2, title_text="CFU Released")
    fig.update_xaxes(row=1, col=2, title_text="Time (h)")

    return fig, metrics_html


if __name__ == "__main__":
    freeze_support()
    overall_start = time.perf_counter()

    m_of_depth, a_of_depth = get_depth_lookup(distance_field, voxel_size, density)

    overall_end = time.perf_counter()
    print(f"Computation Time: {overall_end - overall_start:.2f} seconds.")

    app.run(debug=True, use_reloader=False)

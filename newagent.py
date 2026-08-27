from concurrent.futures import ProcessPoolExecutor
import os
import time
from multiprocessing import freeze_support

from dash import Dash, Input, Output, dcc, html
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.ndimage import distance_transform_edt
from skimage import measure
import trimesh

# Global State & Configurations
file_name = "sample_gyroid.stl"
cache_file = f"{file_name}.npz"

pitch = 0.05
voxel_size = pitch
density = 1.05  # Density in g/cm^3

# Precomputation / File Loading
mesh = (
    trimesh.load(f"./{file_name}")
    if os.path.exists(f"./{file_name}")
    else trimesh.creation.box(extents=(1, 1, 1))
)
volume = abs(mesh.volume)

voxel_grid = mesh.voxelized(pitch).fill()
solid_mask = voxel_grid.matrix

distance_field = distance_transform_edt(solid_mask, sampling=voxel_size)


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
                "marginBottom": "30px",
                "color": "#ffffff",
            },
        ),
        dcc.Interval(id="startup-trigger", interval=1, n_intervals=0, max_intervals=1),
        html.Div([dcc.Graph(id="simulation-graphs")]),
    ],
)


@app.callback(
    Output("simulation-graphs", "figure"),
    Input("startup-trigger", "n_intervals"),
)
def main_loop(n_intervals):
    if m_of_depth is None or a_of_depth is None:
        return go.Figure()

    v0 = volume
    a0 = mesh.area
    m0 = m_of_depth(0)

    t_max = 14.0
    dt = 0.1
    steps = int(t_max / dt)

    target_t_full = 12.0
    k = 3 * density * v0 / (a0 * target_t_full)

    density_bacteria = 1e6  # CFU released per mg hydrogel dissolved
    r_growth = 0.3  # Bacterial growth rate per hour
    k_capacity = 1e9  # Carrying capacity (CFU)

    t_history, percentage_history, n_history = [], [], []

    n = 0.0
    depth = 0.0

    for step in range(steps + 1):
        t = step * dt

        m = m_of_depth(depth)
        a_current = a_of_depth(depth)

        dm_dt = -k * a_current if m > 0 else 0.0
        release_rate = density_bacteria * max(0.0, -dm_dt)
        dn_dt = release_rate + r_growth * n * (1.0 - n / k_capacity)

        n = max(0.0, n + dn_dt * dt)

        t_history.append(t)
        n_history.append(n)
        percentage_history.append(((m0 - m) / m0) * 100 if m0 > 0 else 0.0)

        depth += (k / density) * dt

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("% released over time", "CFU released over time"),
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
            y=percentage_history,
            mode="lines",
            name="% released",
            line=dict(color="royalblue", width=3),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=t_history,
            y=n_history,
            mode="lines",
            name="CFU Released",
            line=dict(color="limegreen", width=3),
        ),
        row=1,
        col=2,
    )

    fig.update_yaxes(row=1, col=1, title_text="% released")
    fig.update_xaxes(row=1, col=1, title_text="Time (h)")
    fig.update_yaxes(row=1, col=2, title_text="CFU Released")
    fig.update_xaxes(row=1, col=2, title_text="Time (h)")

    return fig


if __name__ == "__main__":
    freeze_support()
    overall_start = time.perf_counter()

    m_of_depth, a_of_depth = get_depth_lookup(distance_field, voxel_size, density)

    overall_end = time.perf_counter()
    print(
        f"Dash startup precomputations completed in {overall_end - overall_start:.2f} seconds total."
    )

    app.run(debug=True, use_reloader=False)

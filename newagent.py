from concurrent.futures import ProcessPoolExecutor
import base64
import hashlib
import io
from multiprocessing import freeze_support
import traceback

from dash import Dash, Input, Output, dcc, html
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.ndimage import distance_transform_edt
from skimage import measure
import trimesh

# Constant Configurations
pitch = (
    0.05  # voxel size in mm (0.025 resolves 300 um filaments better, but is ~8x slower)
)
voxel_size = pitch
# Printed core: voxel size follows the filament diameter (12 voxels across a filament puts the
# lattice area within ~10% of the ideal; 6 voxels, i.e. 50 um, under-resolves it by ~35%), capped
# so large parts stay tractable.
PRINTED_VOXELS_PER_FILAMENT = 12
MAX_PRINTED_VOXELS = 12e6
density = 1.05  # Density in g/cm^3 (= mg/mm^3)

T_COLON = 6.0  # h, arrival in colon
T_MAX = 14.0  # h, end of simulation
DT = 0.1  # h, time step
T_REF_FULL = 12.0  # h, a sphere with the same V/A fully erodes in this time (front-speed calibration)
CORE_STRUCTURE = (
    "printed"  # "printed" = 3D-printed filament lattice, "solid" = STL as is
)
FILAMENT_UM = 300  # printed filament diameter (um)
FRONT_SPEED_SCALES = np.array(
    [0.6, 0.8, 1.0, 1.25, 1.6]
)  # erosion-rate uncertainty for the wall recommendation


# ----------------------------------------------------------------------------------------------
# Geometry
# ----------------------------------------------------------------------------------------------
def voxelize_centers(mesh, pix, pad=2):
    """Center-sampled solid voxelization via z-ray parity.

    A voxel is solid if its centre is inside the mesh. (trimesh's voxelized().fill() marks every
    voxel the surface touches, which dilates the shape by ~0.6 voxel: +7.6% volume at 50 um.)
    Returns (bool array, origin) where origin is the corner of voxel [0, 0, 0].
    """
    tri = mesh.triangles.astype(np.float64)
    lo = mesh.bounds[0] - pad * pix
    hi = mesh.bounds[1] + pad * pix
    shape = np.ceil((hi - lo) / pix).astype(int) + 1
    jit = np.array([1.2345e-6, 2.3456e-6]) * pix  # keeps rays off shared edges/vertices
    p = tri - lo
    xy = p[:, :, :2] - jit
    a, b, c = xy[:, 0], xy[:, 1], xy[:, 2]
    det = (b[:, 1] - c[:, 1]) * (a[:, 0] - c[:, 0]) + (c[:, 0] - b[:, 0]) * (
        a[:, 1] - c[:, 1]
    )
    ok = np.abs(det) > 1e-14  # skips degenerate and vertical triangles
    p, xy, det = p[ok], xy[ok], det[ok]
    a, b, c = xy[:, 0], xy[:, 1], xy[:, 2]
    i0 = np.ceil(xy.min(1) / pix - 0.5).astype(int)
    i1 = np.floor(xy.max(1) / pix - 0.5).astype(int)
    span = (i1 - i0 + 1).clip(min=0)
    crossings = np.zeros(int(np.prod(shape)), dtype=np.int32)
    for da in range(span[:, 0].max()):
        for db in range(span[:, 1].max()):
            idx = np.nonzero((span[:, 0] > da) & (span[:, 1] > db))[0]
            ci = i0[idx, 0] + da
            cj = i0[idx, 1] + db
            px = (ci + 0.5) * pix
            py = (cj + 0.5) * pix
            A, B, C, D = a[idx], b[idx], c[idx], det[idx]
            l1 = (
                (B[:, 1] - C[:, 1]) * (px - C[:, 0])
                + (C[:, 0] - B[:, 0]) * (py - C[:, 1])
            ) / D
            l2 = (
                (C[:, 1] - A[:, 1]) * (px - C[:, 0])
                + (A[:, 0] - C[:, 0]) * (py - C[:, 1])
            ) / D
            l3 = 1 - l1 - l2
            hit = (l1 >= 0) & (l2 >= 0) & (l3 >= 0)
            if not hit.any():
                continue
            idx, ci, cj = idx[hit], ci[hit], cj[hit]
            z = l1[hit] * p[idx, 0, 2] + l2[hit] * p[idx, 1, 2] + l3[hit] * p[idx, 2, 2]
            ck = np.ceil(z / pix - 0.5).astype(
                int
            )  # first voxel centre above the crossing
            good = (
                (ci >= 0)
                & (ci < shape[0])
                & (cj >= 0)
                & (cj < shape[1])
                & (ck >= 0)
                & (ck < shape[2])
            )
            flat = np.ravel_multi_index((ci[good], cj[good], ck[good]), shape)
            crossings += np.bincount(flat, minlength=crossings.size).astype(np.int32)
    solid = (np.cumsum(crossings.reshape(shape), axis=2) % 2) == 1
    return solid, lo


def filament_mask(
    shape, origin, bounds_min, pix, width, spacing, layer_h, alternate=True
):
    """Voxel mask of a 3D-printed filament lattice.

    Filaments have an elliptical cross-section (width x layer_h), centre-to-centre `spacing`, one
    layer per `layer_h` starting at the bed (bounds_min z). Alternating layers run along y / x
    (0/90 deg raster); alternate=False keeps every layer parallel.
    """
    nx, ny, nz = shape
    xs = origin[0] + (np.arange(nx) + 0.5) * pix - bounds_min[0]
    ys = origin[1] + (np.arange(ny) + 0.5) * pix - bounds_min[1]
    zs = origin[2] + (np.arange(nz) + 0.5) * pix - bounds_min[2]
    semi_w, semi_h = width / 2.0, layer_h / 2.0

    def off(coord):  # distance to the nearest line centre
        return coord - (np.floor(coord / spacing) + 0.5) * spacing

    dx, dy = off(xs), off(ys)
    mask = np.zeros(shape, dtype=bool)
    for k, z in enumerate(zs):
        if z < 0:
            continue
        layer = int(np.floor(z / layer_h))
        dz = z - (layer + 0.5) * layer_h
        if abs(dz) > semi_h:
            continue
        r2 = (dz / semi_h) ** 2
        if (not alternate) or layer % 2 == 0:  # lines along y -> offset in x
            mask[:, :, k] = ((dx / semi_w) ** 2 + r2 <= 1.0)[:, None]
        else:  # lines along x -> offset in y
            mask[:, :, k] = ((dy / semi_w) ** 2 + r2 <= 1.0)[None, :]
    return mask


# Worker state: the distance field is sent to each worker ONCE (initializer) instead of once per job
_DF = None
_PIX = None


def _init_worker(distance_fld, pix):
    global _DF, _PIX
    _DF = distance_fld
    _PIX = pix


def mass_and_area_at_depth(erosion_depth):
    remaining_mask = _DF > erosion_depth
    n = int(remaining_mask.sum())
    mass = n * (_PIX**3) * density
    if n < 10:
        return mass, 0.0
    verts, faces, _, _ = measure.marching_cubes(
        _DF - erosion_depth, level=0.0, spacing=(_PIX,) * 3
    )
    return mass, measure.mesh_surface_area(verts, faces)


_TABLE_CACHE = {}


def get_tables(contents, core, filament_um):
    """m(d) and a(d) lookup tables for the chosen core. Cached, so changing the wall or front speed
    in the UI re-runs only the fast simulation, not the geometry."""
    digest = hashlib.md5(contents.encode()).hexdigest()
    key = (digest, core, int(round(filament_um)) if core == "printed" else 0)
    if key in _TABLE_CACHE:
        return _TABLE_CACHE[key]

    _, content_string = contents.split(",")
    mesh = trimesh.load(io.BytesIO(base64.b64decode(content_string)), file_type="stl")
    if mesh.is_empty or len(mesh.vertices) == 0:
        raise ValueError("Uploaded STL file is empty or corrupt.")

    pix = pitch
    if core == "printed":
        d = filament_um / 1000.0
        pix = d / PRINTED_VOXELS_PER_FILAMENT
        # cap the grid size: coarsen the voxels if the part is too large for the target resolution
        grid_voxels = np.prod((mesh.extents + 4 * pix) / pix)
        if grid_voxels > MAX_PRINTED_VOXELS:
            pix *= (grid_voxels / MAX_PRINTED_VOXELS) ** (1.0 / 3.0)
        print(
            f"Printed core: voxel {pix * 1000:.1f} um = {d / pix:.1f} voxels per filament"
        )

    solid, origin = voxelize_centers(mesh, pix)
    envelope_volume = solid.sum() * pix**3
    if core == "printed":
        solid = solid & filament_mask(
            solid.shape, origin, mesh.bounds[0], pix, width=d, spacing=d, layer_h=d
        )
    if solid.sum() < 10:
        raise ValueError(
            "Voxelized geometry is empty; check the STL and filament diameter."
        )

    # Distance from each solid voxel centre to the nearest empty voxel centre overstates depth by
    # ~half a voxel, so subtract it. The padded border makes the surface close at the grid edge.
    distance_field = distance_transform_edt(np.pad(solid, 1), sampling=pix) - pix / 2.0
    d_samples = np.linspace(0.0, distance_field.max() * 1.05, 200)
    with ProcessPoolExecutor(
        initializer=_init_worker, initargs=(distance_field, pix)
    ) as executor:
        results = list(executor.map(mass_and_area_at_depth, d_samples, chunksize=10))

    tables = {
        "d": d_samples,
        "m": np.array([r[0] for r in results]),
        "a": np.array([r[1] for r in results]),
        "max_depth": float(distance_field.max()),
        "mesh_volume": abs(mesh.volume),
        "mesh_area": mesh.area,
        "pitch": pix,
        "envelope_volume": envelope_volume,
    }
    if len(_TABLE_CACHE) >= 8:
        _TABLE_CACHE.pop(next(iter(_TABLE_CACHE)))
    _TABLE_CACHE[key] = tables
    return tables


# ----------------------------------------------------------------------------------------------
# Physiology and scoring
# ----------------------------------------------------------------------------------------------
# Physiological GI Transit pH profile
def get_ph(t):
    if t < 2.0:
        return 1.8  # Gastric fluid
    elif t < 6.0:
        return 6.8  # Small intestinal fluid
    else:
        return 6.2  # Colonic fluid


# pH-dependent scaling multiplier for erosion rate k
def get_ph_multiplier(ph, ph_crit=5.0):
    # Sigmoidal response mimicking carboxylic deprotonation / enteric dissolution
    return 1.0 / (1.0 + np.exp(-2.5 * (ph - ph_crit)))


def zero_order_delivery_score(
    t, release_pct, t_colon=6.0, t_end=14.0, premature_power=1.0
):
    """Colon delivery score that rewards zero-order (constant-rate) release in the colon window.

    score = protection * completeness * linearity
      protection   : (1 - F(t_colon)) ** premature_power. 1 = nothing released before the colon;
                     raise premature_power above 1 to punish early leakage harder.
      completeness : share of the payload still present at t_colon that has been released by t_end.
      linearity    : 1 - 2 * mean|G(t) - ideal(t)| over [t_colon, t_end], where G is the
                     within-window release normalised to 0..1 and ideal is a straight line from
                     0 at t_colon to 1 at t_end. 1 = constant release rate across the whole window,
                     0 = all-at-once burst.
    Returns (score, protection, completeness, linearity), each in [0, 1].
    """
    t = np.asarray(t, dtype=float)
    frac = np.clip(np.asarray(release_pct, dtype=float) / 100.0, 0.0, 1.0)
    f_start = np.interp(t_colon, t, frac)
    f_end = np.interp(t_end, t, frac)
    if f_start >= 1.0 - 1e-9 or f_end - f_start < 1e-6:
        return 0.0, (1.0 - f_start) ** premature_power, 0.0, 0.0
    protection = (1.0 - f_start) ** premature_power
    completeness = (f_end - f_start) / (1.0 - f_start)
    in_window = (t >= t_colon) & (t <= t_end)
    g = np.clip((frac[in_window] - f_start) / (f_end - f_start), 0.0, 1.0)
    ideal = (t[in_window] - t_colon) / (t_end - t_colon)
    linearity = float(np.clip(1.0 - 2.0 * np.mean(np.abs(g - ideal)), 0.0, 1.0))
    return protection * completeness * linearity, protection, completeness, linearity


def zero_release_score(t, release_pct, t_colon=6.0, t_end=14.0, premature_power=2.0):
    """Rewards ZERO release before the colon and complete release by t_end (no shape requirement)."""
    s, _, _, _ = zero_order_delivery_score(
        t, release_pct, t_colon, t_end, premature_power
    )
    t = np.asarray(t, dtype=float)
    frac = np.clip(np.asarray(release_pct, dtype=float) / 100.0, 0.0, 1.0)
    f_start, f_end = np.interp(t_colon, t, frac), np.interp(t_end, t, frac)
    if f_start >= 1.0 - 1e-9:
        return 0.0
    return float(
        ((1.0 - f_start) ** premature_power) * (f_end - f_start) / (1.0 - f_start)
    )


# ----------------------------------------------------------------------------------------------
# Wall-thickness analysis (fast, vectorised; uses only the m(d) table)
# ----------------------------------------------------------------------------------------------
class WallAnalysis:
    """Release curves for a sealed wall of thickness W eroding from outside at front speed v.
    The core only starts losing mass once the front has gone through the wall."""

    def __init__(self, tables):
        self.t = np.arange(int(round(T_MAX / DT)) + 1) * DT
        mult = np.array([get_ph_multiplier(get_ph(x)) for x in self.t])
        # integral of the pH multiplier: front depth = (k/density) * cum
        self.cum = np.concatenate(([0.0], np.cumsum(mult[:-1]) * DT))
        self.d, self.m = tables["d"], tables["m"]
        self.m0 = float(self.m[0])

    def release_pct(self, front_mm_h, wall_mm):
        d_core = np.maximum(0.0, front_mm_h * self.cum - wall_mm)
        return (self.m0 - np.interp(d_core, self.d, self.m)) / self.m0 * 100.0

    def sweep(self, front_ref_mm_h, walls_mm):
        zr = np.array(
            [
                zero_release_score(self.t, self.release_pct(front_ref_mm_h, w))
                for w in walls_mm
            ]
        )
        zo = np.array(
            [
                zero_order_delivery_score(self.t, self.release_pct(front_ref_mm_h, w))[
                    0
                ]
                for w in walls_mm
            ]
        )
        speeds = front_ref_mm_h * FRONT_SPEED_SCALES
        grid = np.array(
            [
                [zero_release_score(self.t, self.release_pct(v, w)) for v in speeds]
                for w in walls_mm
            ]
        )
        # recommended wall: best worst-case zero-release score over the front-speed uncertainty,
        # ties broken by the mean
        best = np.lexsort((grid.mean(1), grid.min(1)))[-1]
        # front-speed window (score >= 0.99) for each wall
        scan = front_ref_mm_h * np.geomspace(0.25, 4.0, 80)
        v_lo, v_hi = [], []
        for w in walls_mm:
            ok = np.array(
                [
                    zero_release_score(self.t, self.release_pct(v, w)) >= 0.99
                    for v in scan
                ]
            )
            v_lo.append(scan[ok].min() if ok.any() else np.nan)
            v_hi.append(scan[ok].max() if ok.any() else np.nan)
        return {
            "zero_release": zr,
            "zero_order": zo,
            "recommended_idx": int(best),
            "worst_case": float(grid.min(1)[best]),
            "mean_case": float(grid.mean(1)[best]),
            "v_lo": np.array(v_lo),
            "v_hi": np.array(v_hi),
        }


# ----------------------------------------------------------------------------------------------
# Dash app
# ----------------------------------------------------------------------------------------------
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
            "Erosion Simulation",
            style={"textAlign": "center", "marginBottom": "20px", "color": "#ffffff"},
        ),
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
                "marginBottom": "20px",
                "backgroundColor": "#111111",
                "cursor": "pointer",
                "fontSize": "16px",
            },
            multiple=False,
        ),
        dcc.Loading(
            id="loading-indicator",
            type="circle",
            color="#00ccff",
            style={"marginTop": "30px"},
            children=[
                html.Div(
                    id="metrics-display",
                    style={
                        "display": "flex",
                        "justifyContent": "space-around",
                        "flexWrap": "wrap",
                        "gap": "12px",
                        "marginBottom": "25px",
                        "backgroundColor": "#111111",
                        "padding": "15px",
                        "borderRadius": "8px",
                    },
                ),
                html.Div([dcc.Graph(id="simulation-graphs")]),
            ],
        ),
    ],
)


def tile(title, value, color=None, sub=None):
    value_style = {"margin": "0"}
    if color:
        value_style["color"] = color
    children = [
        html.H4(title, style={"margin": "0 0 5px 0", "color": "#aaa"}),
        html.H2(value, style=value_style),
    ]
    if sub:
        children.append(
            html.Div(
                sub, style={"fontSize": "12px", "color": "#888", "marginTop": "4px"}
            )
        )
    return html.Div(children)


@app.callback(
    [Output("simulation-graphs", "figure"), Output("metrics-display", "children")],
    Input("upload-stl", "contents"),
    prevent_initial_call=False,
)
def run_simulation(contents):
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
            tile("Initial Mass", "-- mg"),
            tile("Min Wall Thickness (6h)", "-- µm", "#ffaa00"),
            tile("Recommended Wall", "-- µm", "#66dd88"),
            tile("Premature Loss (Pre-6h)", "-- %"),
            tile("Colon Released CFU", "-- CFU"),
            tile("Colon Delivery Score", "--", "#00ccff"),
        ]
        return fig, empty_metrics

    core = CORE_STRUCTURE
    filament_um = float(FILAMENT_UM)

    # Parse and compute geometry tables (cached per upload / core / filament diameter)
    try:
        tables = get_tables(contents, core, filament_um)
        # The erosion front speed is a material property: calibrate it ONCE on the solid geometry
        # (a sphere with the same V/A would fully erode in T_REF_FULL), then reuse it for any core.
        solid_tables = tables if core == "solid" else get_tables(contents, "solid", 0)
        m_s, a_s = solid_tables["m"][0], solid_tables["a"][0]
        front_mm_h = 3.0 * m_s / (a_s * T_REF_FULL) / density
        calib_note = "auto: 12 h sphere calibration"
    except Exception as e:
        print(f"Error processing file: {e}")
        traceback.print_exc()
        return go.Figure(), []

    d_samples, m_samples, a_samples = tables["d"], tables["m"], tables["a"]
    m_of_depth = lambda d: np.interp(d, d_samples, m_samples)
    a_of_depth = lambda d: np.interp(d, d_samples, a_samples)

    m0 = m_of_depth(0)
    v0 = m0 / density  # (mm^3)
    a0 = a_of_depth(0)  # (mm^2)
    if core == "solid":
        print(
            f"Volume: mesh {tables['mesh_volume']:.3f} vs voxel {v0:.3f} mm^3 ({100*(v0/tables['mesh_volume']-1):+.2f}%)"
        )
        print(
            f"Area:   mesh {tables['mesh_area']:.3f} vs voxel {a0:.3f} mm^2 ({100*(a0/tables['mesh_area']-1):+.2f}%)"
        )
    else:
        # A printed lattice is NOT the smooth mesh: compare it with the ideal touching-cylinder
        # lattice (fill pi/4, lateral area/volume 4/d) instead of with the envelope mesh.
        d_mm = filament_um / 1000.0
        print(
            f"Printed lattice: fill {v0 / tables['envelope_volume']:.3f} of the envelope (ideal {np.pi / 4:.3f}); "
            f"area/volume {a0 / v0:.2f} 1/mm (ideal {4 / d_mm:.2f}); "
            f"area = {a0 / tables['mesh_area']:.1f}x the smooth envelope area "
            f"({tables['mesh_area']:.1f} mm^2)"
        )

    t_max, dt, t_colon = T_MAX, DT, T_COLON
    steps = int(t_max / dt)
    k_base = front_mm_h * density  # mg / mm^2 / h

    # Minimum wall for zero release before the colon (integrated front depth, 0 h to 6 h)
    min_wall_mm = 0.0
    for step in range(int(t_colon / dt)):
        min_wall_mm += (k_base * get_ph_multiplier(get_ph(step * dt)) / density) * dt
    min_wall_um = min_wall_mm * 1000.0

    # Wall-thickness sweep: score vs wall, tolerated front-speed range, recommended wall
    analysis = WallAnalysis(tables)
    walls_mm = np.arange(0.0, 2.0001, 0.025)
    sweep = analysis.sweep(front_mm_h, walls_mm)
    rec_wall_mm = float(walls_mm[sweep["recommended_idx"]])
    zo_best_mm = float(walls_mm[int(np.argmax(sweep["zero_order"]))])
    rec_lo, rec_hi = (
        sweep["v_lo"][sweep["recommended_idx"]],
        sweep["v_hi"][sweep["recommended_idx"]],
    )

    wall_mm = rec_wall_mm  # the recommended wall is used for the "with wall" design
    wall_um_used = wall_mm * 1000.0
    layers = wall_um_used / filament_um

    density_bacteria = 1e6
    r_growth = 0.3
    k_capacity = 1e9

    t_history, percentage_history, n_history, ph_history = [], [], [], []
    percentage_history_wall, n_history_wall = [], []

    n = 0.0
    n_wall = 0.0
    depth_no_wall = 0.0
    depth_with_wall = 0.0

    m_at_t6 = m0
    n_at_t6 = 0.0

    for step in range(steps + 1):
        t = step * dt
        ph = get_ph(t)
        k_t = k_base * get_ph_multiplier(ph)

        # --- WITHOUT WALL ---
        m = m_of_depth(depth_no_wall)
        a_current = a_of_depth(depth_no_wall)

        if abs(t - t_colon) < (dt / 2.0):
            m_at_t6 = m
            n_at_t6 = n

        dm_dt = -k_t * a_current if m > 0 else 0.0
        release_rate = density_bacteria * max(0.0, -dm_dt)
        dn_dt = release_rate + r_growth * n * (1.0 - n / k_capacity)
        n = max(0.0, n + dn_dt * dt)

        t_history.append(t)
        n_history.append(n)
        percentage_history.append(((m0 - m) / m0) * 100 if m0 > 0 else 0.0)
        ph_history.append(ph)

        # --- WITH WALL ---
        d_core = max(0.0, depth_with_wall - wall_mm)
        m_wall = m_of_depth(d_core)
        a_wall = a_of_depth(d_core)

        dm_dt_wall = (
            -k_t * a_wall if (m_wall > 0 and depth_with_wall >= wall_mm) else 0.0
        )
        release_rate_wall = density_bacteria * max(0.0, -dm_dt_wall)
        dn_dt_wall = release_rate_wall + r_growth * n_wall * (1.0 - n_wall / k_capacity)
        n_wall = max(0.0, n_wall + dn_dt_wall * dt)

        percentage_history_wall.append(((m0 - m_wall) / m0) * 100 if m0 > 0 else 0.0)
        n_history_wall.append(n_wall)

        # Advance erosion depth dynamically based on local pH
        depth_no_wall += (k_t / density) * dt
        depth_with_wall += (k_t / density) * dt

    premature_loss_pct = ((m0 - m_at_t6) / m0) * 100 if m0 > 0 else 0.0
    colon_delivered_cfu = n - n_at_t6

    # Zero-order colon delivery score, computed from the mass-release curves
    t_arr = np.array(t_history)
    score_nw, _, _, _ = zero_order_delivery_score(
        t_arr, percentage_history, t_colon, t_max
    )
    colon_delivery_score, protection_w, completeness_w, linearity_w = (
        zero_order_delivery_score(t_arr, percentage_history_wall, t_colon, t_max)
    )
    premature_wall_pct = float(np.interp(t_colon, t_arr, percentage_history_wall))

    metrics_html = [
        tile("Initial Mass", f"{m0:.2f} mg", sub=f"{core} core"),
        tile(
            "Min Wall Thickness (6h)",
            f"{min_wall_um:.0f} µm",
            "#ffaa00",
            sub=f"front {front_mm_h*1000:.0f} µm/h ({calib_note})",
        ),
        tile(
            "Recommended Wall",
            f"{rec_wall_mm*1000:.0f} µm",
            "#66dd88",
            sub=(
                f"≈ {rec_wall_mm*1000/filament_um:.1f} layers | ok for {rec_lo*1000:.0f}–{rec_hi*1000:.0f} µm/h"
                if np.isfinite(rec_lo)
                else "no wall reaches full score"
            )
            + f" | zero-order best {zo_best_mm*1000:.0f} µm",
        ),
        tile(
            "Premature Loss (Pre-6h)",
            f"{premature_loss_pct:.2f}%",
            sub=f"no wall | with {wall_um_used:.0f} µm wall: {premature_wall_pct:.2f}%",
        ),
        tile("Colon Released CFU", f"{colon_delivered_cfu:.2e}"),
        tile(
            "Colon Delivery Score",
            f"{colon_delivery_score:.4f}",
            "#00ccff",
            sub=f"wall {wall_um_used:.0f} µm ({layers:.1f} layers) | completeness {completeness_w:.0%} x linearity {linearity_w:.2f} | no wall {score_nw:.3f}",
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
            name=f"With {wall_um_used:.0f} µm Wall",
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
            name="No Wall (CFU)",
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
            name="With Wall (CFU)",
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

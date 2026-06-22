"""
Modul visualisasi penampang geolistrik 2D menggunakan Plotly.
Tampilan menyerupai RES2DINV: filled contour + masking area di luar coverage.
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.interpolate import griddata
from forward_model import ForwardResult, classify_material


# ─── Skala warna geolistrik (identik dengan RES2DINV style) ──────────────────

COLORSCALE_GEO = [
    [0.000, "#1A237E"],   # Biru sangat tua
    [0.080, "#1565C0"],   # Biru tua
    [0.160, "#1976D2"],   # Biru
    [0.250, "#29B6F6"],   # Biru muda
    [0.340, "#00BCD4"],   # Cyan
    [0.430, "#00897B"],   # Teal
    [0.500, "#43A047"],   # Hijau
    [0.570, "#8BC34A"],   # Hijau kuning
    [0.640, "#CDDC39"],   # Lime
    [0.700, "#FDD835"],   # Kuning
    [0.760, "#FFB300"],   # Amber
    [0.820, "#EF9F27"],   # Oranye
    [0.870, "#E64A19"],   # Oranye tua
    [0.920, "#C62828"],   # Merah
    [0.960, "#B71C1C"],   # Merah tua
    [1.000, "#4A1B0C"],   # Coklat tua
]

ARRAY_COLORS = {
    "Wenner-Schlumberger": "#185FA5",
    "Wenner": "#0F6E56",
    "Dipole-Dipole": "#993C1D",
}


# ─── Fungsi bantu ─────────────────────────────────────────────────────────────

def _log_scale(values: np.ndarray):
    v = np.array(values, dtype=float)
    v = np.where(v <= 0, 0.1, v)
    return np.log10(v)


def _make_colorbar_ticks(vmin, vmax, n=8):
    tick_vals = np.linspace(vmin, vmax, n)
    tick_text = [f"{10**v:.0f}" for v in tick_vals]
    return tick_vals.tolist(), tick_text


def _interpolate_to_grid(x, z, values, nx=300, nz=150):
    """
    Interpolasi data titik ke grid reguler untuk filled contour.
    Gunakan linear + fallback nearest untuk area kosong.
    """
    xi = np.linspace(x.min(), x.max(), nx)
    zi = np.linspace(z.min(), z.max(), nz)
    xi_g, zi_g = np.meshgrid(xi, zi)

    grid = griddata(np.column_stack([x, z]), values, (xi_g, zi_g), method="cubic", rescale=True)

    # Fallback linear
    mask_nan = np.isnan(grid)
    if mask_nan.any():
        grid_lin = griddata(np.column_stack([x, z]), values, (xi_g, zi_g), method="linear", rescale=True)
        grid = np.where(mask_nan, grid_lin, grid)

    # Fallback nearest untuk sisa NaN
    mask_nan2 = np.isnan(grid)
    if mask_nan2.any():
        grid_near = griddata(np.column_stack([x, z]), values, (xi_g, zi_g), method="nearest", rescale=True)
        grid = np.where(mask_nan2, grid_near, grid)

    return xi, zi, grid


def _apply_pseudosection_mask(xi, zi, log_grid, x_data, z_data):
    """
    Terapkan masking area di luar coverage pseudosection.
    Bentuk coverage RES2DINV: trapesoid/segitiga — lebar menyempit sesuai kedalaman.

    Untuk setiap kedalaman zi[r], titik x yang valid adalah:
      x_left(z)  = x_min + z * slope_kiri
      x_right(z) = x_max - z * slope_kanan

    Slope dihitung dari konveks hull titik datum.
    """
    xi_g, zi_g = np.meshgrid(xi, zi)

    z_max = z_data.max()
    x_min_data = x_data.min()
    x_max_data = x_data.max()

    # Hitung slope dari titik datum per level kedalaman
    # Gunakan linear fit dari boundary kiri/kanan tiap level n
    unique_z = np.unique(np.round(z_data, 4))
    left_bounds = []
    right_bounds = []

    for uz in unique_z:
        mask = np.abs(z_data - uz) < (uz * 0.05 + 0.01)
        if mask.sum() == 0:
            continue
        left_bounds.append([uz, x_data[mask].min()])
        right_bounds.append([uz, x_data[mask].max()])

    left_bounds = np.array(left_bounds)
    right_bounds = np.array(right_bounds)

    # Fit linear: x_boundary = a + b * z
    if len(left_bounds) >= 2:
        lb_coef = np.polyfit(left_bounds[:, 0], left_bounds[:, 1], 1)
        rb_coef = np.polyfit(right_bounds[:, 0], right_bounds[:, 1], 1)
    else:
        lb_coef = [0, x_min_data]
        rb_coef = [0, x_max_data]

    x_left_grid  = np.polyval(lb_coef, zi_g)
    x_right_grid = np.polyval(rb_coef, zi_g)

    # Tambah sedikit margin agar boundary tidak terpotong terlalu ketat
    margin = (x_max_data - x_min_data) * 0.01
    outside = (xi_g < x_left_grid - margin) | (xi_g > x_right_grid + margin)

    masked = log_grid.copy()
    masked[outside] = np.nan
    return masked


def _build_res2dinv_shape(x_data, z_data):
    """
    Bangun polygon shape RES2DINV (trapesoid/segitiga) sebagai
    SVG path untuk overlay putih di luar area penampang.
    Mengembalikan koordinat dua segitiga pojok kiri-bawah dan kanan-bawah.
    """
    x_min = x_data.min()
    x_max = x_data.max()
    z_min = z_data.min()
    z_max = z_data.max()

    unique_z = np.unique(np.round(z_data, 4))
    left_at_zmax = x_data[np.abs(z_data - unique_z[-1]) < (unique_z[-1] * 0.05 + 0.01)].min()
    right_at_zmax = x_data[np.abs(z_data - unique_z[-1]) < (unique_z[-1] * 0.05 + 0.01)].max()

    return (x_min, x_max, z_min, z_max, left_at_zmax, right_at_zmax)


def _add_mask_shapes(fig, x_data, z_data, bgcolor="white"):
    """
    Tambahkan shapes (segitiga putih) untuk menutupi area
    di luar coverage pseudosection, seperti tampilan RES2DINV.
    """
    x_min = float(x_data.min())
    x_max = float(x_data.max())
    z_min = float(z_data.min())

    unique_z = np.unique(np.round(z_data, 4))
    z_max = float(unique_z[-1])

    # Cari batas kiri dan kanan di kedalaman maksimum
    tol = z_max * 0.05 + 0.01
    deep_mask = np.abs(z_data - unique_z[-1]) < tol
    if deep_mask.sum() > 0:
        left_deep  = float(x_data[deep_mask].min())
        right_deep = float(x_data[deep_mask].max())
    else:
        left_deep  = x_min + (z_max / (z_max - z_min)) * ((x_max - x_min) * 0.25)
        right_deep = x_max - (z_max / (z_max - z_min)) * ((x_max - x_min) * 0.25)

    shapes = []

    # Segitiga kiri bawah: (x_min, z_min) → (left_deep, z_max) → (x_min, z_max)
    shapes.append(dict(
        type="path",
        path=f"M {x_min},{z_min} L {left_deep},{z_max} L {x_min},{z_max} Z",
        fillcolor=bgcolor,
        line=dict(color=bgcolor, width=0),
        layer="above",
    ))

    # Segitiga kanan bawah: (x_max, z_min) → (right_deep, z_max) → (x_max, z_max)
    shapes.append(dict(
        type="path",
        path=f"M {x_max},{z_min} L {right_deep},{z_max} L {x_max},{z_max} Z",
        fillcolor=bgcolor,
        line=dict(color=bgcolor, width=0),
        layer="above",
    ))

    # Garis batas bawah penampang (outline trapesoid)
    shapes.append(dict(
        type="path",
        path=f"M {x_min},{z_min} L {left_deep},{z_max} L {right_deep},{z_max} L {x_max},{z_min}",
        fillcolor="rgba(0,0,0,0)",
        line=dict(color="rgba(80,80,80,0.6)", width=1.2),
        layer="above",
    ))

    return shapes


def _base_layout(title, height, margin=None):
    return dict(
        title=dict(text=title, font=dict(size=13, color="#1a1a1a"), x=0.01),
        height=height,
        margin=margin or dict(l=65, r=90, t=45, b=50),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="Inter, sans-serif", size=11),
        hovermode="closest",
    )


# ─── Pseudosection plot ───────────────────────────────────────────────────────

def plot_pseudosection(
    datum_points: np.ndarray,
    title: str = "Pseudosection Apparent Resistivity",
    colorbar_title: str = "ρa (Ω·m)",
    height: int = 360,
    rho_min: float = None,
    rho_max: float = None,
    show_values: bool = False,
) -> go.Figure:
    """
    Plot pseudosection sebagai filled contour dengan masking RES2DINV-style.
    datum_points: array (N, 3) → [x_mid, pseudo_depth, rho_a]
    """
    if datum_points is None or len(datum_points) == 0:
        fig = go.Figure()
        fig.add_annotation(text="Tidak ada data", xref="paper", yref="paper",
                           x=0.5, y=0.5, showarrow=False)
        return fig

    x   = datum_points[:, 0]
    z   = datum_points[:, 1]
    rho = datum_points[:, 2]

    log_rho = _log_scale(rho)
    vmin = float(_log_scale(np.array([rho_min]))[0]) if rho_min else float(log_rho.min())
    vmax = float(_log_scale(np.array([rho_max]))[0]) if rho_max else float(log_rho.max())
    if vmax <= vmin:
        vmax = vmin + 0.5

    tick_vals, tick_text = _make_colorbar_ticks(vmin, vmax)

    # ── Interpolasi ke grid ──
    try:
        xi, zi, log_grid = _interpolate_to_grid(x, z, log_rho)
    except Exception:
        return _scatter_fallback(datum_points, title, colorbar_title, height, vmin, vmax, tick_vals, tick_text)

    # ── Masking area di luar coverage ──
    log_grid_masked = _apply_pseudosection_mask(xi, zi, log_grid, x, z)

    # ── Hover text ──
    rho_grid = np.power(10.0, np.where(np.isnan(log_grid_masked), np.nan, log_grid_masked))
    xi_g, zi_g = np.meshgrid(xi, zi)
    hover_grid = np.where(
        np.isnan(log_grid_masked),
        "",
        np.vectorize(lambda xv, zv, rv: (
            f"<b>x</b> = {xv:.1f} m<br>"
            f"<b>Pseudo-depth</b> = {zv:.2f} m<br>"
            f"<b>ρa</b> = {rv:.1f} Ω·m<br>"
            f"<b>Litologi</b>: {classify_material(rv)[0]}"
        ))(xi_g, zi_g, rho_grid)
    )

    fig = go.Figure()

    # ── Filled contour utama ──
    fig.add_trace(go.Contour(
        z=log_grid_masked,
        x=xi,
        y=zi,
        colorscale=COLORSCALE_GEO,
        zmin=vmin,
        zmax=vmax,
        contours=dict(
            coloring="fill",
            showlines=True,
            start=vmin,
            end=vmax,
            size=(vmax - vmin) / 16,
        ),
        line=dict(width=0.4, color="rgba(0,0,0,0.15)"),
        colorbar=dict(
            title=dict(text=colorbar_title, side="right"),
            tickvals=tick_vals,
            ticktext=tick_text,
            len=0.92,
            thickness=15,
            outlinewidth=0.5,
        ),
        hovertext=hover_grid,
        hovertemplate="%{hovertext}<extra></extra>",
        connectgaps=False,
        name="",
    ))

    # ── Shapes masking putih (trapesoid pojok) ──
    shapes = _add_mask_shapes(fig, x, z)
    shapes.append(dict(
        type="rect",
        xref="paper", yref="paper",
        x0=0, y0=0, x1=1, y1=0,
        fillcolor="white",
        line=dict(color="white", width=0),
        layer="above",
    ))

    # ── Titik datum kecil sebagai referensi ──
    fig.add_trace(go.Scatter(
        x=x, y=z,
        mode="markers",
        marker=dict(size=2.5, color="rgba(255,255,255,0.5)", symbol="circle",
                    line=dict(width=0.5, color="rgba(0,0,0,0.3)")),
        hoverinfo="skip",
        showlegend=False,
    ))

    if show_values:
        fig.add_trace(go.Scatter(
            x=x, y=z,
            mode="text",
            text=[f"{r:.0f}" for r in rho],
            textfont=dict(size=7, color="rgba(0,0,0,0.7)"),
            hoverinfo="skip",
            showlegend=False,
        ))

    layout = _base_layout(title, height)
    layout["xaxis"] = dict(
        title="Jarak (m)", gridcolor="#ebebeb", zeroline=False,
        range=[float(x.min()) - 0.5, float(x.max()) + 0.5],
        showline=True, linecolor="#999", mirror=True,
    )
    layout["yaxis"] = dict(
        title="Pseudo-depth (m)", autorange="reversed",
        gridcolor="#ebebeb", zeroline=False,
        showline=True, linecolor="#999", mirror=True,
    )
    layout["shapes"] = shapes
    fig.update_layout(**layout)
    return fig


def _scatter_fallback(datum_points, title, colorbar_title, height, vmin, vmax, tick_vals, tick_text):
    x, z, rho = datum_points[:, 0], datum_points[:, 1], datum_points[:, 2]
    log_rho = _log_scale(rho)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=z, mode="markers",
        marker=dict(size=16, color=log_rho, colorscale=COLORSCALE_GEO,
                    cmin=vmin, cmax=vmax, symbol="square",
                    colorbar=dict(title=dict(text=colorbar_title, side="right"),
                                  tickvals=tick_vals, ticktext=tick_text, len=0.9, thickness=14)),
        hovertemplate="x=%{x:.1f} m | z=%{y:.2f} m<extra></extra>",
    ))
    layout = _base_layout(title, height)
    layout["xaxis"] = dict(title="Jarak (m)")
    layout["yaxis"] = dict(title="Pseudo-depth (m)", autorange="reversed")
    fig.update_layout(**layout)
    return fig


# ─── True resistivity section ────────────────────────────────────────────────

def plot_true_section(
    rho_matrix: np.ndarray,
    x_positions: list,
    depths: list,
    title: str = "Penampang True Resistivity 2D",
    height: int = 380,
    rho_min: float = None,
    rho_max: float = None,
) -> go.Figure:
    """Plot penampang true resistivity sebagai filled contour RES2DINV-style."""
    if rho_matrix is None or rho_matrix.size == 0:
        fig = go.Figure()
        fig.add_annotation(text="Tidak ada data model", xref="paper",
                           yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig

    mat = rho_matrix.copy().astype(float)
    mat = np.where(mat <= 0, np.nan, mat)
    log_mat = np.where(np.isnan(mat), np.nan, np.log10(mat))

    vmin = float(np.log10(rho_min)) if rho_min else float(np.nanmin(log_mat))
    vmax = float(np.log10(rho_max)) if rho_max else float(np.nanmax(log_mat))
    if vmax <= vmin:
        vmax = vmin + 0.5

    tick_vals, tick_text = _make_colorbar_ticks(vmin, vmax)

    x_arr = x_positions if len(x_positions) == log_mat.shape[1] else list(range(log_mat.shape[1]))
    z_arr = depths if len(depths) == log_mat.shape[0] else list(range(log_mat.shape[0]))

    # ── Scatter ke grid untuk smooth contour ──
    xi_g, zi_g = np.meshgrid(np.array(x_arr, float), np.array(z_arr, float))
    valid = ~np.isnan(log_mat)
    if valid.sum() >= 4:
        try:
            xi = np.linspace(min(x_arr), max(x_arr), 300)
            zi = np.linspace(min(z_arr), max(z_arr), 150)
            xi_out, zi_out = np.meshgrid(xi, zi)
            log_smooth = griddata(
                np.column_stack([xi_g[valid], zi_g[valid]]),
                log_mat[valid],
                (xi_out, zi_out),
                method="cubic", rescale=True,
            )
            mask_nan = np.isnan(log_smooth)
            if mask_nan.any():
                log_lin = griddata(
                    np.column_stack([xi_g[valid], zi_g[valid]]),
                    log_mat[valid],
                    (xi_out, zi_out),
                    method="linear", rescale=True,
                )
                log_smooth = np.where(mask_nan, log_lin, log_smooth)
            mask_nan2 = np.isnan(log_smooth)
            if mask_nan2.any():
                log_near = griddata(
                    np.column_stack([xi_g[valid], zi_g[valid]]),
                    log_mat[valid],
                    (xi_out, zi_out),
                    method="nearest", rescale=True,
                )
                log_smooth = np.where(mask_nan2, log_near, log_smooth)
            plot_x, plot_z, plot_z_data = xi, zi, log_smooth
        except Exception:
            plot_x, plot_z, plot_z_data = np.array(x_arr), np.array(z_arr), log_mat
    else:
        plot_x, plot_z, plot_z_data = np.array(x_arr), np.array(z_arr), log_mat

    rho_display = np.power(10.0, np.where(np.isnan(plot_z_data), np.nan, plot_z_data))
    xi_out_g, zi_out_g = np.meshgrid(plot_x, plot_z)
    hover = np.where(
        np.isnan(plot_z_data), "",
        np.vectorize(lambda xv, zv, rv: f"x={xv:.1f} m, z={zv:.1f} m, ρ={rv:.1f} Ω·m")(
            xi_out_g, zi_out_g, rho_display)
    )

    fig = go.Figure()
    fig.add_trace(go.Contour(
        z=plot_z_data,
        x=plot_x,
        y=plot_z,
        colorscale=COLORSCALE_GEO,
        zmin=vmin,
        zmax=vmax,
        contours=dict(
            coloring="fill",
            showlines=True,
            start=vmin,
            end=vmax,
            size=(vmax - vmin) / 16,
        ),
        line=dict(width=0.4, color="rgba(0,0,0,0.15)"),
        colorbar=dict(
            title=dict(text="ρ (Ω·m)", side="right"),
            tickvals=tick_vals,
            ticktext=tick_text,
            len=0.92,
            thickness=15,
            outlinewidth=0.5,
        ),
        hovertext=hover,
        hovertemplate="%{hovertext}<extra></extra>",
        connectgaps=False,
        name="",
    ))

    layout = _base_layout(title, height)
    layout["xaxis"] = dict(
        title="Jarak (m)", gridcolor="#ebebeb",
        showline=True, linecolor="#999", mirror=True,
    )
    layout["yaxis"] = dict(
        title="Kedalaman (m)", autorange="reversed",
        gridcolor="#ebebeb",
        showline=True, linecolor="#999", mirror=True,
    )
    fig.update_layout(**layout)
    return fig


# ─── RMS convergence plot ─────────────────────────────────────────────────────

def plot_rms_convergence(rms_history: list) -> go.Figure:
    if not rms_history:
        return go.Figure()

    iters = list(range(1, len(rms_history) + 1))
    colors = ["#D85A30" if v > 5 else "#0F6E56" for v in rms_history]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=iters, y=rms_history,
        fill="tozeroy",
        fillcolor="rgba(24, 95, 165, 0.08)",
        line=dict(color="#185FA5", width=2.5),
        mode="lines+markers",
        marker=dict(size=10, color=colors, line=dict(width=1.5, color="white")),
        name="RMS Error",
        hovertemplate="Iterasi %{x}: RMS = %{y:.2f}%<extra></extra>",
    ))
    fig.add_hline(
        y=5.0, line=dict(color="#D85A30", dash="dash", width=1.5),
        annotation_text="Batas 5%", annotation_position="right",
        annotation_font=dict(color="#D85A30", size=11),
    )
    if rms_history:
        fig.add_annotation(
            x=iters[-1], y=rms_history[-1],
            text=f"  RMS akhir: {rms_history[-1]:.1f}%",
            showarrow=False,
            font=dict(size=11, color="#0F6E56" if rms_history[-1] <= 5 else "#D85A30"),
            xanchor="left",
        )

    fig.update_layout(
        title=dict(text="Konvergensi Iterasi Inversi", font=dict(size=13), x=0.02),
        xaxis=dict(title="Iterasi ke-", tickmode="linear", dtick=1, gridcolor="#f5f5f5"),
        yaxis=dict(title="RMS Error (%)", gridcolor="#f5f5f5", rangemode="tozero"),
        height=260,
        margin=dict(l=55, r=60, t=40, b=45),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="Inter, sans-serif", size=12),
        showlegend=False,
    )
    return fig


# ─── Komparasi multi-panel ────────────────────────────────────────────────────

def _add_contour_subplot(fig, x, z, rho, vmin, vmax, row, col,
                          show_colorbar, cb_y, cb_len, hover_prefix=""):
    """Helper: filled contour + masking ke subplot."""
    log_rho = _log_scale(rho)
    tick_vals, tick_text = _make_colorbar_ticks(vmin, vmax)

    try:
        xi, zi, log_grid = _interpolate_to_grid(x, z, log_rho)
        log_grid = _apply_pseudosection_mask(xi, zi, log_grid, x, z)
    except Exception:
        fig.add_trace(go.Scatter(
            x=x, y=z, mode="markers",
            marker=dict(size=12, symbol="square", color=log_rho,
                        colorscale=COLORSCALE_GEO, cmin=vmin, cmax=vmax,
                        showscale=show_colorbar,
                        colorbar=dict(title="ρa (Ω·m)", tickvals=tick_vals,
                                      ticktext=tick_text, len=cb_len, y=cb_y, thickness=12)),
            hovertemplate=hover_prefix + "x=%{x:.1f}m | z=%{y:.2f}m<extra></extra>",
            showlegend=False,
        ), row=row, col=col)
        return

    fig.add_trace(go.Contour(
        z=log_grid, x=xi, y=zi,
        colorscale=COLORSCALE_GEO,
        zmin=vmin, zmax=vmax,
        contours=dict(coloring="fill", showlines=True,
                      start=vmin, end=vmax, size=(vmax - vmin) / 16),
        line=dict(width=0.4, color="rgba(0,0,0,0.15)"),
        showscale=show_colorbar,
        colorbar=dict(
            title=dict(text="ρa (Ω·m)", side="right"),
            tickvals=tick_vals, ticktext=tick_text,
            len=cb_len, y=cb_y, thickness=12,
        ),
        connectgaps=False,
        hovertemplate=hover_prefix + "x=%{x:.1f}m | z=%{y:.2f}m<extra></extra>",
        name="", showlegend=False,
    ), row=row, col=col)

    # Datum overlay
    fig.add_trace(go.Scatter(
        x=x, y=z, mode="markers",
        marker=dict(size=2, color="rgba(255,255,255,0.4)", symbol="circle",
                    line=dict(width=0.4, color="rgba(0,0,0,0.2)")),
        hoverinfo="skip", showlegend=False,
    ), row=row, col=col)


def plot_comparison_spasi(results: dict, rho_min=None, rho_max=None) -> go.Figure:
    spasi_vals = sorted(results.keys())
    n = len(spasi_vals)
    if n == 0:
        return go.Figure()

    all_rho = np.concatenate([results[s].datum_points[:, 2]
                               for s in spasi_vals
                               if results[s].datum_points is not None and len(results[s].datum_points) > 0])
    all_log = _log_scale(all_rho)
    vmin = float(np.log10(rho_min)) if rho_min else float(all_log.min())
    vmax = float(np.log10(rho_max)) if rho_max else float(all_log.max())

    fig = make_subplots(
        rows=n, cols=1,
        subplot_titles=[f"Spasi a = {s} m — {results[s].array_name}" for s in spasi_vals],
        vertical_spacing=0.08,
    )

    shapes_all = []
    for idx, s in enumerate(spasi_vals):
        res = results[s]
        dp = res.datum_points
        if dp is None or len(dp) == 0:
            continue
        x, z, rho = dp[:, 0], dp[:, 1], dp[:, 2]
        _add_contour_subplot(fig, x, z, rho, vmin, vmax,
                              row=idx + 1, col=1,
                              show_colorbar=(idx == n - 1),
                              cb_y=1 - (idx + 0.5) / n,
                              cb_len=0.9 / n,
                              hover_prefix=f"a={s}m | ")
        fig.update_yaxes(autorange="reversed", title_text="Depth (m)", row=idx + 1, col=1)
        fig.update_xaxes(title_text="Jarak (m)" if idx == n - 1 else "", row=idx + 1, col=1)

    fig.update_layout(
        height=290 * n,
        margin=dict(l=65, r=90, t=50, b=50),
        plot_bgcolor="white", paper_bgcolor="white",
        font=dict(family="Inter, sans-serif", size=11),
        title=dict(text="Komparasi Pseudosection — Variasi Spasi Elektroda",
                   font=dict(size=13), x=0.02),
    )
    return fig


def plot_comparison_array(results: dict, rho_min=None, rho_max=None) -> go.Figure:
    array_names = list(results.keys())
    n = len(array_names)
    if n == 0:
        return go.Figure()

    all_rho = np.concatenate([results[name].datum_points[:, 2]
                               for name in array_names
                               if results[name].datum_points is not None and len(results[name].datum_points) > 0])
    all_log = _log_scale(all_rho)
    vmin = float(np.log10(rho_min)) if rho_min else float(all_log.min())
    vmax = float(np.log10(rho_max)) if rho_max else float(all_log.max())

    fig = make_subplots(
        rows=n, cols=1,
        subplot_titles=[f"{name} — a = {results[name].electrode_spacing} m" for name in array_names],
        vertical_spacing=0.08,
    )

    for idx, name in enumerate(array_names):
        res = results[name]
        dp = res.datum_points
        if dp is None or len(dp) == 0:
            continue
        x, z, rho = dp[:, 0], dp[:, 1], dp[:, 2]
        _add_contour_subplot(fig, x, z, rho, vmin, vmax,
                              row=idx + 1, col=1,
                              show_colorbar=(idx == n - 1),
                              cb_y=1 - (idx + 0.5) / n,
                              cb_len=0.9 / n,
                              hover_prefix=f"{name} | ")
        fig.update_yaxes(autorange="reversed", title_text="Depth (m)", row=idx + 1, col=1)
        fig.update_xaxes(title_text="Jarak (m)" if idx == n - 1 else "", row=idx + 1, col=1)

    fig.update_layout(
        height=290 * n,
        margin=dict(l=65, r=90, t=50, b=50),
        plot_bgcolor="white", paper_bgcolor="white",
        font=dict(family="Inter, sans-serif", size=11),
        title=dict(text="Komparasi Pseudosection — Variasi Konfigurasi Elektroda",
                   font=dict(size=13), x=0.02),
    )
    return fig


def plot_layer_bar(layer_avgs: list) -> go.Figure:
    if not layer_avgs:
        return go.Figure()

    labels = [f"Lap. {d['layer']}\n(z={d['depth']:.1f}m)" for d in layer_avgs]
    values = [d["avg_rho"] for d in layer_avgs]
    materials = [classify_material(v)[0] for v in values]
    bar_colors = [classify_material(v)[1] for v in values]

    fig = go.Figure(go.Bar(
        x=values, y=labels,
        orientation="h",
        marker=dict(color=bar_colors, line=dict(color="white", width=0.5)),
        text=[f"{v:.0f} Ω·m" for v in values],
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>ρ rata-rata = %{x:.1f} Ω·m<extra></extra>",
        customdata=materials,
    ))
    fig.update_layout(
        title=dict(text="Rata-rata Resistivitas per Lapisan", font=dict(size=13), x=0.02),
        xaxis=dict(title="Resistivitas (Ω·m)", type="log", gridcolor="#f0f0f0"),
        yaxis=dict(autorange="reversed"),
        height=230,
        margin=dict(l=90, r=80, t=40, b=40),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="Inter, sans-serif", size=12),
    )
    return fig

"""
PyGMT Map Visualizations for Multi-Fidelity Models

Generates individual map figures using the full rainbow colorscale:
  1. LF observation locations
  2. HF observation locations
  3. Data coverage (LF + HF together)
  4. LF gridded surface
  5. HF gridded observations
  6. Per-model: prediction surface
  7. Per-model: uncertainty (std) surface
  8. Per-model: absolute error at HF points
  9. Per-model: HF - LF discrepancy surface
 10. Per-model: signed LOO residuals
 11. Per-model: two-panel LOO (residuals + std)
 12. Multi-model side-by-side comparison
"""

import os
import tempfile
import numpy as np
import pygmt
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Any, List

# Single temp CPT file reused across all maps
_TEMP_CPT = os.path.join(tempfile.gettempdir(), "_mf_model.cpt")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _region(X_lf: np.ndarray, X_hf: np.ndarray = None,
            pad: float = 0.05) -> List[float]:
    """[xmin, xmax, ymin, ymax] with padding."""
    pts = [X_lf] + ([X_hf] if X_hf is not None else [])
    all_pts = np.vstack(pts)
    x0, x1 = all_pts[:, 0].min(), all_pts[:, 0].max()
    y0, y1 = all_pts[:, 1].min(), all_pts[:, 1].max()
    dx = (x1 - x0) * pad or pad
    dy = (y1 - y0) * pad or pad
    return [x0 - dx, x1 + dx, y0 - dy, y1 + dy]


def _snap_region(region: List[float], n: int = 80):
    """Snap region to exact multiples of spacing (required by nearneighbor)."""
    s = min((region[1] - region[0]) / n,
            (region[3] - region[2]) / n)
    nx = round((region[1] - region[0]) / s)
    ny = round((region[3] - region[2]) / s)
    snapped = [region[0], region[0] + nx * s,
               region[2], region[2] + ny * s]
    return snapped, s


def _grid(x: np.ndarray, y: np.ndarray, z: np.ndarray,
          region: List[float], n: int = 80):
    """Grid scattered (x, y, z) → xarray.DataArray via nearneighbor."""
    snapped, s = _snap_region(region, n)
    data = pd.DataFrame({"x": x, "y": y, "z": z})
    return pygmt.nearneighbor(
        data=data,
        region=snapped,
        spacing=f"{s:.8f}",
        search_radius=f"{2 * s:.8f}",
    )


def _cpt(vmin: float, vmax: float,
         cmap: str = "rainbow", reverse: bool = False) -> str:
    """Write CPT to a known temp file and return its path."""
    if vmax <= vmin:
        vmax = vmin + 1e-6
    pygmt.makecpt(cmap=cmap, series=[vmin, vmax],
                  reverse=reverse, output=_TEMP_CPT)
    return _TEMP_CPT


def _frame() -> List[str]:
    return ["xa", "ya", "WeSn"]


def _projection(region: List[float], width: str = "12c") -> str:
    """Cartesian projection scaled to width, aspect-ratio-correct height."""
    aspect = (region[3] - region[2]) / (region[1] - region[0])
    height = f"{float(width[:-1]) * aspect:.2f}c"
    return f"X{width}/{height}"


# ── 1. LF observation locations ───────────────────────────────────────────────

def plot_lf_locations(X_lf: np.ndarray,
                      save_path: Optional[str] = None) -> pygmt.Figure:
    """Map of LF measurement locations (gray circles)."""
    reg = _region(X_lf, pad=0.05)
    fig = pygmt.Figure()
    fig.basemap(region=reg, projection=_projection(reg), frame=_frame())
    fig.plot(x=X_lf[:, 0], y=X_lf[:, 1],
             style="c0.10c", fill="gray50", pen="0.3p,black")
    fig.text(x=np.mean(reg[:2]), y=reg[3],
             text=f"LF Locations  (n={len(X_lf)})",
             font="12p,Helvetica-Bold", justify="TC", offset="0/0.3c")
    if save_path:
        fig.savefig(save_path, dpi=300)
    return fig


# ── 2. HF observation locations ───────────────────────────────────────────────

def plot_hf_locations(X_hf: np.ndarray,
                      save_path: Optional[str] = None) -> pygmt.Figure:
    """Map of HF measurement locations (red squares)."""
    reg = _region(X_hf, pad=0.10)
    fig = pygmt.Figure()
    fig.basemap(region=reg, projection=_projection(reg), frame=_frame())
    fig.plot(x=X_hf[:, 0], y=X_hf[:, 1],
             style="s0.25c", fill="red", pen="0.5p,black")
    fig.text(x=np.mean(reg[:2]), y=reg[3],
             text=f"HF Locations  (n={len(X_hf)})",
             font="12p,Helvetica-Bold", justify="TC", offset="0/0.3c")
    if save_path:
        fig.savefig(save_path, dpi=300)
    return fig


# ── 3. Data coverage ──────────────────────────────────────────────────────────

def plot_data_coverage(X_lf: np.ndarray, X_hf: np.ndarray,
                       save_path: Optional[str] = None) -> pygmt.Figure:
    """Map showing LF (gray) and HF (red) measurement locations together."""
    reg = _region(X_lf, X_hf, pad=0.05)
    fig = pygmt.Figure()
    fig.basemap(region=reg, projection=_projection(reg), frame=_frame())
    fig.plot(x=X_lf[:, 0], y=X_lf[:, 1],
             style="c0.10c", fill="gray60", pen="0.2p,gray30",
             label=f"LF (n={len(X_lf)})")
    fig.plot(x=X_hf[:, 0], y=X_hf[:, 1],
             style="s0.25c", fill="red", pen="0.6p,black",
             label=f"HF (n={len(X_hf)})")
    fig.legend(position="JBR+jBR+o0.2c", box=True)
    fig.text(x=np.mean(reg[:2]), y=reg[3],
             text="Data Coverage: LF & HF",
             font="12p,Helvetica-Bold", justify="TC", offset="0/0.3c")
    if save_path:
        fig.savefig(save_path, dpi=300)
    return fig


# ── 4. LF gridded surface ─────────────────────────────────────────────────────

def plot_lf_surface(X_lf: np.ndarray, Y_lf: np.ndarray,
                    vmin: float = None, vmax: float = None,
                    cmap: str = "rainbow",
                    save_path: Optional[str] = None) -> pygmt.Figure:
    """Gridded surface of LF observations with full colorscale."""
    Y_lf = np.asarray(Y_lf).flatten()
    reg = _region(X_lf, pad=0.05)
    vmin = vmin if vmin is not None else float(Y_lf.min())
    vmax = vmax if vmax is not None else float(Y_lf.max())

    grd = _grid(X_lf[:, 0], X_lf[:, 1], Y_lf, reg)
    cpt = _cpt(vmin, vmax, cmap)

    fig = pygmt.Figure()
    fig.basemap(region=reg, projection=_projection(reg), frame=_frame())
    fig.grdimage(grid=grd, cmap=cpt, nan_transparent=True)
    fig.colorbar(cmap=cpt, frame=["x+lTemperature (°C)", "y+lLF"])
    fig.plot(x=X_lf[:, 0], y=X_lf[:, 1],
             style="c0.07c", fill="white", pen="0.3p,black")
    fig.text(x=np.mean(reg[:2]), y=reg[3],
             text="LF Observations — Gridded Surface",
             font="12p,Helvetica-Bold", justify="TC", offset="0/0.3c")
    if save_path:
        fig.savefig(save_path, dpi=300)
    return fig


# ── 5. HF colored scatter ─────────────────────────────────────────────────────

def plot_hf_surface(X_hf: np.ndarray, Y_hf: np.ndarray,
                    vmin: float = None, vmax: float = None,
                    cmap: str = "rainbow",
                    save_path: Optional[str] = None) -> pygmt.Figure:
    """Color-coded scatter map of HF observations."""
    Y_hf = np.asarray(Y_hf).flatten()
    reg = _region(X_hf, pad=0.10)
    vmin = vmin if vmin is not None else float(Y_hf.min())
    vmax = vmax if vmax is not None else float(Y_hf.max())

    cpt = _cpt(vmin, vmax, cmap)

    fig = pygmt.Figure()
    fig.basemap(region=reg, projection=_projection(reg), frame=_frame())
    fig.plot(data=np.column_stack([X_hf[:, 0], X_hf[:, 1], Y_hf]),
             style="s0.35c", cmap=cpt, pen="0.5p,black")
    fig.colorbar(cmap=cpt, frame=["x+lTemperature (°C)", "y+lHF"])
    fig.text(x=np.mean(reg[:2]), y=reg[3],
             text="HF Observations",
             font="12p,Helvetica-Bold", justify="TC", offset="0/0.3c")
    if save_path:
        fig.savefig(save_path, dpi=300)
    return fig


# ── 6. Model prediction surface ───────────────────────────────────────────────

def plot_prediction_map(model,
                        X_lf: np.ndarray, X_hf: np.ndarray, Y_hf: np.ndarray,
                        model_name: str = "Model",
                        vmin: float = None, vmax: float = None,
                        cmap: str = "rainbow",
                        n_grid: int = 80,
                        save_path: Optional[str] = None) -> pygmt.Figure:
    """Gridded HF prediction surface from a trained model."""
    reg = _region(X_lf, X_hf, pad=0.05)
    Y_hf = np.asarray(Y_hf).flatten()

    x_lin = np.linspace(reg[0], reg[1], n_grid)
    y_lin = np.linspace(reg[2], reg[3], n_grid)
    Xg, Yg = np.meshgrid(x_lin, y_lin)
    XY = np.column_stack([Xg.ravel(), Yg.ravel()])
    Z, _ = model.predict(XY, return_std=False)
    Z = Z.flatten()

    vmin = vmin if vmin is not None else min(float(Z.min()), float(Y_hf.min()))
    vmax = vmax if vmax is not None else max(float(Z.max()), float(Y_hf.max()))

    grd = _grid(XY[:, 0], XY[:, 1], Z, reg, n=n_grid)
    cpt = _cpt(vmin, vmax, cmap)

    fig = pygmt.Figure()
    fig.basemap(region=reg, projection=_projection(reg), frame=_frame())
    fig.grdimage(grid=grd, cmap=cpt, nan_transparent=True)
    fig.colorbar(cmap=cpt, frame=["x+lTemperature (°C)", f"y+l{model_name}"])
    fig.plot(x=X_hf[:, 0], y=X_hf[:, 1],
             style="s0.25c", fill="white", pen="0.6p,black")
    fig.text(x=np.mean(reg[:2]), y=reg[3],
             text=f"{model_name} — HF Prediction",
             font="12p,Helvetica-Bold", justify="TC", offset="0/0.3c")
    if save_path:
        fig.savefig(save_path, dpi=300)
    return fig


# ── 7. Uncertainty surface ────────────────────────────────────────────────────

def plot_uncertainty_map(model,
                         X_lf: np.ndarray, X_hf: np.ndarray,
                         model_name: str = "Model",
                         cmap: str = "hot",
                         n_grid: int = 80,
                         save_path: Optional[str] = None) -> Optional[pygmt.Figure]:
    """Gridded prediction uncertainty (std) surface."""
    reg = _region(X_lf, X_hf, pad=0.05)

    x_lin = np.linspace(reg[0], reg[1], n_grid)
    y_lin = np.linspace(reg[2], reg[3], n_grid)
    XY = np.column_stack([x.ravel() for x in np.meshgrid(x_lin, y_lin)])
    _, Z_std = model.predict(XY, return_std=True)

    if Z_std is None or np.all(Z_std == 0):
        print(f"    [{model_name}] no uncertainty — skipping.")
        return None

    Z_std = Z_std.flatten()
    grd = _grid(XY[:, 0], XY[:, 1], Z_std, reg, n=n_grid)
    cpt = _cpt(float(Z_std.min()), float(Z_std.max()), cmap)

    fig = pygmt.Figure()
    fig.basemap(region=reg, projection=_projection(reg), frame=_frame())
    fig.grdimage(grid=grd, cmap=cpt, nan_transparent=True)
    fig.colorbar(cmap=cpt, frame=["x+lStd Dev (°C)", f"y+l{model_name}"])
    fig.plot(x=X_hf[:, 0], y=X_hf[:, 1],
             style="s0.25c", fill="white", pen="0.6p,black")
    fig.text(x=np.mean(reg[:2]), y=reg[3],
             text=f"{model_name} — Prediction Uncertainty",
             font="12p,Helvetica-Bold", justify="TC", offset="0/0.3c")
    if save_path:
        fig.savefig(save_path, dpi=300)
    return fig


# ── 8. Absolute error at HF points ────────────────────────────────────────────

def plot_error_map(y_true: np.ndarray, y_pred: np.ndarray,
                   X_hf: np.ndarray,
                   model_name: str = "Model",
                   cmap: str = "hot",
                   save_path: Optional[str] = None) -> pygmt.Figure:
    """Scatter map colored by absolute prediction error at HF locations."""
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    err = np.abs(y_true - y_pred)

    reg = _region(X_hf, pad=0.12)
    cpt = _cpt(0.0, float(err.max()), cmap)

    fig = pygmt.Figure()
    fig.basemap(region=reg, projection=_projection(reg), frame=_frame())
    fig.plot(data=np.column_stack([X_hf[:, 0], X_hf[:, 1], err]),
             style="c0.35c", cmap=cpt, pen="0.5p,black")
    fig.colorbar(cmap=cpt, frame=["x+l|Error| (°C)", f"y+l{model_name}"])
    fig.text(x=np.mean(reg[:2]), y=reg[3],
             text=f"{model_name} — Absolute Error at HF Points",
             font="12p,Helvetica-Bold", justify="TC", offset="0/0.3c")
    if save_path:
        fig.savefig(save_path, dpi=300)
    return fig


# ── 9. HF − LF discrepancy surface ───────────────────────────────────────────

def plot_discrepancy_map(model,
                         X_lf: np.ndarray, X_hf: np.ndarray,
                         model_name: str = "Model",
                         cmap: str = "polar",
                         n_grid: int = 80,
                         save_path: Optional[str] = None) -> Optional[pygmt.Figure]:
    """Gridded map of HF prediction minus LF prediction (model discrepancy)."""
    if not hasattr(model, 'predict_lf'):
        print(f"    [{model_name}] no predict_lf — skipping discrepancy map.")
        return None

    reg = _region(X_lf, X_hf, pad=0.05)
    x_lin = np.linspace(reg[0], reg[1], n_grid)
    y_lin = np.linspace(reg[2], reg[3], n_grid)
    XY = np.column_stack([x.ravel() for x in np.meshgrid(x_lin, y_lin)])

    Z_hf, _ = model.predict(XY, return_std=False)
    Z_lf = model.predict_lf(XY)
    delta = Z_hf.flatten() - Z_lf.flatten()

    lim = float(np.abs(delta).max())
    grd = _grid(XY[:, 0], XY[:, 1], delta, reg, n=n_grid)
    cpt = _cpt(-lim, lim, cmap)

    fig = pygmt.Figure()
    fig.basemap(region=reg, projection=_projection(reg), frame=_frame())
    fig.grdimage(grid=grd, cmap=cpt, nan_transparent=True)
    fig.colorbar(cmap=cpt, frame=["x+ldelta-T (deg C)", "y+lHF-LF"])
    fig.plot(x=X_hf[:, 0], y=X_hf[:, 1],
             style="s0.25c", fill="white", pen="0.6p,black")
    fig.text(x=np.mean(reg[:2]), y=reg[3],
             text=f"{model_name} — HF minus LF Discrepancy",
             font="12p,Helvetica-Bold", justify="TC", offset="0/0.3c")
    if save_path:
        fig.savefig(save_path, dpi=300)
    return fig


# ── 10. Signed LOO residuals ──────────────────────────────────────────────────

def plot_signed_residuals(y_true: np.ndarray, y_pred: np.ndarray,
                          X_hf: np.ndarray,
                          model_name: str = "Model",
                          save_path: Optional[str] = None) -> pygmt.Figure:
    """Scatter map of signed LOO residuals (+overpred / -underpred)."""
    resid = np.asarray(y_pred).flatten() - np.asarray(y_true).flatten()
    lim = float(np.abs(resid).max())

    reg = _region(X_hf, pad=0.12)
    cpt = _cpt(-lim, lim, "polar")

    fig = pygmt.Figure()
    fig.basemap(region=reg, projection=_projection(reg), frame=_frame())
    fig.plot(data=np.column_stack([X_hf[:, 0], X_hf[:, 1], resid]),
             style="c0.35c", cmap=cpt, pen="0.5p,black")
    fig.colorbar(cmap=cpt,
                 frame=["x+lResidual (deg C)", "y+l+overpred / -underpred"])
    fig.text(x=np.mean(reg[:2]), y=reg[3],
             text=f"{model_name} — Signed LOO Residuals",
             font="12p,Helvetica-Bold", justify="TC", offset="0/0.3c")
    if save_path:
        fig.savefig(save_path, dpi=300)
    return fig


# ── 11. Two-panel LOO (residuals + std) ───────────────────────────────────────

def plot_loo_spatial(y_true: np.ndarray, y_pred: np.ndarray,
                     y_std: np.ndarray,
                     X_hf: np.ndarray,
                     model_name: str = "Model",
                     save_path: Optional[str] = None) -> pygmt.Figure:
    """Two-panel map: signed LOO residuals (left) and LOO std (right)."""
    resid = np.asarray(y_pred).flatten() - np.asarray(y_true).flatten()
    lim = float(np.abs(resid).max())

    reg = _region(X_hf, pad=0.15)
    proj = _projection(reg, width="10c")

    cpt_resid = _cpt(-lim, lim, "polar")
    resid_cpt_copy = _TEMP_CPT + ".resid.cpt"
    import shutil
    shutil.copy(_TEMP_CPT, resid_cpt_copy)

    fig = pygmt.Figure()
    with fig.subplot(nrows=1, ncols=2, figsize=("22c", "10c"),
                     sharex="b", sharey="l",
                     title=f"{model_name} — LOO Residuals & Uncertainty"):
        with fig.set_panel(panel=0):
            fig.basemap(region=reg, projection=proj, frame=_frame())
            fig.plot(data=np.column_stack([X_hf[:, 0], X_hf[:, 1], resid]),
                     style="c0.35c", cmap=resid_cpt_copy, pen="0.5p,black")
            fig.colorbar(cmap=resid_cpt_copy, frame=["x+lResidual (deg C)"])

        if y_std is not None and not np.all(np.isnan(y_std)):
            y_std = np.asarray(y_std).flatten()
            cpt_std = _cpt(0.0, float(y_std.max()), "hot")
            with fig.set_panel(panel=1):
                fig.basemap(region=reg, projection=proj, frame=_frame())
                fig.plot(data=np.column_stack([X_hf[:, 0], X_hf[:, 1], y_std]),
                         style="c0.35c", cmap=cpt_std, pen="0.5p,black")
                fig.colorbar(cmap=cpt_std, frame=["x+lStd Dev (deg C)"])

    if save_path:
        fig.savefig(save_path, dpi=300)
    return fig


# ── 12. Multi-model prediction comparison panel ───────────────────────────────

def plot_model_comparison(models: Dict[str, Any],
                          X_lf: np.ndarray, X_hf: np.ndarray, Y_hf: np.ndarray,
                          vmin: float = None, vmax: float = None,
                          cmap: str = "rainbow",
                          n_grid: int = 60,
                          save_path: Optional[str] = None) -> pygmt.Figure:
    """One panel per model, all using the same shared colorscale."""
    reg = _region(X_lf, X_hf, pad=0.05)
    Y_hf = np.asarray(Y_hf).flatten()

    x_lin = np.linspace(reg[0], reg[1], n_grid)
    y_lin = np.linspace(reg[2], reg[3], n_grid)
    XY = np.column_stack([x.ravel() for x in np.meshgrid(x_lin, y_lin)])

    preds = {}
    for name, m in models.items():
        Z, _ = m.predict(XY, return_std=False)
        preds[name] = Z.flatten()

    all_Z = np.concatenate(list(preds.values()))
    vmin = vmin if vmin is not None else min(float(all_Z.min()), float(Y_hf.min()))
    vmax = vmax if vmax is not None else max(float(all_Z.max()), float(Y_hf.max()))
    cpt = _cpt(vmin, vmax, cmap)

    n = len(models)
    proj = _projection(reg, width="10c")

    fig = pygmt.Figure()
    with fig.subplot(nrows=1, ncols=n,
                     figsize=(f"{10 * n}c", "10c"),
                     sharex="b", sharey="l",
                     title="Model Comparison — HF Prediction"):
        for idx, (name, _) in enumerate(models.items()):
            grd = _grid(XY[:, 0], XY[:, 1], preds[name], reg, n=n_grid)
            with fig.set_panel(panel=idx):
                fig.basemap(region=reg, projection=proj, frame=_frame())
                fig.grdimage(grid=grd, cmap=cpt, nan_transparent=True)
                fig.plot(x=X_hf[:, 0], y=X_hf[:, 1],
                         style="s0.22c", fill="white", pen="0.6p,black")
                fig.text(x=np.mean(reg[:2]), y=reg[2],
                         text=name, font="11p,Helvetica-Bold",
                         justify="TC", offset="0/0.3c")

    fig.colorbar(cmap=cpt, frame=["x+lTemperature (deg C)"],
                 position="JBC+w12c+o0/1c+h")
    if save_path:
        fig.savefig(save_path, dpi=300)
    return fig


# ── Master runner ─────────────────────────────────────────────────────────────

def generate_all_maps(models: Dict[str, Any],
                      X_lf: np.ndarray, Y_lf: np.ndarray,
                      X_hf: np.ndarray, Y_hf: np.ndarray,
                      loo_results: Dict[str, Dict] = None,
                      figures_dir: str = "figures/pygmt",
                      vmin: float = None, vmax: float = None,
                      cmap: str = "rainbow"):
    """
    Generate and save every map type for every model.

    Args:
        models:       {model_name: trained_model}
        X_lf, Y_lf:  LF data
        X_hf, Y_hf:  HF training data
        loo_results:  {model_name: {y_true, y_pred, y_std}} from run_loo_cv
        figures_dir:  output directory
        vmin, vmax:   shared temperature color range (auto if None)
        cmap:         GMT colormap for prediction maps
    """
    out = Path(figures_dir)
    out.mkdir(parents=True, exist_ok=True)

    Y_hf_f = np.asarray(Y_hf).flatten()
    Y_lf_f = np.asarray(Y_lf).flatten()

    v0 = vmin if vmin is not None else float(min(Y_lf_f.min(), Y_hf_f.min()))
    v1 = vmax if vmax is not None else float(max(Y_lf_f.max(), Y_hf_f.max()))

    print(f"Saving PyGMT maps to: {out}")
    print(f"  Colorscale: {cmap}  [{v0:.2f}, {v1:.2f}]")

    # ── Data maps ──────────────────────────────────────────────────────────
    print("  [1/5] LF locations...")
    plot_lf_locations(X_lf, save_path=str(out / "01_lf_locations.png"))

    print("  [2/5] HF locations...")
    plot_hf_locations(X_hf, save_path=str(out / "02_hf_locations.png"))

    print("  [3/5] Data coverage...")
    plot_data_coverage(X_lf, X_hf, save_path=str(out / "03_data_coverage.png"))

    print("  [4/5] LF gridded surface...")
    plot_lf_surface(X_lf, Y_lf, vmin=v0, vmax=v1, cmap=cmap,
                    save_path=str(out / "04_lf_surface.png"))

    print("  [5/5] HF observations...")
    plot_hf_surface(X_hf, Y_hf, vmin=v0, vmax=v1, cmap=cmap,
                    save_path=str(out / "05_hf_observations.png"))

    # ── Per-model maps ─────────────────────────────────────────────────────
    for name, model in models.items():
        slug = name.lower().replace(" ", "_").replace("-", "_")
        print(f"\n  Model: {name}")

        print("    -> prediction surface")
        plot_prediction_map(model, X_lf, X_hf, Y_hf, model_name=name,
                            vmin=v0, vmax=v1, cmap=cmap,
                            save_path=str(out / f"{slug}_prediction.png"))

        print("    -> uncertainty surface")
        plot_uncertainty_map(model, X_lf, X_hf, model_name=name,
                             save_path=str(out / f"{slug}_uncertainty.png"))

        print("    -> HF-LF discrepancy")
        plot_discrepancy_map(model, X_lf, X_hf, model_name=name,
                             save_path=str(out / f"{slug}_discrepancy.png"))

        if loo_results and name in loo_results:
            r = loo_results[name]
            y_true = np.asarray(r.get('y_true', [])).flatten()
            y_pred = np.asarray(r.get('y_pred', [])).flatten()
            y_std  = r.get('y_std', None)

            if len(y_true) == len(X_hf):
                print("    -> absolute error map")
                plot_error_map(y_true, y_pred, X_hf, model_name=name,
                               save_path=str(out / f"{slug}_error.png"))

                print("    -> signed residuals")
                plot_signed_residuals(y_true, y_pred, X_hf, model_name=name,
                                      save_path=str(out / f"{slug}_residuals.png"))

                print("    -> LOO two-panel")
                plot_loo_spatial(y_true, y_pred, y_std, X_hf, model_name=name,
                                 save_path=str(out / f"{slug}_loo_spatial.png"))

    # ── Multi-model panel ──────────────────────────────────────────────────
    if len(models) > 1:
        print("\n  -> multi-model comparison panel")
        plot_model_comparison(models, X_lf, X_hf, Y_hf,
                              vmin=v0, vmax=v1, cmap=cmap,
                              save_path=str(out / "comparison_panel.png"))

    n_saved = len(list(out.glob("*.png")))
    print(f"\nDone. {n_saved} maps saved to {out}")


# ── Standalone test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    np.random.seed(42)

    n_lf, n_hf = 80, 12
    X_lf = np.random.uniform([0, 0], [10, 8], (n_lf, 2)).astype(np.float32)
    Y_lf = (20 + 2 * np.sin(X_lf[:, 0]) + 0.5 * X_lf[:, 1]
            + 0.3 * np.random.randn(n_lf)).reshape(-1, 1).astype(np.float32)

    X_hf = np.random.uniform([0, 0], [10, 8], (n_hf, 2)).astype(np.float32)
    Y_hf = (20 + 2 * np.sin(X_hf[:, 0]) + 0.5 * X_hf[:, 1]
            + 1.0 * X_hf[:, 1] / 8
            + 0.1 * np.random.randn(n_hf)).reshape(-1, 1).astype(np.float32)

    class _StubModel:
        name = "Stub"
        def predict(self, X, return_std=True):
            X = np.asarray(X)
            m = (20 + 2 * np.sin(X[:, 0]) + 0.5 * X[:, 1]).reshape(-1, 1)
            s = 0.3 * np.ones_like(m)
            return (m, s) if return_std else (m, None)
        def predict_lf(self, X):
            X = np.asarray(X)
            return (19 + 1.5 * np.sin(X[:, 0]) + 0.4 * X[:, 1]).reshape(-1, 1)

    stub = _StubModel()
    generate_all_maps(
        models={"Stub": stub},
        X_lf=X_lf, Y_lf=Y_lf,
        X_hf=X_hf, Y_hf=Y_hf,
        figures_dir=str(Path(__file__).parent.parent / "figures" / "pygmt"),
        vmin=17, vmax=25,
    )
    print("\nAll done!")

"""
Two Parker spirals with slow and fast solar wind speeds,
rendered as one clean spiral per time step and latitude slice.

Output
- One PNG per timestep for GIF creation.
- Optional in-Python GIF assembly using imageio.

author: Fabrizio Musacchio
date: Jul 2020 / Jan 2026
"""
# %% IMPORTS
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt

# %% CONSTANTS
AU = 1.495978707e11  # m
R_SUN = 6.9634e8     # m


@dataclass
class ParkerZonalParams:
    # Rotation and geometry
    omega: float = 2.86533e-6      # rad/s
    r0: float = 10.0 * R_SUN       # m
    r_max: float = 2.0 * AU        # m

    # solar wind structure:
    v_slow: float = 400e3          # m/s
    v_fast: float = 750e3          # m/s
    lambda_band_deg: float = 20.0  # equatorial half-width in degrees for slow wind

    # Sampling
    n_lat: int = 11                # number of latitude slices shown
    lat_max_deg: float = 70.0      # max heliolatitude shown (symmetric about equator)

    n_lines_per_lat: int = 6       # number of distinct spirals (different phi0) per latitude
    n_r: int = 2000                # radial resolution along each spiral

    # Time stepping
    dt: float = 6 * 3600.0         # s
    n_steps: int = 80

    # Plot
    figsize: tuple[float, float] = (7.0, 7.0)
    dpi: int = 180

# %% FUNCTIONS
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def wrap_pi(a: np.ndarray) -> np.ndarray:
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def v_sw_latitude(lambda_rad: np.ndarray, v_slow: float, v_fast: float, lambda_band_rad: float) -> np.ndarray:
    """
    Piecewise-constant solar wind speed profile by latitude:
    slow for |lambda| <= lambda_band, fast otherwise.
    """
    return np.where(np.abs(lambda_rad) <= lambda_band_rad, v_slow, v_fast)


def render_parker_spiral_zonal_frames(
    params: ParkerZonalParams,
    out_dir: str = "frames_parker_spiral_2_speeds",
    make_gif: bool = False,
    gif_path: str = "parker_spiral_2_speeds.gif",
    gif_frame_duration_s: float = 0.08,
) -> None:
    ensure_dir(out_dir)

    # radii:
    r = np.linspace(params.r0, params.r_max, params.n_r)

    # latitude slices:
    lat_max = np.deg2rad(params.lat_max_deg)
    lambdas = np.linspace(-lat_max, lat_max, params.n_lat)

    # boundary:
    lambda_band = np.deg2rad(params.lambda_band_deg)

    # per-latitude wind speeds:
    v_lam = v_sw_latitude(lambdas, params.v_slow, params.v_fast, lambda_band)

    # field-line footpoint longitudes (same set for each latitude):
    phi0s = np.linspace(0.0, 2.0 * np.pi, params.n_lines_per_lat, endpoint=False)

    gif_frames = []

    for k in range(params.n_steps):
        t = k * params.dt

        fig = plt.figure(figsize=params.figsize)
        ax = fig.add_subplot(111, projection="polar")

        # reference circles: Sun and r0:
        theta_ref = np.linspace(-np.pi, np.pi, 512)
        ax.plot(theta_ref, np.full_like(theta_ref, R_SUN / AU), linewidth=1.0)
        ax.plot(theta_ref, np.full_like(theta_ref, params.r0 / AU), linewidth=1.0)

        # draw spirals by latitude.
        # We avoid explicit colors and distinguish regimes by linestyle:
        # slow: solid, fast: dotted. Latitude is conveyed by transparency via alpha scaling.
        # Matplotlib will choose default cycle colors, but the regime distinction remains in linestyle.
        for lam, v in zip(lambdas, v_lam):
            is_slow = np.abs(lam) <= lambda_band
            ls = "-" if is_slow else ":"

            # Mild alpha modulation so different latitudes are visually separable without manual colors.
            # This does not fix colors, it only changes transparency.
            alpha = 0.35 + 0.55 * (1.0 - np.abs(lam) / lat_max)

            for phi0 in phi0s:
                phi = phi0 + params.omega * t - (params.omega / v) * (r - params.r0)
                theta = wrap_pi(phi)
                ax.plot(theta, r / AU, linewidth=1.1, linestyle=ls, alpha=alpha)

        ax.set_rlim(0.0, params.r_max / AU)
        ax.set_rlabel_position(135)

        ax.set_title(
            "Parker spiral with zonal wind structure\n"
            f"slow in |λ| ≤ {params.lambda_band_deg:.0f}°, fast outside, "
            f"t = {t/3600:.1f} h, Ω = {params.omega:.2e} rad/s\n"
            f"v_slow = {params.v_slow/1e3:.0f} km/s, v_fast = {params.v_fast/1e3:.0f} km/s",
            va="bottom",
        )

        # Minimal legend-like annotation without explicit legend boxes
        ax.text(
            0.02, 0.02,
            "Line style: solid = slow wind, dotted = fast wind\n"
            "Multiple spirals: different longitudes; multiple stacks: different latitudes",
            transform=ax.transAxes,
            ha="left", va="bottom",
            fontsize=9,
        )

        frame_path = os.path.join(out_dir, f"frame_{k:05d}.png")
        fig.savefig(frame_path, dpi=params.dpi, bbox_inches="tight")
        plt.close(fig)

        if make_gif:
            try:
                import imageio.v2 as imageio
            except Exception as e:
                raise RuntimeError("make_gif=True requires imageio. Install via: pip install imageio") from e
            gif_frames.append(imageio.imread(frame_path))

    if make_gif:
        import imageio.v2 as imageio
        imageio.mimsave(gif_path, gif_frames, duration=gif_frame_duration_s)
        print(f"Wrote GIF: {gif_path}")

    print(f"Wrote {params.n_steps} frames to: {out_dir}")

# %% MAIN
if __name__ == "__main__":
    params = ParkerZonalParams(
        omega=2.865e-6,
        r0=10 * R_SUN,
        r_max=2.0 * AU,
        v_slow=400e3,
        v_fast=2000e3,
        lambda_band_deg=20.0,
        n_lat=13,
        lat_max_deg=70.0,
        n_lines_per_lat=6,
        n_r=2500,
        dt=6 * 3600.0,
        n_steps=90,
        dpi=180)

    render_parker_spiral_zonal_frames(
        params,
        out_dir="frames_parker_spiral_zonal",
        make_gif=True,
        gif_path="parker_spiral_zonal.gif",
        gif_frame_duration_s=0.08)
# %% END
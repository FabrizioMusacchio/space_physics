"""
3D Parker spiral with a simple zonal wind structure:
slow solar wind in an equatorial band, fast wind at higher latitudes.

Why 3D
- In a 2D polar plot, latitude is not visible, so multiple latitude slices project on top
  of each other. If v_sw is piecewise constant, you only see two families of curves.
- In 3D, latitude becomes the z coordinate, so the zonal structure is visually explicit.

Model
- Solar rotation: constant angular speed omega
- Radial wind speed depends on heliographic latitude lambda:
    v_sw(lambda) = v_slow for |lambda| <= lambda_band
                  v_fast otherwise
  Optionally: smooth transition.
- Ballistic mapping and frozen-in field:
    phi(r, t; lambda) = phi0 + omega * t - (omega / v_sw(lambda)) * (r - r0)
- 3D embedding for a latitude slice lambda:
    x = r cos(lambda) cos(phi)
    y = r cos(lambda) sin(phi)
    z = r sin(lambda)

Output
- One PNG per timestep.
- Optional GIF via imageio.

No em dashes are used.

author: Fabrizio Musacchio
date: Jul 2020 / Jan 2026
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt

AU = 1.495978707e11  # m
R_SUN = 6.9634e8     # m


@dataclass
class ParkerZonal3DParams:
    omega: float = 2.86533e-6    # rad/s
    r0: float = 10.0 * R_SUN     # m
    r_max: float = 2.0 * AU      # m

    v_slow: float = 400e3        # m/s
    v_fast: float = 750e3        # m/s
    lambda_band_deg: float = 20.0

    # If smooth_transition=True, use a tanh profile instead of a step.
    smooth_transition: bool = True
    transition_width_deg: float = 6.0  # only used if smooth_transition=True

    # Sampling
    n_lat: int = 13
    lat_max_deg: float = 70.0
    n_lines_per_lat: int = 5
    n_r: int = 1600

    # Time stepping
    dt: float = 6 * 3600.0
    n_steps: int = 80

    # Plot
    figsize: tuple[float, float] = (8.0, 8.0)
    dpi: int = 180

    # 3D camera
    elev: float = 25.0
    azim: float = 35.0


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def v_sw_profile(
    lam: np.ndarray,
    v_slow: float,
    v_fast: float,
    lambda_band_rad: float,
    smooth_transition: bool,
    transition_width_rad: float,
) -> np.ndarray:
    """
    Latitude dependent wind speed.

    Step profile:
        v = v_slow inside band, v_fast outside.

    Smooth profile:
        v transitions from slow to fast using tanh in |lambda|.
    """
    if not smooth_transition:
        return np.where(np.abs(lam) <= lambda_band_rad, v_slow, v_fast)

    # Smooth transition in |lambda| around lambda_band
    x = (np.abs(lam) - lambda_band_rad) / max(transition_width_rad, 1e-12)
    w = 0.5 * (1.0 + np.tanh(x))  # 0 inside band, 1 far outside
    return (1.0 - w) * v_slow + w * v_fast


def render_parker_spiral_zonal_3d(
    params: ParkerZonal3DParams,
    out_dir: str = "frames_parker_spiral_zonal_3d",
    make_gif: bool = False,
    gif_path: str = "parker_spiral_zonal_3d.gif",
    gif_frame_duration_s: float = 0.08,
) -> None:
    ensure_dir(out_dir)

    r = np.linspace(params.r0, params.r_max, params.n_r)

    lat_max = np.deg2rad(params.lat_max_deg)
    lambdas = np.linspace(-lat_max, lat_max, params.n_lat)

    lambda_band = np.deg2rad(params.lambda_band_deg)
    transition_width = np.deg2rad(params.transition_width_deg)

    v_lam = v_sw_profile(
        lambdas,
        params.v_slow,
        params.v_fast,
        lambda_band,
        params.smooth_transition,
        transition_width,
    )

    phi0s = np.linspace(0.0, 2.0 * np.pi, params.n_lines_per_lat, endpoint=False)

    gif_frames = []

    # For equal aspect ratio in 3D
    rmax_AU = params.r_max / AU

    for k in range(params.n_steps):
        t = k * params.dt

        fig = plt.figure(figsize=params.figsize)
        ax = fig.add_subplot(111, projection="3d")
        ax.view_init(elev=params.elev, azim=params.azim)

        # Draw spirals for each latitude
        for lam, v in zip(lambdas, v_lam):
            cosl = np.cos(lam)
            sinl = np.sin(lam)

            # Use linestyle to emphasize slow vs fast band
            is_slow = np.abs(lam) <= lambda_band
            ls = "-" if is_slow else ":"

            for phi0 in phi0s:
                phi = phi0 + params.omega * t - (params.omega / v) * (r - params.r0)

                x = (r * cosl * np.cos(phi)) / AU
                y = (r * cosl * np.sin(phi)) / AU
                z = (r * sinl) / AU

                ax.plot(x, y, z, linewidth=1.0, linestyle=ls, alpha=0.75)

        # Draw a small sphere marker for the Sun (as a point)
        ax.scatter([0.0], [0.0], [0.0], s=20)

        # Axes limits and labels
        ax.set_xlim(-rmax_AU, rmax_AU)
        ax.set_ylim(-rmax_AU, rmax_AU)
        ax.set_zlim(-rmax_AU, rmax_AU)

        ax.set_xlabel("x [AU]")
        ax.set_ylabel("y [AU]")
        ax.set_zlabel("z [AU]")

        title_profile = "smooth" if params.smooth_transition else "step"
        ax.set_title(
            "3D Parker spiral with zonal wind structure\n"
            f"slow band |λ| ≤ {params.lambda_band_deg:.0f}°, profile = {title_profile}, "
            f"t = {t/3600:.1f} h\n"
            f"v_slow = {params.v_slow/1e3:.0f} km/s, v_fast = {params.v_fast/1e3:.0f} km/s, "
            f"Ω = {params.omega:.2e} rad/s"
        )

        # Small annotation
        ax.text2D(
            0.02, 0.02,
            "Line style: solid = slow band, dotted = fast outside",
            transform=ax.transAxes,
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


if __name__ == "__main__":
    params = ParkerZonal3DParams(
        omega=2.865e-6,
        r0=10 * R_SUN,
        r_max=2.0 * AU,
        v_slow=400e3,
        v_fast=2000e3,              
        lambda_band_deg=20.0,
        smooth_transition=True,
        transition_width_deg=6.0,
        n_lat=13,
        lat_max_deg=70.0,
        n_lines_per_lat=5,
        n_r=1800,
        dt=6 * 3600.0,
        n_steps=90,
        dpi=180,
        elev=25.0,
        azim=35.0)

    render_parker_spiral_zonal_3d(
        params,
        out_dir="frames_parker_spiral_zonal_3d",
        make_gif=True,
        gif_path="parker_spiral_zonal_3d.gif",
        gif_frame_duration_s=0.08)
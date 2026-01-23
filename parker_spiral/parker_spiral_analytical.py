"""
Parker spiral (analytic curve) rendered as one clean spiral per time step.

Model
- Constant solar wind speed v_sw (radial, for r >= r0)
- Constant solar rotation rate Omega
- Frozen-in field, ballistic mapping

Field line equation at time t:
Choose a field line by its footpoint longitude phi0 at t = 0 (in the inertial frame).
The plasma element currently at radius r left the source at time
    t_launch = t - (r - r0)/v_sw .
At launch, the footpoint longitude was
    phi_foot(t_launch) = phi0 + Omega * t_launch .
Frozen-in implies the element keeps that longitude, so
    phi(r,t) = phi0 + Omega * (t - (r - r0)/v_sw)
            = (phi0 + Omega t) - (Omega/v_sw) * (r - r0).

This is an Archimedean spiral in polar coordinates (r, phi).

Output
- One PNG per timestep in out_dir, suitable for ffmpeg/ImageMagick.
- Optional GIF assembly via imageio.
"""
#  %% IMPORTS
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt

import imageio.v2 as imageio
#  %% CONSTANTS
AU = 1.495978707e11  # m, astronomical unit
R_SUN = 6.9634e8     # m, solar radius

# next, define parameters dataclass that holds all simulation settings:
@dataclass
class ParkerAnalyticParams:
    v_sw: float = 400e3          # m/s
    omega: float = 2.86533e-6    # rad/s
    r0: float = 10.0 * R_SUN     # m
    r_max: float = 2.0 * AU      # m

    n_lines: int = 12            # number of spirals with different phi0
    n_r: int = 2000              # resolution along r

    dt: float = 6 * 3600.0       # s
    n_steps: int = 80            # number of frames

    # plot styling:
    figsize: tuple[float, float] = (7.0, 7.0)
    dpi: int = 170

# %% FUNCTIONS

# function to ensure output directory exists:
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

# function to wrap angles to [-pi, pi]:
def wrap_pi(a: np.ndarray) -> np.ndarray:
    """Wrap angles to [-pi, pi] for nicer polar plotting."""
    return (a + np.pi) % (2.0 * np.pi) - np.pi

# function to render frames of the Parker spiral:
def render_parker_spiral_frames(
    params: ParkerAnalyticParams,
    out_dir: str = "frames_parker_spiral_clean",
    make_gif: bool = False,
    gif_path: str = "parker_spiral_clean.gif",
    gif_frame_duration_s: float = 0.08) -> None:
    ensure_dir(out_dir)

    # radii to draw (meters):
    r = np.linspace(params.r0, params.r_max, params.n_r)

    # choose evenly spaced field-line footpoint longitudes at t=0:
    phi0s = np.linspace(0.0, 2.0 * np.pi, params.n_lines, endpoint=False)

    gif_frames = []

    for k in range(params.n_steps):
        t = k * params.dt

        fig = plt.figure(figsize=params.figsize)
        ax = fig.add_subplot(111, projection="polar")

        # reference circles: Sun radius and r0:
        theta_ref = np.linspace(-np.pi, np.pi, 512)
        ax.plot(theta_ref, np.full_like(theta_ref, R_SUN / AU), linewidth=1.0)
        ax.plot(theta_ref, np.full_like(theta_ref, params.r0 / AU), linewidth=1.0)

        # Draw each spiral
        for phi0 in phi0s:
            phi = phi0 + params.omega * t - (params.omega / params.v_sw) * (r - params.r0)
            theta = wrap_pi(phi)
            ax.plot(theta, r / AU, linewidth=1.3)

        ax.set_rlim(0.0, params.r_max / AU)
        ax.set_rlabel_position(135)

        ax.set_title(
            "Parker spiral (analytic curve)\n"
            f"t = {t/3600:.1f} h, v_sw = {params.v_sw/1e3:.0f} km/s, Ω = {params.omega:.2e} rad/s",
            va="bottom")

        frame_path = os.path.join(out_dir, f"frame_{k:05d}.png")
        fig.savefig(frame_path, dpi=params.dpi, bbox_inches="tight")
        plt.close(fig)

        if make_gif:
            try:
                import imageio.v2 as imageio
            except Exception as e:
                raise RuntimeError("make_gif=True requires imageio. Install via: pip install imageio") from e
            gif_frames.append(imageio.imread(frame_path))

    # assemble GIF:
    if make_gif:
        import imageio.v2 as imageio
        imageio.mimsave(gif_path, gif_frames, duration=gif_frame_duration_s)
        print(f"Wrote GIF: {gif_path}")

    print(f"Wrote {params.n_steps} frames to: {out_dir}")

# %% MAIN
if __name__ == "__main__":
    params = ParkerAnalyticParams(
        v_sw=400e3,         # 400 km/s
        omega=2.865e-6,     # rad/s
        r0=10 * R_SUN,
        r_max=2.0 * AU,
        n_lines=14,
        n_r=2500,
        dt=6 * 3600.0,
        n_steps=90,
        dpi=180)

    render_parker_spiral_frames(
        params,
        out_dir="frames_parker_spiral_analytic",
        make_gif=True,
        gif_path="parker_spiral_analytic.gif",
        gif_frame_duration_s=0.08)
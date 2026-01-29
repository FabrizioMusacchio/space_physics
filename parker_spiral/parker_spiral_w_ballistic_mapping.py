"""
Parker spiral simulator (2D, ecliptic plane) with per-timestep frames.

What it does
- Simulates field-line "footpoints" co-rotating with the Sun.
- Solar wind plasma parcels stream radially outward at constant speed v_sw.
- At each timestep, for each launched parcel: r(t) = r0 + v_sw (t - t_launch),
  phi(t) = phi_foot(t_launch) (frozen-in field, ballistic mapping).
- Produces one PNG per timestep for easy GIF creation later.
- Optionally also writes an animated GIF directly in Python (requires imageio).

Units
- r is in meters internally, plotted in AU for readability.
- Angles in radians.

author: Fabrizio Musacchio
date: Jul 2020 / Jan 2026
"""
# %% IMPORTS
from __future__ import annotations

import os
import math
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt

import imageio.v2 as imageio
# %% CONSTANTS
AU = 1.495978707e11  # m
R_SUN = 6.9634e8     # m


# next, define parameters dataclass that holds all simulation settings:

@dataclass
class ParkerSpiralParams:
    # solar wind speed (constant):
    v_sw: float = 400e3  # m/s

    # solar rotation rate (sidereal ~ 25.38 days at equator):
    # You can also use synodic ~ 27.27 days:
    omega: float = 2.86533e-6  # rad/s

    # launch radius (start of spiral), e.g. a few solar radii:
    r0: float = 10.0 * R_SUN  # m

    # max plotting radius (domain):
    r_max: float = 2.0 * AU  # m

    # number of distinct field lines (different longitudes):
    n_lines: int = 16

    # initial longitudes for the lines at t=0 (if None: uniform):
    phi0: np.ndarray | None = None

    # simulation time settings:
    dt: float = 6 * 3600.0  # s (6 hours)
    n_steps: int = 80

    # launch cadence for new parcels per line 
    # (smaller => smoother spirals, heavier compute):
    launch_every: int = 1  # launch each step

    # for each line, keep all parcels or cap for performance:
    max_parcels_per_line: int = 5000

# %% FUNCTIONS

# function to ensure output directory exists:
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

# function to create uniform phi0 values:
def make_phi0(n_lines: int) -> np.ndarray:
    return np.linspace(0.0, 2.0 * np.pi, n_lines, endpoint=False)

# main simulation function:
def simulate_parker_spiral(
    params: ParkerSpiralParams,
    out_dir: str = "frames_parker_spiral",
    make_gif: bool = False,
    gif_path: str = "parker_spiral.gif",
    dpi: int = 160) -> None:
    """
    Main simulation function for Parker spiral with ballistic mapping.
    """
    ensure_dir(out_dir)

    phi0 = params.phi0 if params.phi0 is not None else make_phi0(params.n_lines)
    phi0 = np.asarray(phi0, dtype=float)
    if phi0.shape != (params.n_lines,):
        raise ValueError(f"phi0 must have shape ({params.n_lines},), got {phi0.shape}")

    # for each line we store arrays of launch_times and launch_phis:
    launch_times = [np.empty(0, dtype=float) for _ in range(params.n_lines)]
    launch_phis = [np.empty(0, dtype=float) for _ in range(params.n_lines)]

    # helper: wrap angles to [-pi, pi] for plotting continuity:
    def wrap_pi(a: np.ndarray) -> np.ndarray:
        return (a + np.pi) % (2.0 * np.pi) - np.pi

    # for optional GIF assembly:
    gif_frames = []

    # precompute plot extent:
    r_plot_max_AU = params.r_max / AU

    for k in range(params.n_steps):
        t = k * params.dt

        # launch new parcels (footpoints co-rotate with the Sun):
        if (k % params.launch_every) == 0:
            for i in range(params.n_lines):
                phi_foot = phi0[i] + params.omega * t  # co-rotating footpoint
                # Append one launch
                launch_times[i] = np.append(launch_times[i], t)
                launch_phis[i] = np.append(launch_phis[i], phi_foot)

                # Cap memory if needed
                if launch_times[i].size > params.max_parcels_per_line:
                    launch_times[i] = launch_times[i][-params.max_parcels_per_line :]
                    launch_phis[i] = launch_phis[i][-params.max_parcels_per_line :]

        # build frame plot:
        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(111, projection="polar")

        # plot Sun and launch radius as reference circles:
        theta_ref = np.linspace(-np.pi, np.pi, 512)
        ax.plot(theta_ref, np.full_like(theta_ref, R_SUN / AU), linewidth=1.0)
        ax.plot(theta_ref, np.full_like(theta_ref, params.r0 / AU), linewidth=1.0)

        # for each field line, compute current parcel positions:
        for i in range(params.n_lines):
            if launch_times[i].size == 0:
                continue

            tau = t - launch_times[i]  # time since launch
            # only parcels that have had time to travel outward:
            valid = tau >= 0.0
            if not np.any(valid):
                continue

            tau = tau[valid]
            phi_launch = launch_phis[i][valid]

            # ballistic mapping:
            r = params.r0 + params.v_sw * tau  # meters
            keep = r <= params.r_max
            if not np.any(keep):
                continue

            r = r[keep]
            phi = phi_launch[keep]  # frozen-in angle
            # convert to polar plot coordinates: theta = inertial longitude:
            theta = wrap_pi(phi)
            ax.plot(theta, r / AU, linewidth=1.2)

        # formatting:
        ax.set_rlim(0.0, r_plot_max_AU)
        ax.set_rlabel_position(135)
        ax.set_title(
            "Parker spiral (ballistic mapping)\n"
            f"t = {t/3600:.1f} h, v_sw = {params.v_sw/1e3:.0f} km/s, "
            f"Ω = {params.omega:.2e} rad/s",
            va="bottom")

        # save current frame:
        frame_path = os.path.join(out_dir, f"frame_{k:05d}.png")
        fig.savefig(frame_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)

        # collect for GIF (if requested):
        if make_gif:
            try:
                import imageio.v2 as imageio
            except Exception as e:
                raise RuntimeError("make_gif=True requires imageio. Install via: pip install imageio") from e
            gif_frames.append(imageio.imread(frame_path))

    # assemble GIF:
    if make_gif:
        # duration per frame in seconds, set equal to dt in some human-friendly scaling
        # Here: 0.08 s per frame by default, adjust as you like.
        imageio.mimsave(gif_path, gif_frames, duration=0.08)
        print(f"Wrote GIF: {gif_path}")

    print(f"Wrote {params.n_steps} frames to: {out_dir}")

# %% MAIN
if __name__ == "__main__":
    params = ParkerSpiralParams(
        v_sw=400e3,      # 400 km/s for slow solar, try 800e3 for fast solar wind
        omega=2.865e-6,  # rad/s
        r0=10 * R_SUN,
        r_max=2.0 * AU,
        n_lines=18,
        dt=6 * 3600.0,
        n_steps=90,
        launch_every=1)

    simulate_parker_spiral(
        params,
        out_dir="frames_parker_spiral",
        make_gif=True,              # set False if you only want PNG frames
        gif_path="parker_spiral_ballistic_mapping.gif",
        dpi=170)
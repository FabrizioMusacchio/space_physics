"""
Toy model of magnetic reconnection (2D flux-function advection–diffusion).

We evolve a scalar magnetic flux function A(x,y,t) (out-of-plane vector potential),
such that B = ∇×(A ẑ):
    Bx =  ∂A/∂y
    By = -∂A/∂x

We prescribe an incompressible stagnation-point flow that drives an X-point collapse:
    v = (-α x, +α y)

and evolve resistive induction in 2D for A:
    ∂A/∂t + v·∇A = η ∇²A

This is NOT a full MHD simulation (no momentum equation, no self-consistent flow),
but it captures the core idea: advection thins the current sheet and resistive
diffusion changes topology, producing reconnection.

Outputs:
* ./reconnection_out_var/frames/frame_000000.png ... (all timesteps)
* ./reconnection_out_var/reconnection.gif           (animated)
* ./reconnection_out_var/initial.png
* ./reconnection_out_var/middle.png
* ./reconnection_out_var/final.png

author: Fabrizio Musacchio
date: Jul 2020 / Jan 2026
"""
# %% IMPORTS
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio
# %% FUNCTIONS

# function to ensure output directory exists:
def ensure_dir(p: str | Path) -> None:
    Path(p).mkdir(parents=True, exist_ok=True)

# central difference and Laplacian with periodic BC:
def ddx_central(f: np.ndarray, dx: float) -> np.ndarray:
    """Central difference with periodic BC in x."""
    return (np.roll(f, -1, axis=1) - np.roll(f, 1, axis=1)) / (2.0 * dx)

# central difference and Laplacian with periodic BC:
def ddy_central(f: np.ndarray, dy: float) -> np.ndarray:
    """Central difference with periodic BC in y."""
    return (np.roll(f, -1, axis=0) - np.roll(f, 1, axis=0)) / (2.0 * dy)

# Laplacian with periodic BC:
def laplacian_periodic(f: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """2D Laplacian with periodic BC in both directions."""
    f_xx = (np.roll(f, -1, axis=1) - 2.0 * f + np.roll(f, 1, axis=1)) / (dx * dx)
    f_yy = (np.roll(f, -1, axis=0) - 2.0 * f + np.roll(f, 1, axis=0)) / (dy * dy)
    return f_xx + f_yy

# Initial condition for A(x,y):
def make_initial_A(X: np.ndarray, Y: np.ndarray,
                   B0: float = 1.0,
                   a: float = 0.15,
                   sheet_sigma: float = 0.12,
                   mode: int = 1) -> np.ndarray:
    """
    X-point flux plus a perturbation that seeds a thin current sheet around x=0.

    A_X = B0 * x * y  gives an X-type field.
    The perturbation adds strong gradients near x=0, supporting a current layer.
    """
    A_x = B0 * X * Y
    A_pert = a * np.cos(2.0 * np.pi * mode * Y) * np.exp(-(X * X) / (2.0 * sheet_sigma * sheet_sigma))
    return A_x + A_pert

# compute B and Jz from A:
def compute_fields(A: np.ndarray, dx: float, dy: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return Bx, By, and current proxy Jz = -∇²A."""
    Bx = ddy_central(A, dy)
    By = -ddx_central(A, dx)
    Jz = -laplacian_periodic(A, dx, dy)
    return Bx, By, Jz

# main plotting function:
def plot_frame(A: np.ndarray,
               X: np.ndarray,
               Y: np.ndarray,
               dx: float,
               dy: float,
               t: float,
               out_png: str,
               quiver_stride: int = 10,
               n_contours: int = 28) -> None:
    Bx, By, Jz = compute_fields(A, dx, dy)

    Jabs = np.abs(Jz)
    Jmax = np.percentile(Jabs, 99.5) + 1e-12

    fig = plt.figure(figsize=(8.0, 6.5), dpi=140)
    ax = fig.add_subplot(1, 1, 1)

    im = ax.imshow(
        Jabs,
        origin="lower",
        extent=[X.min(), X.max(), Y.min(), Y.max()],
        interpolation="bilinear",
        aspect="auto",
        vmin=0.0,
        vmax=Jmax)
    cb = fig.colorbar(im, ax=ax, shrink=0.9)
    cb.set_label(r"$|J_z| \propto |\nabla^2 A|$ (arb.)")

    levels = np.linspace(np.percentile(A, 2), np.percentile(A, 98), n_contours)
    ax.contour(X, Y, A, levels=levels, linewidths=0.8)

    # Quiver of B (subsampled):
    ys = slice(0, A.shape[0], quiver_stride)
    xs = slice(0, A.shape[1], quiver_stride)
    ax.quiver(
        X[ys, xs], Y[ys, xs],
        Bx[ys, xs], By[ys, xs],
        angles="xy",
        scale_units="xy",
        scale=None,
        width=0.0022,
        alpha=0.9)

    ax.set_title(f"Toy reconnection: contours of A, background |Jz|   t = {t:.4f}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_xlim(X.min(), X.max())
    ax.set_ylim(Y.min(), Y.max())

    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)

# main simulation function:
def run_reconnection_sim(
    out_dir: str = "reconnection_out",
    nx: int = 360,
    ny: int = 240,
    Lx: float = 2.0,
    Ly: float = 2.0,
    alpha: float = 1.2,     # stagnation flow strength
    eta: float = 2.5e-3,    # resistivity (dimensionless in this toy model)
    t_end: float = 1.2,
    cfl_adv: float = 0.35,
    cfl_diff: float = 0.20,
    frame_every: int = 1,
    gif_fps: int = 30) -> None:
    """
    Explicit time stepping for A with periodic boundaries.

    Stability guidelines (explicit):
    * advection: dt <= cfl_adv * min(dx/|vx|max, dy/|vy|max)
    * diffusion: dt <= cfl_diff * min(dx^2, dy^2) / eta
    """
    out_path = Path(out_dir)
    frames_dir = out_path / "frames"
    ensure_dir(frames_dir)

    # Grid
    x = np.linspace(-Lx / 2.0, Lx / 2.0, nx, endpoint=False)
    y = np.linspace(-Ly / 2.0, Ly / 2.0, ny, endpoint=False)
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    X, Y = np.meshgrid(x, y)

    # Prescribed stagnation-point flow (incompressible)
    vx = -alpha * X
    vy = +alpha * Y
    # or use a localized version:
    # w = 0.5 
    # vx = -alpha * X * np.exp(-(X**2 + Y**2)/w**2)
    # vy =  alpha * Y * np.exp(-(X**2 + Y**2)/w**2)
    # or set to zero for pure diffusion test:
    # vx = 0
    # vy = 0
    # or use Sweet-Parker-like inflow:
    # delta = 0.1
    # v0 = 0.5 * alpha * (Lx/2)
    # vx = -v0 * np.tanh(X / delta)
    # vy = 0
    vmax = max(np.max(np.abs(vx)), np.max(np.abs(vy))) + 1e-12

    # initial flux function:
    A = make_initial_A(X, Y, B0=1.0, a=0.16, sheet_sigma=0.10, mode=1)

    # time step from stability constraints:
    dt_adv = cfl_adv * min(dx, dy) / vmax
    dt_diff = cfl_diff * min(dx * dx, dy * dy) / (eta + 1e-30)
    dt = min(dt_adv, dt_diff)

    # determine number of steps, and "middle" frame index:
    n_steps = int(np.ceil(t_end / dt))
    mid_step = n_steps // 2

    # save initial, mid, final PNG once:
    initial_png = out_path / "initial.png"
    middle_png = out_path / "middle.png"
    final_png = out_path / "final.png"
    ensure_dir(out_path)

    # frame list for GIF:
    frame_paths: list[Path] = []

    t = 0.0

    def maybe_save(step: int, Afield: np.ndarray, t_now: float) -> None:
        if step % frame_every != 0:
            return
        fn = frames_dir / f"frame_{step:06d}.png"
        plot_frame(Afield, X, Y, dx, dy, t_now, str(fn))
        frame_paths.append(fn)

    # initial snapshots:
    plot_frame(A, X, Y, dx, dy, t, str(initial_png))
    maybe_save(0, A, t)

    # main time-stepping loop:
    for step in range(1, n_steps + 1):
        if t + dt > t_end:
            dt = t_end - t

        # compute gradients:
        Ax = ddx_central(A, dx)
        Ay = ddy_central(A, dy)
        lapA = laplacian_periodic(A, dx, dy)

        # explicit Euler update for advection-diffusion
        # ∂A/∂t = - v·∇A + η ∇²A:
        rhs = -(vx * Ax + vy * Ay) + eta * lapA
        A = A + dt * rhs

        t += dt

        if step == mid_step:
            plot_frame(A, X, Y, dx, dy, t, str(middle_png))

        maybe_save(step, A, t)

    # final snapshot:
    plot_frame(A, X, Y, dx, dy, t, str(final_png))

    # build GIF from the saved frames:
    gif_path = out_path / "reconnection.gif"
    images = [imageio.imread(p) for p in frame_paths]
    imageio.mimsave(gif_path, images, fps=gif_fps)

    print(f"Done.")
    print(f"Output directory: {out_path.resolve()}")
    print(f"GIF: {gif_path.resolve()}")
    print(f"Initial: {initial_png.resolve()}")
    print(f"Middle: {middle_png.resolve()}")
    print(f"Final: {final_png.resolve()}")
    print(f"Frames: {frames_dir.resolve()}  (count: {len(frame_paths)})")
# %% MAIN
run_reconnection_sim(
    out_dir="reconnection_out_var",     # set the output directory
    nx=360,                             # number of grid points in x
    ny=240,                             # number of grid points in y
    Lx=2.0,                             # domain size in x
    Ly=2.0,                             # domain size in y
    alpha=1.2,                          # flow parameter
    eta=2.5e-3,                         # resistivity
    t_end=1.2,                          # end-time of simulation
    cfl_adv=0.35,                       # CFL number for advection
    cfl_diff=0.20,                      # CFL number for diffusion
    frame_every=1,                      # save every step; the higher the faster the simulation
    gif_fps=30,                         # frames per second for the GIF
)
# %% END
print("done.")
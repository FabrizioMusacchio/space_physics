"""
Didactic 2D magnetic reconnection toy model that matches:
inflow from top/bottom, outflow left/right, current sheet along x at y=0.

We evolve a flux function A(x,y,t) (out-of-plane vector potential), with
    Bx =  dA/dy
    By = -dA/dx

Evolution (kinematic resistive MHD for A):
    dA/dt + v·∇A = ∇·(η ∇A)

Key design choices:
* Initial field: Harris-like current sheet (anti-parallel Bx across y=0),
  plus a localized perturbation that seeds an X-point at x=0.
* Flow: localized stagnation flow with inflow in y and outflow in x:
      v = ( +α x, -α y ) * exp(-(x^2+y^2)/w_v^2)
* Resistivity: background + localized enhancement (diffusion region):
      η = η_bg + η_peak * exp(-(x^2+y^2)/w_η^2)

Outputs:
* ./reconnection_out/frames/frame_000000.png ...
* ./reconnection_out/reconnection.gif
* ./reconnection_out/initial.png, middle.png, final.png

author: Fabrizio Musacchio
date: Jul 2020
"""
# %% IMPORTS
from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio
# %% FUNCTIONS

# utilities:

# function to ensure output directory exists:
def ensure_dir(p: str | Path) -> None:
    Path(p).mkdir(parents=True, exist_ok=True)

# functions for finite differences with various BCs:
def ddx_periodic(f: np.ndarray, dx: float) -> np.ndarray:
    return (np.roll(f, -1, axis=1) - np.roll(f, 1, axis=1)) / (2.0 * dx)

# function for finite differences with various BCs:
def ddy_neumann(f: np.ndarray, dy: float) -> np.ndarray:
    """
    Central difference in y with zero-gradient (Neumann) boundary.
    Implemented by edge padding.
    """
    fp = np.pad(f, ((1, 1), (0, 0)), mode="edge")
    return (fp[2:, :] - fp[:-2, :]) / (2.0 * dy)

# function for finite differences with various BCs:
def laplacian_mixed(f: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """
    Laplacian: periodic in x, Neumann (zero-gradient) in y.
    """
    f_xx = (np.roll(f, -1, axis=1) - 2.0 * f + np.roll(f, 1, axis=1)) / (dx * dx)

    fp = np.pad(f, ((1, 1), (0, 0)), mode="edge")
    f_yy = (fp[2:, :] - 2.0 * fp[1:-1, :] + fp[:-2, :]) / (dy * dy)

    return f_xx + f_yy

# function for finite differences with various BCs:
def grad_eta_dot_grad_A(eta: np.ndarray, A: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """
    Compute ∇·(η ∇A) = η ∇²A + (∇η)·(∇A), with mixed BCs.
    """
    lapA = laplacian_mixed(A, dx, dy)
    etax = ddx_periodic(eta, dx)
    etay = ddy_neumann(eta, dy)
    Ax = ddx_periodic(A, dx)
    Ay = ddy_neumann(A, dy)
    return eta * lapA + (etax * Ax + etay * Ay)

# function to compute B and Jz from A:
def compute_B(A: np.ndarray, dx: float, dy: float) -> tuple[np.ndarray, np.ndarray]:
    Bx = ddy_neumann(A, dy)
    By = -ddx_periodic(A, dx)
    return Bx, By

# function to compute Jz from A:
def compute_Jz(A: np.ndarray, dx: float, dy: float) -> np.ndarray:
    # Jz ∝ (∇×B)_z = -∇²A in these conventions
    return -laplacian_mixed(A, dx, dy)


# initial conditions:

# function to make initial A:
def A_harris(Y: np.ndarray, B0: float, delta_cs: float) -> np.ndarray:
    """
    Harris sheet: Bx(y) = B0 tanh(y/delta_cs)
    A(y) = B0 * delta_cs * ln cosh(y/delta_cs)
    """
    return B0 * delta_cs * np.log(np.cosh(Y / delta_cs) + 1e-30)

# function to make initial A perturbation:
def A_perturbation(X: np.ndarray, Y: np.ndarray, a: float, kx: float, sigma_y: float) -> np.ndarray:
    """
    Localized perturbation that seeds an X-point around (0,0).
    """
    return a * np.cos(kx * X) * np.exp(-(Y * Y) / (2.0 * sigma_y * sigma_y))


# plotting:

# main plotting function for a frame:
def plot_frame(A: np.ndarray,
                       vx: np.ndarray,
                       vy: np.ndarray,
                       X: np.ndarray,
                       Y: np.ndarray,
                       dx: float,
                       dy: float,
                       t: float,
                       out_png: str,
                       diff_box: float = 0.12,
                       streamline_density: float = 1.2,
                       show_velocity: bool = False) -> None:
    """
    Plot:
    * background: |Jz|
    * field lines: streamlines in red (top) and blue (bottom)
    * optional velocity quiver
    * yellow arrows for inflow/outflow
    * yellow rectangle for diffusion region
    """
    Bx, By = compute_B(A, dx, dy)
    Jz = compute_Jz(A, dx, dy)
    Jabs = np.abs(Jz)
    vmaxJ = np.percentile(Jabs, 99.7) + 1e-12

    fig = plt.figure(figsize=(8.2, 5.6), dpi=150)
    ax = fig.add_subplot(1, 1, 1)

    im = ax.imshow(
        Jabs,
        origin="lower",
        extent=[X.min(), X.max(), Y.min(), Y.max()],
        aspect="auto",
        vmin=0.0,
        vmax=vmaxJ,
        interpolation="bilinear")
    cb = fig.colorbar(im, ax=ax, shrink=0.9)
    cb.set_label(r"$|J_z| \propto |\nabla^2 A|$ (arb.)")

    # streamlines: top (red) and bottom (blue)
    # choose start points in two horizontal lines above and below the sheet
    nseed = 20
    xs = np.linspace(X.min() * 0.95, X.max() * 0.95, nseed)

    y_top = 0.65 * Y.max()
    y_bot = 0.65 * Y.min()

    start_top = np.column_stack([xs, np.full_like(xs, y_top)])
    start_bot = np.column_stack([xs, np.full_like(xs, y_bot)])

    # draw separately to enforce color
    """ ax.streamplot(X, Y, Bx, By, start_points=start_top, density=streamline_density,
                  linewidth=2.2, arrowsize=1.1, color="red")
    ax.streamplot(X, Y, Bx, By, start_points=start_bot, density=streamline_density,
                  linewidth=2.2, arrowsize=1.1, color="blue")
    """
    levels = np.linspace(np.percentile(A, 5), np.percentile(A, 95), 18)

    A_top = np.where(Y >= 0, A, np.nan)
    A_bot = np.where(Y <= 0, A, np.nan)

    ax.contour(X, Y, A_top, levels=levels, colors="red", linewidths=2.2)
    ax.contour(X, Y, A_bot, levels=levels, colors="blue", linewidths=2.2)
    
    
    # diffusion region box
    from matplotlib.patches import Rectangle
    rect = Rectangle((-diff_box, -diff_box), 2.0 * diff_box, 2.0 * diff_box,
                     fill=False, lw=2.0, edgecolor="gold")
    ax.add_patch(rect)

    # schematic inflow/outflow arrows
    # inflow: from top and bottom towards origin
    ax.annotate("", xy=(0.0, 0.22), xytext=(0.0, 0.60),
                arrowprops=dict(arrowstyle="simple", color="gold", alpha=0.5))
    ax.annotate("", xy=(0.0, -0.22), xytext=(0.0, -0.60),
                arrowprops=dict(arrowstyle="simple", color="gold", alpha=0.5))

    # outflow: from origin to left/right
    ax.annotate("", xy=(0.60, 0.0), xytext=(0.22, 0.0),
                arrowprops=dict(arrowstyle="simple", color="gold", alpha=0.5))
    ax.annotate("", xy=(-0.60, 0.0), xytext=(-0.22, 0.0),
                arrowprops=dict(arrowstyle="simple", color="gold", alpha=0.5))

    # optional velocity quiver (subsampled)
    if show_velocity:
        stride = 14
        ax.quiver(X[::stride, ::stride], Y[::stride, ::stride],
                  vx[::stride, ::stride], vy[::stride, ::stride],
                  angles="xy", scale_units="xy", scale=None, width=0.0022, alpha=0.35, color="k")

    ax.set_title(f"2D reconnection simulation (toy model)   t = {t:.4f}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_xlim(X.min(), X.max())
    ax.set_ylim(Y.min(), Y.max())

    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)



# simulation:

# main simulation function:
def run_reconnection_simulation(out_dir: str = "reconnection_out",
                             nx: int = 420,
                             ny: int = 260,
                             Lx: float = 2.0,
                             Ly: float = 2.0,
                             B0: float = 1.0,
                             delta_cs: float = 0.08,
                             a_pert: float = 0.06,
                             sigma_y: float = 0.20,
                             alpha: float = 1.2,
                             w_v: float = 0.65,
                             eta_bg: float = 8e-4,
                             eta_peak: float = 8e-3,
                             w_eta: float = 0.14,
                             t_end: float = 1.0,
                             cfl_adv: float = 0.35,
                             cfl_diff: float = 0.20,
                             frame_every: int = 1,
                             gif_fps: int = 28,
                             show_velocity: bool = False) -> None:
    out_path = Path(out_dir)
    frames_dir = out_path / "frames"
    ensure_dir(frames_dir)
    ensure_dir(out_path)

    # grid
    x = np.linspace(-Lx / 2.0, Lx / 2.0, nx, endpoint=False)
    y = np.linspace(-Ly / 2.0, Ly / 2.0, ny, endpoint=True)
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    X, Y = np.meshgrid(x, y)

    # prescribed inflow/outflow flow: inflow in y, outflow in x
    phi_v = np.exp(-(X * X + Y * Y) / (w_v * w_v))
    vx = (+alpha * X) * phi_v
    vy = (-alpha * Y) * phi_v

    # resistivity profile: background + localized enhancement (diffusion region)
    eta = eta_bg + eta_peak * np.exp(-(X * X + Y * Y) / (w_eta * w_eta))

    # initial A: Harris sheet + perturbation
    kx = 2.0 * np.pi / Lx
    A = A_harris(Y, B0=B0, delta_cs=delta_cs) + A_perturbation(X, Y, a=a_pert, kx=kx, sigma_y=sigma_y)

    # stable dt (explicit)
    vmax = max(np.max(np.abs(vx)), np.max(np.abs(vy)))
    if vmax == 0.0:
        dt_adv = np.inf
    else:
        dt_adv = cfl_adv * min(dx, dy) / (vmax + 1e-30)

    # diffusion stability uses max(eta)
    eta_max = float(np.max(eta))
    dt_diff = cfl_diff * min(dx * dx, dy * dy) / (eta_max + 1e-30)
    dt = min(dt_adv, dt_diff)

    n_steps = int(np.ceil(t_end / dt))
    mid_step = n_steps // 2

    initial_png = out_path / "initial.png"
    middle_png = out_path / "middle.png"
    final_png = out_path / "final.png"

    frame_paths: list[Path] = []

    def save_frame(step: int, t_now: float) -> None:
        fn = frames_dir / f"frame_{step:06d}.png"
        plot_frame(A, vx, vy, X, Y, dx, dy, t_now, str(fn),
                           diff_box=0.12, streamline_density=1.15,
                           show_velocity=show_velocity)
        frame_paths.append(fn)

    # initial snapshot + frame 0
    t = 0.0
    plot_frame(A, vx, vy, X, Y, dx, dy, t, str(initial_png),
                       diff_box=0.12, streamline_density=1.15,
                       show_velocity=show_velocity)
    if frame_every == 1:
        save_frame(0, t)

    # time stepping:
    for step in range(1, n_steps + 1):
        if t + dt > t_end:
            dt = t_end - t

        # advection terms
        Ax = ddx_periodic(A, dx)
        Ay = ddy_neumann(A, dy)

        adv = -(vx * Ax + vy * Ay)

        # diffusion term with spatially varying eta
        diff = grad_eta_dot_grad_A(eta, A, dx, dy)

        # explicit Euler update
        A = A + dt * (adv + diff)
        t += dt

        if step == mid_step:
            plot_frame(A, vx, vy, X, Y, dx, dy, t, str(middle_png),
                               diff_box=0.12, streamline_density=1.15,
                               show_velocity=show_velocity)

        if step % frame_every == 0:
            save_frame(step, t)
            
        # reconnection diagnostic at X-point:
        if step % 100 == 0:
            iy, ix = np.unravel_index(np.argmin(X**2 + Y**2), X.shape)
            Ez = eta[iy, ix] * compute_Jz(A, dx, dy)[iy, ix]
            print(f"step={step}, t={t:.3f}, Ez={Ez:.3e}")

    # final snapshot:
    plot_frame(A, vx, vy, X, Y, dx, dy, t, str(final_png),
                       diff_box=0.12, streamline_density=1.15,
                       show_velocity=show_velocity)

    # build gif
    gif_path = out_path / "reconnection.gif"
    images = [imageio.imread(p) for p in frame_paths]
    imageio.mimsave(gif_path, images, fps=gif_fps)

    print("Done.")
    print(f"Output directory: {out_path.resolve()}")
    print(f"GIF: {gif_path.resolve()}")
    print(f"Initial: {initial_png.resolve()}")
    print(f"Middle: {middle_png.resolve()}")
    print(f"Final: {final_png.resolve()}")
    print(f"Frames: {frames_dir.resolve()} (count: {len(frame_paths)})")
    print(f"dt used ~ {t_end / max(n_steps,1):.3e}  steps: {n_steps}")


# %% MAIN RUN
run_reconnection_simulation(
        out_dir="reconnection_out", # defined output directory
        nx=420,                     # resolution: grid points in x
        ny=260,                     # resolution: grid points in y
        Lx=2.0,                     # box size in x (box defines the central region)
        Ly=2.0,                     # box size in y
        B0=1.0,                     # asymptotic magnetic field strength
        delta_cs=0.08,              # current sheet half-thickness
        a_pert=0.20,                # perturbation amplitude; use values between 0.06 and 0.20; higher values lead to faster reconnection onset
        sigma_y=0.08,               # perturbation width in y; smaller values lead to faster reconnection onset
        alpha=1.4,                  # inflow/outflow strength
        w_v=0.65,                   # inflow/outflow width scale
        eta_bg=8e-4,                # background resistivity
        eta_peak=8.0e-2, # or 8e-3  # peak resistivity in diffusion region
        w_eta=0.05,      # or 0.14  # resistivity enhancement width scale
        t_end=4.0,       # or 3.0   # end time of simulation
        frame_every=100,            # save every N steps; the higher the faster the simulation
        gif_fps=28,                 # fps for the output gif
        show_velocity=False,        # whether to show velocity quiver
    )
# %% END
print("done.")
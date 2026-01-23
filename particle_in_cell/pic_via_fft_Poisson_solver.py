"""
Minimal 1D electrostatic Particle in Cell (PIC) code using FFT-based Poisson solver.

Goal:
* 1D periodic electrostatic PIC for the two stream instability.
* Uses CIC deposition and CIC field interpolation.
* Solves Poisson via FFT in the same "n - n0" style as many textbook PIC demos.

Key features / simplifications:
* Deposit number density n(x) (positive), not charge density rho.
* Solve Poisson in normalized form:  phi'' = -(n - n0)
  (equivalently: (-k^2) phi_k = (n - n0)_k, hence phi_k = -(n - n0)_k / k^2)
* E = - dphi/dx.
* Electrons: q/m = -1  ->  a = (q/m) E = -E.
* Leapfrog pusher (electrostatic Boris specialization).

author: Fabrizio Musacchio
date: Oct 2020
"""

# %% IMPORTS
import os
import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio

# remove spines right and top for better aesthetics:
plt.rcParams["axes.spines.right"] = False
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.left"] = False
plt.rcParams["axes.spines.bottom"] = False
plt.rcParams.update({"font.size": 12})

# %% FUNCTIONS
def cic_deposit_number_density(xp, Nx, L):
    """
    CIC number density deposition onto a uniform grid.
    CIC stands for Cloud-In-Cell, a linear weighting scheme.

    xp : particle positions in [0, L)
    returns:
        n on grid (length Nx), normalized so mean(n) ~ 1
    """
    dx = L / Nx

    g = xp / dx
    i = np.floor(g).astype(int)
    f = g - i

    i0 = i % Nx
    i1 = (i + 1) % Nx

    w0 = 1.0 - f
    w1 = f

    n = np.zeros(Nx, dtype=np.float64)
    np.add.at(n, i0, w0)
    np.add.at(n, i1, w1)

    # convert counts to number density:
    n /= dx

    # normalize so that mean(n) = 1 (for uniform xp distribution).
    # for uniform distribution, expected mean of (counts/dx) is Np/L.
    # multiply by L/Np to get mean ~ 1:
    Np = xp.size
    n *= (L / Np)

    return n


def cic_interpolate_field(Eg, xp, Nx, L):
    """
    CIC interpolation of grid field Eg to particle positions xp.
    """
    dx = L / Nx

    g = xp / dx
    i = np.floor(g).astype(int)
    f = g - i

    i0 = i % Nx
    i1 = (i + 1) % Nx

    w0 = 1.0 - f
    w1 = f

    Ep = Eg[i0] * w0 + Eg[i1] * w1
    return Ep


def poisson_solve_fft_from_n(n, n0, L):
    """
    Solve 1D periodic Poisson in normalized form:
        phi'' = -(n - n0)
    then E = -phi'.

    In Fourier space:
        (-k^2) phi_k = (n - n0)_k   =>  phi_k = -(n - n0)_k / k^2  for k != 0

    Returns:
        Eg : electric field on grid (length Nx)
        phi: potential on grid (length Nx)
    """
    Nx = n.size
    dx = L / Nx

    # Fourier wavenumbers (rad / length):
    k = 2.0 * np.pi * np.fft.fftfreq(Nx, d=dx)

    rhs = n - n0
    rhs_k = np.fft.fft(rhs)
    rhs_k[0] = 0.0 + 0.0j  # remove k=0 mode (neutrality)

    phi_k = np.zeros_like(rhs_k)
    mask = (k != 0.0)
    phi_k[mask] = -rhs_k[mask] / (k[mask] ** 2)

    # E_k = - i k phi_k:
    E_k = -1j * k * phi_k

    phi = np.fft.ifft(phi_k).real
    Eg = np.fft.ifft(E_k).real
    return Eg, phi


def initialize_two_stream(Np, L, vth, v0, seed=0, A=0.05):
    """
    Two drifting Maxwellians:
    * half particles with drift +v0, half with drift -v0, same thermal spread vth
    * positions uniform on [0, L)
    * apply a small sinusoidal velocity perturbation to seed the instability
    """
    rng = np.random.default_rng(seed)
    xp = rng.random(Np) * L

    vp = rng.normal(loc=0.0, scale=vth, size=Np)
    half = Np // 2
    vp[:half] += v0
    vp[half:] -= v0

    # deterministic seed (Mocz-style):
    vp *= (1.0 + A * np.sin(2.0 * np.pi * xp / L))

    return xp, vp


def make_gif_from_folder(folder, gif_name, fps=10):
    files = sorted([os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".png")])
    frames = [imageio.imread(f) for f in files]
    imageio.mimsave(gif_name, frames, fps=fps)


def run_pic_1d_es(
    *,
    Nx=256,
    Np=30000,
    L=2.0 * np.pi,
    dt=0.03,
    nsteps=4000,
    # normalized electron q/m:
    q_over_m=-1.0,
    # background density
    n0=1.0,
    # initial condition params
    vth=0.25,
    v0=1.0,
    A=0.05,
    seed=1,
    # output
    results_path="pic_frames_fft_fixed",
    plot_every=200,
    diag_every=10,
    fps=10):
    os.makedirs(results_path, exist_ok=True)

    dx = L / Nx

    # initialize particles:
    xp, vp = initialize_two_stream(Np, L, vth=vth, v0=v0, seed=seed, A=A)

    # initial fields:
    n_grid = cic_deposit_number_density(xp, Nx, L)
    Eg, _ = poisson_solve_fft_from_n(n_grid, n0=n0, L=L)
    Ep = cic_interpolate_field(Eg, xp, Nx, L)

    # leapfrog init: velocities at half step:
    a = q_over_m * Ep
    v_half = vp + a * (0.5 * dt)

    # prepare diagnostics storage:
    t_hist = []
    Wk_hist = []
    We_hist = []
    Wtot_hist = []

    fig = None
    rng_plot = np.random.default_rng(seed + 123)

    for step in range(1, nsteps + 1):
        # drift: x^{n+1} = x^n + v^{n+1/2} dt
        xp = xp + v_half * dt
        xp %= L

        # deposit n(x):
        n_grid = cic_deposit_number_density(xp, Nx, L)

        # field solve:
        Eg, _ = poisson_solve_fft_from_n(n_grid, n0=n0, L=L)

        # gather: E(xp)
        Ep = cic_interpolate_field(Eg, xp, Nx, L)

        # kick: v^{n+3/2} = v^{n+1/2} + (q/m) E dt:
        a = q_over_m * Ep
        v_half = v_half + a * dt

        # diagnostics every diag_every steps:
        if (step % diag_every) == 0:
            # reconstruct v at integer time for diagnostics:
            # v^n = v^{n+1/2} - (q/m) E(x^n) dt/2
            vp_diag = v_half - a * (0.5 * dt)

            # energies (normalized):
            # kinetic: sum 1/2 v^2 (mass=1)
            Wk = 0.5 * np.sum(vp_diag ** 2)

            # field energy: 1/2 ∫ E^2 dx:
            We = 0.5 * np.sum(Eg ** 2) * dx

            t = step * dt
            t_hist.append(t)
            Wk_hist.append(Wk)
            We_hist.append(We)
            Wtot_hist.append(Wk + We)

        # plotting:
        if (step % plot_every) == 0 or step == 1:
            vp_plot = v_half - a * (0.5 * dt)

            if fig is None:
                fig = plt.figure(figsize=(11, 8))
            plt.clf()

            ax1 = plt.subplot2grid((2, 2), (0, 0))
            ax2 = plt.subplot2grid((2, 2), (0, 1))
            ax3 = plt.subplot2grid((2, 2), (1, 0), colspan=2)

            # phase space (subsample):
            n_show = min(6000, Np)
            idx = rng_plot.choice(Np, size=n_show, replace=False)

            # define once, right after initialization (outside the loop):
            beam_id = np.zeros(Np, dtype=int)
            beam_id[:Np//2] = 0
            beam_id[Np//2:] = 1

            # inside plotting, after choosing idx:
            mask0 = beam_id[idx] == 0
            mask1 = ~mask0
            ax1.scatter(xp[idx][mask0], vp_plot[idx][mask0], s=1, alpha=0.5, color="tab:blue")
            ax1.scatter(xp[idx][mask1], vp_plot[idx][mask1], s=1, alpha=0.5, color="magenta")
            ax1.set_xlabel("x")
            ax1.set_ylabel("v")
            ax1.set_title(f"phase space at t={step*dt:.2f}")

            # field E(x):
            xg = np.linspace(0.0, L, Nx, endpoint=False)
            ax2.plot(xg, Eg)
            ax2.set_xlabel("x")
            ax2.set_ylabel("E(x)")
            ax2.set_title("electric field")

            # energy history:
            if len(t_hist) > 2:
                ax3.plot(t_hist, Wk_hist, label="$W_k$ kinetic", c="tab:blue", lw=3.5)
                #ax3.plot(t_hist, We_hist,   label="$W_e$ field", c="tab:green")
                ax3.plot(t_hist, Wtot_hist, '--', label="$W_{tot}=W_k + W_e$ total", c="tab:olive")
                ax3.set_xlabel("t")
                ax3.set_ylabel("energy (kinetic / total)")
                ax3.set_ylim(0, 21000)
                
                # plot W_e on secondary y-axis:
                ax3r = ax3.twinx()
                ax3r.plot(t_hist, We_hist, label="$W_e$ field", c="tab:orange")
                ax3r.set_ylabel("field energy")
                ax3r.set_ylim(0,3)
                
                # combined legend:
                lines_l, labels_l = ax3.get_legend_handles_labels()
                lines_r, labels_r = ax3r.get_legend_handles_labels()
                ax3.legend(lines_l + lines_r, labels_l + labels_r, loc="center right", frameon=False)
                
                # color the right y-axis label to match the line color:
                ax3r.yaxis.label.set_color("tab:orange")
                ax3r.tick_params(axis="y", colors="tab:orange")
                
                ax3.set_title("energy diagnostics")

            plt.tight_layout()
            plt.savefig(os.path.join(results_path, f"pic_{step:05d}.png"), dpi=150)

    plt.close(fig)

    # GIF
    # for storing the gif, the go the parent folder of results_path:
    print("saving gif...", end="")
    GIF_results_path = os.path.dirname(results_path)
    make_gif_from_folder(results_path, gif_name=os.path.join(GIF_results_path, "pic_evolution.gif"), fps=fps)
    print("done.")

    return {
        "t": np.array(t_hist),
        "Wk": np.array(Wk_hist),
        "We": np.array(We_hist),
        "Wtot": np.array(Wtot_hist),
    }


# %% MAIN
out = run_pic_1d_es(
    Nx=512,             # grid points, e.g., 256; the higher the better resolution but more compute
    Np=40000,           # number of particles
    L=6.0 * np.pi,      # box size
    dt=0.01,            # timestep
    nsteps=80000,       # number of timesteps (simulation time = nsteps*dt)
    q_over_m=-1.0,      # normalized electron charge/mass
    n0=1.0,             # background density
    vth=0.15,           # thermal velocity, e.g., 0.25
    v0=1.0,             # drift velocity of the two streams, e.g., 1.0
    A=0.05,             # amplitude of initial perturbation
    seed=42,            # random seed
    results_path="pic_frames_fft",
    plot_every=200,     # plot every N steps
    diag_every=10,      # diagnostics every N steps
    fps=10,             # frames per second for gif
)
# %% END
""" 
This script implements a 1D1V Vlasov-Poisson solver using Strang splitting and 
semi-Lagrangian advection. It can simulate both Landau damping and the two-stream instability.

The equations solved are:
    ∂f/∂t + v ∂f/∂x + E ∂f/∂v = 0
    dE/dx = 1 - n(x),   n(x) = ∫ f dv

The solver uses semi-Lagrangian steps for advection in x and v, and solves Poisson's 
equation in Fourier space.

author: Fabrizio Musacchio
date: Aug 2020 / Jan 2026
"""
# %% IMPORTS
import numpy as np
import matplotlib.pyplot as plt

# remove spines right and top for better aesthetics:
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.left'] = False
plt.rcParams['axes.spines.bottom'] = False
plt.rcParams.update({'font.size': 12})
# %% 1D1V VLASOV-POISSON SOLVER WITH STRANG SPLITTING

def maxwellian(v, vt=1.0):
    """1D Maxwellian with thermal speed vt (normalized)."""
    return np.exp(-v**2 / (2.0 * vt**2)) / (np.sqrt(2.0 * np.pi) * vt)


def init_distribution(X, V, mode="landau", alpha=0.01, k=0.5, vt=1.0, u=2.4):
    """
    Initialize f(x,v,0) for Landau damping or two-stream instability.
    X, V are meshgrids with shape (Nv, Nx).
    """
    if mode == "landau":
        f = (1.0 + alpha * np.cos(k * X)) * maxwellian(V, vt=vt)
        return f

    if mode == "twostream":
        f0 = 0.5 * maxwellian(V - u, vt=vt) + 0.5 * maxwellian(V + u, vt=vt)
        f = (1.0 + alpha * np.cos(k * X)) * f0
        return f

    raise ValueError("mode must be 'landau' or 'twostream'")


def advect_x_semi_lagrange(f, x, v, dt, Lx):
    """
    Semi-Lagrangian advection in x:
    f_t + v f_x = 0  ->  f(x,v,t+dt)=f(x - v dt, v, t)
    Periodic in x, linear interpolation.
    f has shape (Nv, Nx).
    """
    Nv, Nx = f.shape
    dx = Lx / Nx
    out = np.empty_like(f)

    for j in range(Nv):
        x_back = (x - v[j] * dt) % Lx
        xi = x_back / dx
        i0 = np.floor(xi).astype(int)
        frac = xi - i0
        i1 = (i0 + 1) % Nx
        out[j, :] = (1.0 - frac) * f[j, i0] + frac * f[j, i1]

    return out


def advect_v_semi_lagrange(f, v, E_x, dt):
    """
    Semi-Lagrangian advection in v:
    f_t + E f_v = 0  ->  f(x,v,t+dt)=f(x, v - E dt, t)
    Here E_x depends on x only.
    We use linear interpolation in v with zero-fill outside [-vmax, vmax].
    f has shape (Nv, Nx), E_x has shape (Nx,).
    """
    Nv, Nx = f.shape
    out = np.empty_like(f)

    vmin = v[0]
    vmax = v[-1]

    # For each x-column, shift the velocity grid by E(x) dt
    for i in range(Nx):
        v_back = v - E_x[i] * dt
        # np.interp requires increasing x, and returns boundary values outside by default
        # We want zero outside range, so we mask explicitly.
        col = np.interp(v_back, v, f[:, i], left=0.0, right=0.0)
        out[:, i] = col

    return out


def electric_field_from_density(n_x, Lx):
    """
    Solve Poisson in Fourier space (periodic, zero-mean potential).
    Normalization:
      dE/dx = 1 - n  (immobile ion background of density 1)
      equivalently: d^2 phi/dx^2 = n - 1, E = -dphi/dx
    """
    Nx = n_x.size
    dx = Lx / Nx

    rho = n_x - 1.0  # charge density (up to sign conventions)

    rho_k = np.fft.rfft(rho)
    k = 2.0 * np.pi * np.fft.rfftfreq(Nx, d=dx)

    phi_k = np.zeros_like(rho_k, dtype=np.complex128)
    # avoid k=0
    phi_k[1:] = -rho_k[1:] / (k[1:] ** 2)

    E_k = 1j * k * phi_k
    E_x = np.fft.irfft(E_k, n=Nx)

    # Enforce zero-mean E (should already hold if rho has zero mean)
    E_x -= np.mean(E_x)

    return E_x


def compute_density(f, v):
    """n(x) = ∫ f dv."""
    dv = v[1] - v[0]
    return np.sum(f, axis=0) * dv


def field_energy(E_x, Lx):
    """W_E = 1/2 ∫ E^2 dx."""
    dx = Lx / E_x.size
    return 0.5 * np.sum(E_x**2) * dx


def run_vlasov_poisson(
    mode="landau",
    Nx=128,
    Nv=256,
    Lx=4.0 * np.pi,
    vmax=6.0,
    dt=0.05,
    nsteps=400,
    alpha=0.01,
    k_mode=1,
    vt=1.0,
    u=2.4,
    snapshot_steps=(0, 100, 200, 400),
):
    """
    Run 1D1V Vlasov-Poisson with Strang splitting:
      x half-step -> solve E -> v full-step -> x half-step
    """
    x = np.linspace(0.0, Lx, Nx, endpoint=False)
    v = np.linspace(-vmax, vmax, Nv)
    X, V = np.meshgrid(x, v, indexing="xy")  # shape (Nv, Nx)

    k = 2.0 * np.pi * k_mode / Lx

    f = init_distribution(X, V, mode=mode, alpha=alpha, k=k, vt=vt, u=u)

    times = []
    energies = []
    snapshots = {}

    # initial diagnostics
    n_x = compute_density(f, v)
    E_x = electric_field_from_density(n_x, Lx)
    times.append(0.0)
    energies.append(field_energy(E_x, Lx))
    if 0 in snapshot_steps:
        snapshots[0] = f.copy()

    for step in range(1, nsteps + 1):
        # x half-step
        f = advect_x_semi_lagrange(f, x, v, 0.5 * dt, Lx)

        # field from updated density
        n_x = compute_density(f, v)
        E_x = electric_field_from_density(n_x, Lx)

        # v full-step
        f = advect_v_semi_lagrange(f, v, E_x, dt)

        # x half-step
        f = advect_x_semi_lagrange(f, x, v, 0.5 * dt, Lx)

        t = step * dt
        times.append(t)

        # diagnostics after full step
        n_x = compute_density(f, v)
        E_x = electric_field_from_density(n_x, Lx)
        energies.append(field_energy(E_x, Lx))

        if step in snapshot_steps:
            snapshots[step] = f.copy()

    return x, v, times, energies, snapshots


def plot_results(x, v, times, energies, snapshots, Lx, vmax, mode):
    # field energy plot:
    fig1, ax1 = plt.subplots(figsize=(6.5, 4.2))
    ax1.plot(times, energies, lw=2)
    ax1.set_xlabel("time")
    ax1.set_ylabel(r"$W_E(t)=\frac{1}{2}\int E^2\,dx$")
    ax1.set_title(f"Vlasov–Poisson: {mode}")
    ax1.set_yscale("log")
    ax1.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(f"vlasov_poisson_energy_{mode}.png", dpi=300)
    #plt.show()
    plt.close()

    # phase space snapshots:
    steps_sorted = sorted(snapshots.keys())
    ncols = 2
    nrows = int(np.ceil(len(steps_sorted) / ncols))
    fig2, axes = plt.subplots(nrows, ncols, figsize=(8.0, 3.6 * nrows), constrained_layout=True)
    axes = axes.flatten()
    if ncols == 1:
        axes = [axes]

    X, V = np.meshgrid(x, v, indexing="xy")

    for ax, step in zip(axes, steps_sorted):
        f = snapshots[step]
        im = ax.imshow(
            f,
            origin="lower",
            aspect="auto",
            extent=[0.0, Lx, -vmax, vmax],)
        ax.set_title(f"step {step}")
        ax.set_xlabel("x")
        ax.set_ylabel("v")
        fig2.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    plt.savefig(f"vlasov_poisson_phase_space_{mode}.png", dpi=300)
    plt.close()
    
    # plot perturbation df = f - <f>_x at final time:
    # f shape (Nv, Nx)
    f_mean_x = np.mean(f, axis=1, keepdims=True)   # mean over x, keeps v-dependence
    df = f - f_mean_x
    plt.figure(figsize=(6,4))
    plt.imshow(
        df,
        origin="lower",
        aspect="auto",
        extent=[0.0, Lx, -vmax, vmax])
    plt.colorbar(label=r"$\delta f = f - \langle f\rangle_x$")
    plt.xlabel("x")
    plt.ylabel("v")
    plt.title("Landau: perturbation in phase space")
    plt.savefig(f"vlasov_poisson_perturbation_{mode}.png", dpi=300)
    #plt.show()
    plt.close()


# %% MAIN

# Choose mode: "landau" or "twostream"
mode = "landau"
#mode = "twostream"

# Landau: small alpha, moderate k, vt=1
# Two-stream: larger alpha helps trigger growth, choose u large enough
if mode == "landau":
    params = dict(alpha=0.1, k_mode=1, vt=1.0, u=0.0)
    dt = 0.05
    nsteps = 500
    snapshots = (0, 100, 250, 500)
else:
    params = dict(alpha=0.1, k_mode=1, vt=1.0, u=2.4)
    dt = 0.04
    nsteps = 600
    snapshots = (0, 150, 300, 600)

Lx = 4.0 * np.pi
vmax = 6.0

x, v, times, energies, snaps = run_vlasov_poisson(
    mode=mode,
    Nx=128,
    Nv=256,
    Lx=Lx,
    vmax=vmax,
    dt=dt,
    nsteps=nsteps,
    snapshot_steps=snapshots,
    **params)

plot_results(x, v, times, energies, snaps, Lx=Lx, vmax=vmax, mode=mode)
# %% END
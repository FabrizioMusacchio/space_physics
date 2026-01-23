""" 
Linearized 1D MHD (transverse Alfven dynamics) with various boundary conditions.

As integration method, we use a simple explicit RK2 (Heun) scheme:

    k1 = rhs(u,b)
    u1 = u + dt*k1u
    b1 = b + dt*k1b

    k2 = rhs(u1,b1)
    u = u + 0.5*dt*(k1u + k2u)
    b = b + 0.5*dt*(k1b + k2b)

For diagnostics, we produces: (1) spacetime heatmaps, (2) snapshots, (3) energy budget

author: Fabrizio Musacchio
date: June 2024
"""
# %% IMPORTS
import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.fftpack

RESULTS_FOLDER = "alfven_wave_1D_results"
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# remove spines right and top for better aesthetics:
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.left'] = False
plt.rcParams['axes.spines.bottom'] = False
plt.rcParams.update({'font.size': 12})
# %% PARAMETERS

# we set necessary physical parameters:
mu0 = 4e-7 * np.pi

B0 = 50e-9            # T
rho0 = 5.0 * 1.6726e-27 * 1e6  # kg/m^3   (example: 5 cm^-3 protons)
vA = B0 / np.sqrt(mu0 * rho0)

nu = 0.0              # m^2/s (set >0 for viscous damping)
eta = 0.0             # m^2/s (set >0 for resistive damping)

print(f"vA = {vA:.2e} m/s")


# set the numerical parameters for the 1D grid and time integration:
L = 2.0e7             # m
Nx = 1200
dx = L / Nx
x = np.linspace(0, L, Nx, endpoint=False)

# CFL-type stability constraints for explicit scheme
# wave: dt < dx/vA
# diffusion: dt < 0.5 * dx^2 / max(nu,eta) (roughly)
cfl = 0.35
dt_wave = cfl * dx / max(vA, 1e-30)

diff = max(nu, eta)
dt_diff = 0.25 * dx * dx / diff if diff > 0 else np.inf

dt = min(dt_wave, dt_diff)
T = 60.0              # seconds
Nt = int(np.ceil(T / dt))
dt = T / Nt

print(f"dx={dx:.2e} m, dt={dt:.2e} s, Nt={Nt}")

# choose the boundary conditions: "periodic", "reflecting", "open"
BC = "reflecting"
#BC = "periodic"
#BC = "open"
# %% HELPER FUNCTIONS

# envelope function for plotting:
def local_rms(f, win):
    """
    Local RMS envelope sqrt( <f^2>_win ), using a simple moving average.
    win: window length in grid points (odd recommended).
    """
    win = int(max(3, win))
    if win % 2 == 0:
        win += 1
    kernel = np.ones(win) / win
    return np.sqrt(np.convolve(f*f, kernel, mode="same"))

# window around peak:
def window_around_peak(f, half_width_pts):
    """
    Returns slice indices [i0:i1] around the peak of |f|.
    """
    i_peak = int(np.argmax(np.abs(f)))
    i0 = max(0, i_peak - half_width_pts)
    i1 = min(len(f), i_peak + half_width_pts)
    return i0, i1

# calculate mean spatial frequency:
def mean_spatial_frequency(u_x, dx):
    """
    PSD-weighted mean spatial frequency in cycles/m for a spatial snapshot u(x).
    """
    fhat = scipy.fftpack.fft(u_x - np.mean(u_x))
    psd = np.abs(fhat)**2
    k_cyc = scipy.fftpack.fftfreq(len(u_x), d=dx)  # cycles/m

    pos = k_cyc > 0
    k_cyc = k_cyc[pos]
    psd = psd[pos]

    return np.sum(k_cyc * psd) / (np.sum(psd) + 1e-30)
# %% DERIVATIVE OPERATORS WITH BOUNDARY CONDITIONS
# define the spatial derivative operators with chosen BCs:

def ddx(f):
    """Centered first derivative with chosen boundary."""
    if BC == "periodic":
        return (np.roll(f, -1) - np.roll(f, 1)) / (2 * dx)

    # nonperiodic: use ghost cells via padding
    fpad = np.empty(Nx + 2)
    fpad[1:-1] = f

    if BC == "reflecting":
        # Neumann: df/dx = 0 at boundaries -> mirror values
        fpad[0] = fpad[2]
        fpad[-1] = fpad[-3]
    elif BC == "open":
        # simple radiation-like copy (weakly absorbing, not perfect)
        fpad[0] = fpad[1]
        fpad[-1] = fpad[-2]
    else:
        raise ValueError("Unknown BC")

    return (fpad[2:] - fpad[:-2]) / (2 * dx)

def ddxx(f):
    """Centered second derivative with chosen boundary."""
    if BC == "periodic":
        return (np.roll(f, -1) - 2 * f + np.roll(f, 1)) / (dx * dx)

    fpad = np.empty(Nx + 2)
    fpad[1:-1] = f

    if BC == "reflecting":
        fpad[0] = fpad[2]
        fpad[-1] = fpad[-3]
    elif BC == "open":
        fpad[0] = fpad[1]
        fpad[-1] = fpad[-2]
    else:
        raise ValueError("Unknown BC")

    return (fpad[2:] - 2 * fpad[1:-1] + fpad[:-2]) / (dx * dx)
# %% INITIAL CONDITIONS

# Set the initial condition: localized wave packet

# Use Elsasser variables: z+ = u - b/sqrt(mu0 rho0), z- = u + b/sqrt(mu0 rho0)
# Pure right-going wave:  z- = 0  -> b = -sqrt(mu0 rho0) * u
# Pure left-going wave:   z+ = 0  -> b = +sqrt(mu0 rho0) * u

u = np.zeros(Nx)
b = np.zeros(Nx)

x0 = 0.35 * L
sigma = 0.05 * L
k0 = 2 * np.pi / (0.10 * L)     # sets internal oscillations in the packet
A = 30.0                        # m/s amplitude

packet = np.exp(-0.5 * ((x - x0) / sigma) ** 2) * np.cos(k0 * (x - x0))
u[:] = A * packet

# choose propagation direction: "right" or "left" or "both"
direction = "right"

if direction == "right":
    b[:] = -np.sqrt(mu0 * rho0) * u
elif direction == "left":
    b[:] = +np.sqrt(mu0 * rho0) * u
elif direction == "both":
    # excite both z+ and z- by setting b=0 initially
    b[:] = 0.0
else:
    raise ValueError("direction must be right/left/both")
# %% TIME INTEGRATION
# time integration: RK2 (Heun)

# define the right-hand side function to be integrated:
def rhs(u, b):
    uy_t = (B0 / (mu0 * rho0)) * ddx(b) + nu * ddxx(u)
    by_t = B0 * ddx(u) + eta * ddxx(b)
    return uy_t, by_t

# collectors for spacetime plots (downsample in time):
store_every = max(1, Nt // 400)
u_hist = []
b_hist = []
t_hist = []

# energy diagnostics:
E_kin = np.zeros(Nt + 1)
E_mag = np.zeros(Nt + 1)
E_tot = np.zeros(Nt + 1)

def energies(u, b):
    # energy densities: (1/2) rho u^2 , (1/2mu0) b^2
    ek = 0.5 * rho0 * np.mean(u * u)
    em = 0.5 / mu0 * np.mean(b * b)
    return ek, em, ek + em

ek0, em0, et0 = energies(u, b)
E_kin[0], E_mag[0], E_tot[0] = ek0, em0, et0

# main time loop:
for n in range(1, Nt + 1):
    # RK2 step:
    k1u, k1b = rhs(u, b)
    u1 = u + dt * k1u
    b1 = b + dt * k1b

    k2u, k2b = rhs(u1, b1)
    u = u + 0.5 * dt * (k1u + k2u)
    b = b + 0.5 * dt * (k1b + k2b)

    # compute energies:
    ek, em, et = energies(u, b)
    E_kin[n], E_mag[n], E_tot[n] = ek, em, et

    # store for plotting:
    if (n % store_every) == 0:
        u_hist.append(u.copy())
        b_hist.append(b.copy())
        t_hist.append(n * dt)

u_hist = np.array(u_hist)
b_hist = np.array(b_hist)
t_hist = np.array(t_hist)
# %% PLOTTING

# plot 1: spacetime heatmaps
plt.figure(figsize=(6.0, 5.0))
plt.imshow(u_hist, aspect="auto", origin="lower",
           extent=[x[0], x[-1] + dx, t_hist[0], t_hist[-1]],
           cmap="RdBu_r", vmin=-np.max(np.abs(u_hist)), vmax=np.max(np.abs(u_hist)))
plt.xlabel("$x$ [m]")
plt.ylabel("$t$ [s]")
plt.title("Linearized 1D MHD: $u_y(x,t)$ spacetime")
plt.colorbar(label="$u_y$ [m/s]")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_FOLDER, f"alfven_wave_1D_uy_spacetime_BC_{BC}.png"), dpi=300)
plt.close()

plt.figure(figsize=(6.0, 5.0))
plt.imshow(b_hist / 1e-9, aspect="auto", origin="lower",
           extent=[x[0], x[-1] + dx, t_hist[0], t_hist[-1]],
           cmap="RdBu_r", vmin=-np.max(np.abs(b_hist)) / 1e-9, vmax=np.max(np.abs(b_hist)) / 1e-9)
plt.xlabel("$x$ [m]")
plt.ylabel("$t$ [s]")
plt.title("Linearized 1D MHD: $b_y(x,t)$ spacetime")
plt.colorbar(label="$b_y$ [nT]")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_FOLDER, f"alfven_wave_1D_by_spacetime_BC_{BC}.png"), dpi=300)
plt.close()



# plot 2: snapshots
snap_ids = np.linspace(0, len(t_hist) - 1, 6, dtype=int)

# choose a wavelength threshold for "highly oscillatory"
lambda_thresh = 0.02 * L          # e.g. 2% of domain length; adjust
k_thresh_cyc = 1.0 / lambda_thresh # cycles/m

# plot u snapshots:
plt.figure(figsize=(9.0, 3.6))
hi_u = []
hi_u_color = []
for i in snap_ids:
    plt.plot(x, u_hist[i], label=f"t={t_hist[i]:.2f} s", alpha=0.7)
    
    # get the spatial frequency of the current u_hist[i] using e.g., FFT;
    kmean = mean_spatial_frequency(u_hist[i], dx)   # cycles/m
    if kmean > k_thresh_cyc:
        hi_u.append(i)
        hi_u_color.append(plt.gca().lines[-1].get_color())
plt.xlabel("x [m]")
plt.ylabel(r"$u_y$ [m/s]")
plt.xlim(0, L)
plt.title(f"Linearized 1D MHD: transverse velocity snapshots\nBC={BC}")
plt.legend(fontsize=8, ncols=2, loc="upper right")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_FOLDER, f"alfven_wave_1D_uy_snapshots_BC_{BC}.png"), dpi=300)
plt.close()

# zoomed plots for high-k snapshots (use RMS only to find packet center)
for i in hi_u:
    win_pts = int(0.02 * Nx)
    u_rms = local_rms(u_hist[i], win_pts)
    i0, i1 = window_around_peak(u_rms, half_width_pts=int(0.325 * Nx))

    plt.figure(figsize=(12.0, 3.0))  # elongated
    plt.plot(x[i0:i1], u_hist[i][i0:i1], label=f"t={t_hist[i]:.2f} s", alpha=0.9,
             color=hi_u_color[hi_u.index(i)])
    # also plot the envelope:
    plt.plot(x[i0:i1], u_rms[i0:i1], "k--", label="RMS envelope")
    plt.plot(x[i0:i1], -u_rms[i0:i1], "k--")
    plt.xlabel("x [m]")
    plt.ylabel(r"$u_y$ [m/s]")
    plt.xlim(x[i0], x[i1-1])
    plt.title(f"Linearized 1D MHD: zoomed u_y snapshot (high spatial k)\nBC={BC}")
    plt.legend(fontsize=8, loc="upper right")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_FOLDER, f"alfven_wave_1D_uy_zoom_BC_{BC}_snap{i}.png"), dpi=300)
    plt.close()

# plot b snapshots:
plt.figure(figsize=(9.0, 3.6))
hi_b = []
hi_b_color = []
for i in snap_ids:
    plt.plot(x, b_hist[i] / 1e-9, label=f"t={t_hist[i]:.2f} s")
    
    kmean = mean_spatial_frequency(b_hist[i], dx)  # cycles/m
    if kmean > k_thresh_cyc:
        hi_b.append(i)
        hi_b_color.append(plt.gca().lines[-1].get_color())
plt.xlabel("x [m]")
plt.ylabel(r"$b_y$ [nT]")
plt.xlim(0, L)
plt.title(f"Linearized 1D MHD: transverse magnetic field snapshots\nBC={BC}")
plt.legend(fontsize=8, ncols=2, loc="upper right")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_FOLDER, f"alfven_wave_1D_by_snapshots_BC_{BC}.png"), dpi=300)
plt.close()

for i in hi_b:
    win_pts = int(0.02 * Nx)
    b_rms = local_rms(b_hist[i], win_pts)
    i0, i1 = window_around_peak(b_rms, half_width_pts=int(0.325 * Nx))

    plt.figure(figsize=(12.0, 3.0))  # elongated
    plt.plot(x[i0:i1], b_hist[i][i0:i1] / 1e-9, label=f"t={t_hist[i]:.2f} s", alpha=0.9,
             color=hi_b_color[hi_b.index(i)])
    # also plot the envelope:
    plt.plot(x[i0:i1], b_rms[i0:i1] / 1e-9, "k--", label="RMS envelope")
    plt.plot(x[i0:i1], -b_rms[i0:i1] / 1e-9, "k--")
    plt.xlabel("x [m]")
    plt.ylabel(r"$b_y$ [nT]")
    plt.xlim(x[i0], x[i1-1])
    plt.title(f"Linearized 1D MHD: zoomed b_y snapshot (high spatial k)\nBC={BC}")
    plt.legend(fontsize=8, loc="upper right")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_FOLDER, f"alfven_wave_1D_by_zoom_BC_{BC}_snap{i}.png"), dpi=300)
    plt.close()



# plot 3: energy budget
t = np.linspace(0, T, Nt + 1)

plt.figure(figsize=(5.2, 3.6))
plt.plot(t, (E_tot - E_tot[0]) / E_tot[0])
plt.xlabel("$t$ [s]")
plt.ylabel(r"$\frac{E(t)-E(0)}{E(0)}$")
plt.title("Total energy: relative change\n(ideal ~ 0; dissipative < 0)")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_FOLDER, f"alfven_wave_1D_energy_budget_BC_{BC}.png"), dpi=300)
plt.close()

plt.figure(figsize=(5.2, 3.6))
plt.plot(t, E_kin / E_tot[0], label="kinetic")
plt.plot(t, E_mag / E_tot[0], "--", label="magnetic")
plt.xlabel("$t$ [s]")
plt.ylabel("energy / $E(0)$")
plt.title("Energy partition")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_FOLDER, f"alfven_wave_1D_energy_partition_BC_{BC}.png"), dpi=300)
plt.close()
# %% END
""" 
A simple 1D1V Vlasov equation solver using a semi-Lagrangian scheme.
The Vlasov equation in 1D1V without forces is:
    ∂f/∂t + v ∂f/∂x = 0
This code initializes a two-stream-like distribution in phase space and advects it using a semi-Lagrangian method.
It also computes and displays the total number of particles to verify conservation.

author: Fabrizio Musacchio
date: Aug 2020
"""
# %% IMPORTS
import numpy as np
import matplotlib.pyplot as plt
# %% PARAMETERS, GRIDS, INITIAL CONDITION

# grids:
Lx = 2.0 * np.pi
Nx = 256
Nv = 256
vmax = 4.0

# create grids:
x = np.linspace(0.0, Lx, Nx, endpoint=False)
v = np.linspace(-vmax, vmax, Nv)
dx = Lx / Nx
dv = (2.0 * vmax) / (Nv - 1)

X, V = np.meshgrid(x, v, indexing="xy")


# initial condition: two-stream style (still smooth):
# density modulation times bimodal velocity structure
k = 1.0
sigv = 0.5
v0 = 1.2

# initial distribution function
def f0(x, v):
    spatial = 1.0 + 0.2 * np.cos(k * x)
    vel = 0.5 * np.exp(-((v - v0)**2) / (2.0 * sigv**2)) + 0.5 * np.exp(-((v + v0)**2) / (2.0 * sigv**2))
    return spatial * vel

F = f0(X, V)

# %% SEMI-LAGRANGIAN SCHEME
# semi-Lagrangian step in x with periodicity, linear interpolation
def semi_lagrange_step(Fn, dt):
    Fnp1 = np.zeros_like(Fn)
    for j in range(Nv):
        x_back = (x - v[j] * dt) % Lx
        xi = x_back / dx
        i0 = np.floor(xi).astype(int)
        frac = xi - i0
        i1 = (i0 + 1) % Nx
        Fnp1[j, :] = (1.0 - frac) * Fn[j, i0] + frac * Fn[j, i1]
    return Fnp1

# diagnostics: total number conservation:
def total_number(F):
    return np.sum(F) * dx * dv

N_init = total_number(F)

# %% MAIN TIME LOOP AND PLOTTING

dt = 0.08
nsteps = 40
snapshots = [0, 10, 20, 40]

fig, axes = plt.subplots(1, len(snapshots), figsize=(16, 3.8), constrained_layout=True)

step_to_ax = {s: ax for s, ax in zip(snapshots, axes)}

for n in range(1, nsteps + 1):
    F = semi_lagrange_step(F, dt)
    if n in step_to_ax:
        ax = step_to_ax[n]
        im = ax.imshow(F, origin="lower", aspect="auto", extent=[0.0, Lx, -vmax, vmax])
        ax.set_title(f"step = {n}, t = {n*dt:.2f}\nN/N0 = {total_number(F)/N_init:.6f}")
        ax.set_xlabel("x")
        if ax is axes[0]:
            ax.set_ylabel("v")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

# plot initial in first axis as step 0
axes[0].imshow(f0(X, V), origin="lower", aspect="auto", extent=[0.0, Lx, -vmax, vmax])
axes[0].set_title(f"step = 0, t = 0.00\nN/N0 = 1.000000")
axes[0].set_xlabel("x")
axes[0].set_ylabel("v")
#plt.tight_layout()
plt.savefig("vlasov_semi_lagrange.png", dpi=300)
#plt.show()
plt.close()
# %% END
""" 
A simple 1D1V Vlasov equation solver without forces, using the method of characteristics.
The Vlasov equation in 1D1V without forces is:
    ∂f/∂t + v ∂f/∂x = 0
The solution is given by:
    f(x,v,t) = f0(x - v t, v)
This code initializes a Gaussian packet in phase space and advects it exactly using the characteristics.
It also computes and displays the total number of particles to verify conservation.

author: Fabrizio Musacchio
date: Aug 2020
"""
# %% IMPORTS
import numpy as np
import matplotlib.pyplot as plt
# %% PARAMETERS, GRIDS, INITIAL CONDITION
# phase space grid
Lx = 2.0 * np.pi          # spatial domain length
Nx = 256
Nv = 256
vmax = 4.0

# create grids:
x = np.linspace(0.0, Lx, Nx, endpoint=False)
v = np.linspace(-vmax, vmax, Nv)
dx = Lx / Nx
dv = (2.0 * vmax) / (Nv - 1)

X, V = np.meshgrid(x, v, indexing="xy")   # shapes: (Nv, Nx)


# initial condition: Gaussian packet in x times Gaussian in v
x0 = 0.9 * np.pi
v0 = 1.0
sigx = 0.35
sigv = 0.7

# initial distribution function
def f0(x, v):
    # periodic distance in x for a localized packet on a periodic domain
    # map x-x0 to [-Lx/2, Lx/2)
    dxp = (x - x0 + 0.5 * Lx) % Lx - 0.5 * Lx
    return np.exp(-(dxp**2) / (2.0 * sigx**2)) * np.exp(-((v - v0)**2) / (2.0 * sigv**2))

# initialize distribution function
F_init = f0(X, V)

# %% EXACT VLASOV SOLUTION VIA CHARACTERISTICS
# f(x,v,t) = f0(x - v t, v)
# implement x-shift with periodicity and 1D linear interpolation in x
def advect_exact(F0, t):
    # For each v-row, shift x by -v t
    # We'll interpolate on periodic x-grid.
    F = np.zeros_like(F0)
    for j in range(Nv):
        shift = (v[j] * t) % Lx
        x_back = (x - shift) % Lx   # x - v t modulo Lx

        # convert x_back to fractional indices on grid
        xi = x_back / dx
        i0 = np.floor(xi).astype(int)
        frac = xi - i0
        i1 = (i0 + 1) % Nx

        F[j, :] = (1.0 - frac) * F0[j, i0] + frac * F0[j, i1]
    return F

# diagnostics: total number conservation:
def total_number(F):
    return np.sum(F) * dx * dv

# %% VISUALIZATION
# visualize at several times

times = [0.0, 0.8, 1.6, 2.4]
N0 = total_number(F_init)

fig, axes = plt.subplots(1, len(times), figsize=(16, 3.8), constrained_layout=True)

for ax, t in zip(axes, times):
    Ft = advect_exact(F_init, t)
    Nt = total_number(Ft)
    im = ax.imshow(
        Ft,
        origin="lower",
        aspect="auto",
        extent=[0.0, Lx, -vmax, vmax]
    )
    ax.set_title(f"t = {t:.2f}\nN(t)/N(0) = {Nt/N0:.6f}")
    ax.set_xlabel("x")
    if ax is axes[0]:
        ax.set_ylabel("v")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

#plt.tight_layout()
plt.savefig("vlasov_exact_solution.png", dpi=300)
#plt.show()
plt.close()
# %% END
"""
Particle-in-Cell (PIC) simulation of electrostatic plasma instabilitys in 1D1V:
- Two-stream instability setup with two counter-propagating electron beams.
- Periodic boundary conditions in space.
- Leapfrog (Boris) scheme for particle pushing.
- Charge deposition on grid with linear weighting.
- Poisson equation solved on grid for electric field.

This code visualizes the phase space evolution and energy conservation.

Original script by Philip Mocz (2020) Princeton University, @PMocz
https://github.com/pmocz/pic-python/tree/master

modified by: Fabrizio Musacchio (Oct 2020 / Jan 2026) for educational purposes.

"""
# %% IMPORTS
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
import imageio.v2 as imageio

# remove spines right and top for better aesthetics:
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.left'] = False
plt.rcParams['axes.spines.bottom'] = False
plt.rcParams.update({'font.size': 12})
# %% FUNCTIONS

def getAcc(pos, Nx, boxsize, n0, Gmtx, Lmtx, return_grid=False):
    """
    Calculate the acceleration on each particle due to electric field
        pos      is an Nx1 matrix of particle positions
        Nx       is the number of mesh cells
        boxsize  is the domain [0,boxsize]
        n0       is the electron number density
        Gmtx     is an Nx x Nx matrix for calculating the gradient on the grid
        Lmtx     is an Nx x Nx matrix for calculating the laplacian on the grid
        a        is an Nx1 matrix of accelerations
    """
    # calculate electron number density on the mesh by placing particles into 
    # the 2 nearest bins (j & j+1, with proper weights) and normalizing:
    N = pos.shape[0]
    dx = boxsize / Nx
    j = np.floor(pos / dx).astype(int)
    jp1 = j + 1
    weight_j = (jp1 * dx - pos) / dx
    weight_jp1 = (pos - j * dx) / dx
    jp1 = np.mod(jp1, Nx)  # periodic BC
    
    # we secure indices to be within [0,Nx):
    j = np.mod(j, Nx)
    
    n = np.bincount(j[:, 0], weights=weight_j[:, 0], minlength=Nx)
    n += np.bincount(jp1[:, 0], weights=weight_jp1[:, 0], minlength=Nx)
    n *= n0 * boxsize / N / dx

    # solve Poisson's equation: laplacian(phi) = n-n0
    phi_grid = spsolve(Lmtx, n - n0, permc_spec="MMD_AT_PLUS_A")

    # apply derivative to get the electric field:
    E_grid = -Gmtx @ phi_grid

    # interpolate grid value onto particle locations:
    E = weight_j * E_grid[j] + weight_jp1 * E_grid[jp1]

    a = -E
    
    if return_grid:
        return a, E_grid, n

    return a

def make_gif_from_folder(folder, gif_name, fps=10):
    files = sorted([os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".png")])
    frames = [imageio.imread(f) for f in files]
    imageio.mimsave(gif_name, frames, fps=fps)

# %% MAIN
"""Plasma PIC simulation"""

plot_folder = "pic_output"
plot_folder_energy = "pic_output_energy"
os.makedirs(plot_folder, exist_ok=True)
os.makedirs(plot_folder_energy, exist_ok=True)

# simulation parameters:
N = 40000   # number of particles
Nx = 400    # number of mesh cells
t = 0       # current time of the simulation
tEnd = 500  # time at which simulation ends
dt = 1      # timestep
boxsize = 50  # periodic domain [0,boxsize]
n0 = 1      # electron number density
vb = 3      # beam velocity
vth = 1     # beam width
A = 0.1     # perturbation
plotRealTime = True  # switch on for plotting as the simulation goes along

# generate initial conditions:
np.random.seed(42)  # set the random number generator seed
# construct 2 opposite-moving Gaussian beams
pos = np.random.rand(N, 1) * boxsize
vel = vth * np.random.randn(N, 1) + vb
Nh = int(N / 2)
vel[Nh:] *= -1
# add perturbation
vel *= 1 + A * np.sin(2 * np.pi * pos / boxsize)

# construct matrix G to computer Gradient  (1st derivative):
dx = boxsize / Nx
e = np.ones(Nx)
diags = np.array([-1, 1])
vals = np.vstack((-e, e))
Gmtx = sp.spdiags(vals, diags, Nx, Nx)
Gmtx = sp.lil_matrix(Gmtx)
Gmtx[0, Nx - 1] = -1
Gmtx[Nx - 1, 0] = 1
Gmtx /= 2 * dx
Gmtx = sp.csr_matrix(Gmtx)

# construct matrix L to computer Laplacian (2nd derivative):
diags = np.array([-1, 0, 1])
vals = np.vstack((e, -2 * e, e))
Lmtx = sp.spdiags(vals, diags, Nx, Nx)
Lmtx = sp.lil_matrix(Lmtx)
Lmtx[0, Nx - 1] = 1
Lmtx[Nx - 1, 0] = 1
Lmtx /= dx**2
Lmtx = sp.csr_matrix(Lmtx)

# prepare collectors for energy diagnostics:
t_hist = []
Wk_hist = []
We_hist = []

# calculate initial gravitational accelerations:
acc, E_grid, n = getAcc(pos, Nx, boxsize, n0, Gmtx, Lmtx, return_grid=True)

# energies:
dx = boxsize / Nx
Wk = 0.5 * np.sum(vel**2)
We = 0.5 * np.sum(E_grid**2) * dx
t_hist.append(t)
Wk_hist.append(Wk)
We_hist.append(We)



# main simulation Loop:
Nt = int(np.ceil(tEnd / dt))  # number of timesteps
for i in range(Nt):
    # (1/2) kick:
    vel += acc * dt / 2.0

    # drift (and apply periodic boundary conditions):
    pos += vel * dt
    pos = np.mod(pos, boxsize)

    # update accelerations:
    #acc = getAcc(pos, Nx, boxsize, n0, Gmtx, Lmtx)
    acc, E_grid, n_grid = getAcc(pos, Nx, boxsize, n0, Gmtx, Lmtx, return_grid=True)

    # (1/2) kick:
    vel += acc * dt / 2.0

    # update time:
    t += dt
    
    # update energies (need E_grid from the most recent field solve):
    Wk = 0.5 * np.sum(vel**2)
    We = 0.5 * np.sum(E_grid**2) * dx
    t_hist.append(t)
    Wk_hist.append(Wk)
    We_hist.append(We)

    # plot in real time - color 1/2 particles blue, other half red:
    if plotRealTime or (i == Nt - 1):
        #plt.cla()
        fig = plt.figure(figsize=(6, 5))
        plt.scatter(pos[0:Nh], vel[0:Nh], s=0.4, color="tab:cyan", alpha=0.5)
        plt.scatter(pos[Nh:], vel[Nh:], s=0.4, color="tab:red", alpha=0.5)
        plt.axis([0, boxsize, -6, 6])
        plt.xlabel("x")
        plt.ylabel("v")
        plt.title(f"Time = {t:.2f}")
        plt.tight_layout()
        plt.savefig(os.path.join(plot_folder, f"pic_{i:04d}.png"), dpi=150)
        plt.close(fig)
        
        """ # plot the energies on a separate plot:
        fig, ax1 = plt.subplots(figsize=(7, 4))
        l1, = ax1.plot(t_hist, Wk_hist, label="$W_K$ (kinetic)")
        l2, = ax1.plot(t_hist, np.array(Wk_hist) + np.array(We_hist), '--', label="$W_{tot}$ (total)")
        ax1.set_xlabel("t")
        ax1.set_ylabel("energy")
        # now jump to the second/right y-axis for field energy
        ax2 = ax1.twinx()
        l3, = ax2.plot(t_hist, We_hist, '--', label="$W_E$ (field)", color="magenta")
        ax2.set_ylabel("field energy")
        ax2.set_yscale("log")
        ax2.yaxis.label.set_color("magenta")
        ax2.tick_params(axis="y", colors="magenta")
        ax2.spines["right"].set_color("magenta")
        
        ax1.legend(handles=[l1, l2, l3], loc="lower right")
        plt.title("Energy $W_K$ (kinetic), $W_E$ (field), $W_{tot}$ (total)")
        plt.tight_layout()
        plt.savefig(os.path.join(plot_folder_energy, f"energy_{i:04d}.png"), dpi=150)
        plt.close(fig) """
        
        # plot the energies (two-panel layout)
        fig, (ax_top, ax_bot) = plt.subplots(
            2, 1, figsize=(7, 6), sharex=True,
            gridspec_kw={"height_ratios": [2, 1]}
        )

        # --- TOP PANEL: energies ---
        l1, = ax_top.plot(t_hist, Wk_hist, label=r"$W_K$ (kinetic)")
        l2, = ax_top.plot(
            t_hist,
            np.array(Wk_hist) + np.array(We_hist),
            "--",
            label=r"$W_{\mathrm{tot}}$ (total)")
        ax_top.set_ylabel("energy")

        # right axis for field energy:
        ax_top_r = ax_top.twinx()
        l3, = ax_top_r.plot(
            t_hist,
            We_hist,
            "--",
            color="magenta",
            label=r"$W_E$ (field)"
        )
        ax_top_r.set_ylabel("field energy")
        ax_top_r.set_yscale("log")
        ax_top_r.yaxis.label.set_color("magenta")
        ax_top_r.tick_params(axis="y", colors="magenta")
        ax_top_r.spines["right"].set_color("magenta")

        # combined legend_
        ax_top.legend(handles=[l1, l2, l3], loc="lower right")

        # --- BOTTOM PANEL: difference W_tot - W_K ---
        ax_bot.plot(
            t_hist,
            np.array(Wk_hist) + np.array(We_hist) - np.array(Wk_hist),
            color="black")
        ax_bot.set_xlabel("t")
        ax_bot.set_ylabel(r"$W_{\mathrm{tot}} - W_K$")
        ax_bot.set_yscale("log")

        plt.suptitle(r"Energy evolution: $W_K$ (kinetic), $W_E$ (field), $W_{\mathrm{tot}}$ (total)")
        plt.tight_layout()
        plt.savefig(os.path.join(plot_folder_energy, f"energy_{i:04d}.png"), dpi=150)
        plt.close(fig)


# plt.plot(t_hist, np.array(Wk_hist) + np.array(We_hist) - np.array(Wk_hist))


# Create GIFs from saved frames
make_gif_from_folder(
    plot_folder,
    gif_name="phase_space.gif",
    fps=10)

make_gif_from_folder(
    plot_folder_energy,
    gif_name="energy_evolution.gif",
    fps=10)

# %% END
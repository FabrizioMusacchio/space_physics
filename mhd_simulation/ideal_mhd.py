"""
2D ideal MHD toy code with hyperbolic divergence cleaning (Dedner style GLM).

This script solves the two-dimensional ideal magnetohydrodynamics (MHD) equations
in conservative form using a finite-volume discretization with a Rusanov
(local Lax–Friedrichs) numerical flux and second-order Runge–Kutta (RK2)
time stepping.

To control numerical violations of the solenoidal constraint div(B)=0, the
system is augmented by an auxiliary scalar field psi following the hyperbolic
divergence cleaning approach of Dedner et al.:

    * divergence errors are propagated away with a cleaning wave speed c_h
    * psi is damped on a timescale set by c_p via a source term

This does not enforce div(B)=0 exactly (unlike constrained transport), but keeps
divergence errors bounded and mobile for demonstration purposes.

State vector (conserved variables):

    U = [rho, mx, my, mz, E, Bx, By, Bz, psi]

with m = rho * v and total energy density

    E = p/(gamma-1) + 0.5*rho*|v|^2 + 0.5*|B|^2 + 0.5*psi^2.

Initial condition:
    Orszag–Tang vortex in a doubly periodic unit square domain.


This is just for demonstration: not high-end!

Author: Fabrizio Musacchio
Date: Jul 2020 / Jan 2026
"""
# %% IMPORTS
import numpy as np
import matplotlib.pyplot as plt
import os
import imageio.v2 as imageio
from PIL import Image
import glob
# %% FUNCTIONS
def prim_to_cons(rho, vx, vy, vz, p, Bx, By, Bz, psi, gamma):
    kinetic = 0.5 * rho * (vx*vx + vy*vy + vz*vz)
    mag = 0.5 * (Bx*Bx + By*By + Bz*Bz)
    E = p/(gamma - 1.0) + kinetic + mag + 0.5 * psi*psi
    mx = rho * vx
    my = rho * vy
    mz = rho * vz
    return np.stack([rho, mx, my, mz, E, Bx, By, Bz, psi], axis=0)

def cons_to_prim(U, gamma):
    rho = U[0]
    mx, my, mz = U[1], U[2], U[3]
    E = U[4]
    Bx, By, Bz = U[5], U[6], U[7]
    psi = U[8]

    vx = mx / rho
    vy = my / rho
    vz = mz / rho

    kinetic = 0.5 * rho * (vx*vx + vy*vy + vz*vz)
    mag = 0.5 * (Bx*Bx + By*By + Bz*Bz)
    p = (gamma - 1.0) * (E - kinetic - mag - 0.5*psi*psi)

    return rho, vx, vy, vz, p, Bx, By, Bz, psi

def flux_x(U, gamma):
    rho, vx, vy, vz, p, Bx, By, Bz, psi = cons_to_prim(U, gamma)
    mx, my, mz = U[1], U[2], U[3]
    E = U[4]

    B2 = Bx*Bx + By*By + Bz*Bz
    vB = vx*Bx + vy*By + vz*Bz
    pt = p + 0.5*B2

    # GLM coupling
    # In x-direction: Bx flux gets psi, psi flux gets c_h^2 * Bx
    # c_h is handled outside in Rusanov signal speed and psi flux below
    Fx = np.empty_like(U)
    Fx[0] = mx
    Fx[1] = mx*vx + pt - Bx*Bx
    Fx[2] = my*vx - Bx*By
    Fx[3] = mz*vx - Bx*Bz
    Fx[4] = (E + pt)*vx - Bx*vB
    Fx[5] = 0.0      # replaced later by +psi in Rusanov interface form
    Fx[6] = By*vx - Bx*vy
    Fx[7] = Bz*vx - Bx*vz
    Fx[8] = 0.0      # replaced later by c_h^2 * Bx
    return Fx

def flux_y(U, gamma):
    rho, vx, vy, vz, p, Bx, By, Bz, psi = cons_to_prim(U, gamma)
    mx, my, mz = U[1], U[2], U[3]
    E = U[4]

    B2 = Bx*Bx + By*By + Bz*Bz
    vB = vx*Bx + vy*By + vz*Bz
    pt = p + 0.5*B2

    Fy = np.empty_like(U)
    Fy[0] = my
    Fy[1] = mx*vy - By*Bx
    Fy[2] = my*vy + pt - By*By
    Fy[3] = mz*vy - By*Bz
    Fy[4] = (E + pt)*vy - By*vB
    Fy[5] = Bx*vy - By*vx
    Fy[6] = 0.0      # replaced later by +psi in Rusanov interface form
    Fy[7] = Bz*vy - By*vz
    Fy[8] = 0.0      # replaced later by c_h^2 * By
    return Fy

def fast_magnetosonic_speed(rho, p, Bx, By, Bz, gamma, nx, ny):
    # Directional fast speed approximation.
    # c_f^2 = 0.5 * (a^2 + vA^2 + sqrt((a^2 + vA^2)^2 - 4 a^2 vAn^2))
    a2 = gamma * p / rho
    B2 = Bx*Bx + By*By + Bz*Bz
    vA2 = B2 / rho
    Bn = Bx*nx + By*ny
    vAn2 = (Bn*Bn) / rho
    term = (a2 + vA2)**2 - 4.0*a2*vAn2
    term = np.maximum(term, 0.0)
    cf2 = 0.5 * (a2 + vA2 + np.sqrt(term))
    return np.sqrt(np.maximum(cf2, 0.0))

def rusanov_flux_x(UL, UR, gamma, c_h):
    FL = flux_x(UL, gamma)
    FR = flux_x(UR, gamma)

    # Insert GLM components in x-direction
    # Bx flux = psi, psi flux = c_h^2 * Bx
    FL = FL.copy()
    FR = FR.copy()
    FL[5] = UL[8]
    FR[5] = UR[8]
    FL[8] = (c_h*c_h) * UL[5]
    FR[8] = (c_h*c_h) * UR[5]

    # Signal speed
    rL, vLx, vLy, vLz, pL, BLx, BLy, BLz, psiL = cons_to_prim(UL, gamma)
    rR, vRx, vRy, vRz, pR, BRx, BRy, BRz, psiR = cons_to_prim(UR, gamma)

    cfL = fast_magnetosonic_speed(rL, pL, BLx, BLy, BLz, gamma, 1.0, 0.0)
    cfR = fast_magnetosonic_speed(rR, pR, BRx, BRy, BRz, gamma, 1.0, 0.0)
    smax = np.maximum(np.abs(vLx) + cfL, np.abs(vRx) + cfR)
    smax = np.maximum(smax, c_h)

    return 0.5*(FL + FR) - 0.5*smax*(UR - UL)

def rusanov_flux_y(UL, UR, gamma, c_h):
    FL = flux_y(UL, gamma)
    FR = flux_y(UR, gamma)

    # Insert GLM components in y-direction
    # By flux = psi, psi flux = c_h^2 * By
    FL = FL.copy()
    FR = FR.copy()
    FL[6] = UL[8]
    FR[6] = UR[8]
    FL[8] = (c_h*c_h) * UL[6]
    FR[8] = (c_h*c_h) * UR[6]

    rL, vLx, vLy, vLz, pL, BLx, BLy, BLz, psiL = cons_to_prim(UL, gamma)
    rR, vRx, vRy, vRz, pR, BRx, BRy, BRz, psiR = cons_to_prim(UR, gamma)

    cfL = fast_magnetosonic_speed(rL, pL, BLx, BLy, BLz, gamma, 0.0, 1.0)
    cfR = fast_magnetosonic_speed(rR, pR, BRx, BRy, BRz, gamma, 0.0, 1.0)
    smax = np.maximum(np.abs(vLy) + cfL, np.abs(vRy) + cfR)
    smax = np.maximum(smax, c_h)

    return 0.5*(FL + FR) - 0.5*smax*(UR - UL)

def periodic_pad(U):
    # U: (nvars, ny, nx)
    return np.pad(U, ((0,0),(1,1),(1,1)), mode="wrap")

def compute_rhs(U, dx, dy, gamma, c_h, c_p):
    # GLM damping: psi_t += - (c_h^2 / c_p^2) psi
    # We'll apply as a source term.
    Up = periodic_pad(U)
    nvars, ny, nx = U.shape

    # Fluxes at interfaces
    Fx = np.zeros((nvars, ny, nx+1), dtype=U.dtype)
    Fy = np.zeros((nvars, ny+1, nx), dtype=U.dtype)

    # x-interfaces
    for i in range(nx+1):
        UL = Up[:, 1:-1, i]     # left cell at interface
        UR = Up[:, 1:-1, i+1]   # right cell
        Fx[:, :, i] = rusanov_flux_x(UL, UR, gamma, c_h)

    # y-interfaces
    for j in range(ny+1):
        UL = Up[:, j, 1:-1]
        UR = Up[:, j+1, 1:-1]
        Fy[:, j, :] = rusanov_flux_y(UL, UR, gamma, c_h)

    rhs = np.zeros_like(U)
    rhs -= (Fx[:, :, 1:] - Fx[:, :, :-1]) / dx
    rhs -= (Fy[:, 1:, :] - Fy[:, :-1, :]) / dy

    # psi damping source
    rhs[8] += -(c_h*c_h / (c_p*c_p)) * U[8]
    return rhs

def step_rk2(U, dt, dx, dy, gamma, c_h, c_p):
    k1 = compute_rhs(U, dx, dy, gamma, c_h, c_p)
    U1 = U + dt*k1
    k2 = compute_rhs(U1, dx, dy, gamma, c_h, c_p)
    return 0.5*(U + U1 + dt*k2)

def orszag_tang_ic(nx, ny, gamma):
    # Domain [0, 1] x [0, 1]
    x = (np.arange(nx) + 0.5) / nx
    y = (np.arange(ny) + 0.5) / ny
    X, Y = np.meshgrid(x, y, indexing="xy")

    rho = np.ones((ny, nx))
    p = (gamma - 1.0) * np.ones((ny, nx))  # common choice

    vx = -np.sin(2.0*np.pi*Y)
    vy =  np.sin(2.0*np.pi*X)
    vz = np.zeros_like(vx)

    Bx = -np.sin(2.0*np.pi*Y)
    By =  np.sin(4.0*np.pi*X)
    Bz = np.zeros_like(Bx)

    psi = np.zeros_like(rho)
    U = prim_to_cons(rho, vx, vy, vz, p, Bx, By, Bz, psi, gamma)
    return U

def compute_dt(U, dx, dy, gamma, cfl, c_h):
    rho, vx, vy, vz, p, Bx, By, Bz, psi = cons_to_prim(U, gamma)
    cf_x = fast_magnetosonic_speed(rho, p, Bx, By, Bz, gamma, 1.0, 0.0)
    cf_y = fast_magnetosonic_speed(rho, p, Bx, By, Bz, gamma, 0.0, 1.0)
    smax_x = np.max(np.abs(vx) + cf_x)
    smax_y = np.max(np.abs(vy) + cf_y)
    smax_x = max(smax_x, c_h)
    smax_y = max(smax_y, c_h)
    dt = cfl / (smax_x/dx + smax_y/dy + 1e-14)
    return dt

# %% MAIN RUN
gamma = 5.0/3.0

# set grid size and resolution:
nx, ny = 256, 256
dx, dy = 1.0/nx, 1.0/ny

# GLM parameters:
c_h = 1.0
c_p = 0.2

cfl = 0.4
t_end = 0.5 # increase for more dynamic evolution (but longer runtime)

# output settings:
out_dir = "frames"
os.makedirs(out_dir, exist_ok=True)

plot_every = 20
dpi = 200

gif_path = os.path.join(out_dir, "mhd_evolution.gif")
gif_fps = 10   # frames per second in GIF

frame_files = []   # collect frame paths here

U = orszag_tang_ic(nx, ny, gamma)

t = 0.0
it = 0

# prepare GIF writer:
gif_path = os.path.join(out_dir, "mhd_evolution.gif")
gif_fps = 10
gif_writer = imageio.get_writer(gif_path, mode="I", fps=gif_fps)

# main time-stepping loop
while t < t_end:
    dt = compute_dt(U, dx, dy, gamma, cfl, c_h)
    if t + dt > t_end:
        dt = t_end - t

    U = step_rk2(U, dt, dx, dy, gamma, c_h, c_p)
    t += dt
    it += 1

    if it % plot_every == 0 or t >= t_end:
        rho, vx, vy, vz, p, Bx, By, Bz, psi = cons_to_prim(U, gamma)

        divB_rms = np.sqrt(np.mean(
            ((np.roll(Bx, -1, axis=1) - np.roll(Bx, 1, axis=1))/(2*dx) +
             (np.roll(By, -1, axis=0) - np.roll(By, 1, axis=0))/(2*dy))**2))

        print(f"t={t:.4f}, dt={dt:.3e}, iter={it}, divB_rms~{divB_rms:.3e}")

        dBy_dx = (np.roll(By, -1, axis=1) - np.roll(By, 1, axis=1)) / (2*dx)
        dBx_dy = (np.roll(Bx, -1, axis=0) - np.roll(Bx, 1, axis=0)) / (2*dy)
        jz = dBy_dx - dBx_dy

        fig, ax = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)

        im0 = ax[0].imshow(rho, origin="lower", extent=(0, 1, 0, 1), vmin=0, vmax=3)
        ax[0].set_title(f"density ρ at t={t:.3f} (iter {it})")
        ax[0].set_xlabel("x")
        ax[0].set_ylabel("y")
        plt.colorbar(im0, ax=ax[0], fraction=0.046, pad=0.04)

        im1 = ax[1].imshow(jz, origin="lower", extent=(0, 1, 0, 1), vmin=-60, vmax=70)
        ax[1].set_title("current proxy jz = ∂x By − ∂y Bx")
        ax[1].set_xlabel("x")
        ax[1].set_ylabel("y")
        plt.colorbar(im1, ax=ax[1], fraction=0.046, pad=0.04)

        frame_path = os.path.join(out_dir, f"frame_{len(frame_files):06d}.png")
        plt.savefig(frame_path, dpi=dpi)
        
        # append to GIF:
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
        rgb = buf[:, :, :3]  # Convert RGBA to RGB
        gif_writer.append_data(rgb)
        
        plt.close(fig)

        frame_files.append(frame_path)

# save final frame explicitly:
final_path = os.path.join(out_dir, "final.png")
rho, vx, vy, vz, p, Bx, By, Bz, psi = cons_to_prim(U, gamma)
dBy_dx = (np.roll(By, -1, axis=1) - np.roll(By, 1, axis=1)) / (2*dx)
dBx_dy = (np.roll(Bx, -1, axis=0) - np.roll(Bx, 1, axis=0)) / (2*dy)
jz = dBy_dx - dBx_dy

fig, ax = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
im0 = ax[0].imshow(rho, origin="lower", extent=(0, 1, 0, 1), vmin=0, vmax=3)
ax[0].set_title(f"density ρ at t={t:.3f} (final)")
ax[0].set_xlabel("x")
ax[0].set_ylabel("y")
plt.colorbar(im0, ax=ax[0], fraction=0.046, pad=0.04)

im1 = ax[1].imshow(jz, origin="lower", extent=(0, 1, 0, 1), vmin=-60, vmax=70)
ax[1].set_title("current proxy jz = ∂x By − ∂y Bx")
ax[1].set_xlabel("x")
ax[1].set_ylabel("y")
plt.colorbar(im1, ax=ax[1], fraction=0.046, pad=0.04)

plt.savefig(final_path, dpi=dpi)
plt.close(fig)

# finalize the GIF:
print("Creating GIF...")
gif_writer.close()
print(f"GIF written to {gif_path}")
# %% END
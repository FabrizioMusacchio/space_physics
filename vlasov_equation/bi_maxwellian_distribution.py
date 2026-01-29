"""
Bi-Maxwellian distributions and anisotropic pressure
----------------------------------------------------
Goal
* Visualize a bi-Maxwellian distribution f(v_perp, v_par) for different anisotropies T_perp != T_par
* Compute the parallel and perpendicular pressures from velocity-space moments:
    P_par  = m ∫ (v_par - u_par)^2 f d^3v
    P_perp = (m/2) ∫ v_perp^2 f d^3v
* Show that P_perp / P_par matches T_perp / T_par for a bi-Maxwellian (gyrotropic, zero drift)

Conventions
* We assume a gyrotropic distribution: f depends on v_perp = sqrt(vx^2 + vy^2) and v_par = vz only.
* Velocity-space volume element: d^3v = 2π v_perp dv_perp dv_par
* We set u_par = 0 by default.
* Units are arbitrary unless you plug in m, k_B explicitly. We keep k_B = 1, m = 1 by default.

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
# %% FUNCTIONS

# Bi-Maxwellian distribution:
def bi_maxwellian(v_perp, v_par, n=1.0, T_perp=1.0, T_par=1.0, m=1.0, kB=1.0, u_par=0.0):
    """
    Gyrotropic bi-Maxwellian f(v_perp, v_par) normalized to density n in 3D:

        f = n * (m/(2π kB T_perp)) * sqrt(m/(2π kB T_par))
            * exp( - m v_perp^2 / (2 kB T_perp) - m (v_par-u_par)^2 / (2 kB T_par) )

    This normalization satisfies:
        ∫ f d^3v = ∫∫ f * 2π v_perp dv_perp dv_par = n
    """
    pref = n * (m / (2.0 * np.pi * kB * T_perp)) * np.sqrt(m / (2.0 * np.pi * kB * T_par))
    expo = np.exp(
        -(m * v_perp**2) / (2.0 * kB * T_perp)
        -(m * (v_par - u_par) ** 2) / (2.0 * kB * T_par)
    )
    return pref * expo

# function to compute moments and pressures:
def compute_moments_pressure(vperp_grid, vpar_grid, f, m=1.0, u_par=0.0):
    """
    Compute density and pressures for gyrotropic f(v_perp, v_par).

    Integrals:
        n = ∫ f d^3v = ∫∫ f * 2π v_perp dv_perp dv_par

        P_par = m ∫ (v_par - u_par)^2 f d^3v

        P_perp = (m/2) ∫ v_perp^2 f d^3v
    """
    dvperp = vperp_grid[1] - vperp_grid[0]
    dvpar = vpar_grid[1] - vpar_grid[0]

    VPERP, VPAR = np.meshgrid(vperp_grid, vpar_grid, indexing="xy")
    weight = 2.0 * np.pi * VPERP * dvperp * dvpar  # d^3v weights

    n = np.sum(f * weight)
    P_par = m * np.sum(((VPAR - u_par) ** 2) * f * weight)
    P_perp = 0.5 * m * np.sum((VPERP**2) * f * weight)

    return n, P_perp, P_par


# numerical grid in velocity space
def make_velocity_grids(T_perp, T_par, m=1.0, kB=1.0, n_sigma=6.0, Nvperp=400, Nvpar=600):
    """
    Choose velocity ranges that cover the distribution well.
    For a Maxwellian, typical thermal speed scale is sqrt(kB T / m).
    Use n_sigma times thermal speed.
    """
    vth_perp = np.sqrt(kB * T_perp / m)
    vth_par = np.sqrt(kB * T_par / m)

    vperp_max = n_sigma * vth_perp
    vpar_max = n_sigma * vth_par

    vperp = np.linspace(0.0, vperp_max, Nvperp)
    vpar = np.linspace(-vpar_max, vpar_max, Nvpar)
    return vperp, vpar
# %% MAIN RUN

# some demo parameters
m = 1.0         # particle mass
kB = 1.0        # Boltzmann constant (here set to 1 for convenience)
n0 = 1.0        # number density
u_par = 0.0     # parallel bulk velocity

# base parallel temperature and a sweep over anisotropy ratios:
T_par0 = 1.0
anisotropy_list = [0.25, 0.5, 1.0, 2.0, 4.0]  # T_perp / T_par

# for the pressure-ratio curve:
anisotropy_curve = np.logspace(-1, 1, 17)  # 0.1 ... 10



# plot 1: contour plots for selected anisotropies:
# create a row of panels for f(v_par, v_perp) contours.
fig, axes = plt.subplots(1, len(anisotropy_list), 
                         figsize=(4.2 * len(anisotropy_list), 4), 
                         sharey=True)

for ax, a in zip(axes, anisotropy_list):
    T_perp0 = a * T_par0
    vperp, vpar = make_velocity_grids(T_perp0, T_par0, m=m, kB=kB, n_sigma=6.0, Nvperp=280, Nvpar=420)

    VPERP, VPAR = np.meshgrid(vperp, vpar, indexing="xy")
    f = bi_maxwellian(VPERP, VPAR, n=n0, T_perp=T_perp0, T_par=T_par0, m=m, kB=kB, u_par=u_par)

    # use log contours for dynamic range (avoid log(0) by adding tiny floor):
    f_floor = np.max(f) * 1e-12
    f_log = np.log10(f + f_floor)

    cs = ax.contour(VPAR, VPERP, f_log, levels=12)
    ax.set_title(rf"$T_\perp/T_\parallel = {a:g}$")
    ax.set_xlabel(r"$v_\parallel$")
    if ax is axes[0]:
        ax.set_ylabel(r"$v_\perp$")

plt.tight_layout()
plt.savefig("bi_maxwellian_anisotropy_contours.png", dpi=300)
plt.close()



# plot 2: pressure ratio vs temperature ratio:
ratios_T = []
ratios_P = []
densities = []

for a in anisotropy_curve:
    T_perp0 = a * T_par0
    vperp, vpar = make_velocity_grids(T_perp0, T_par0, m=m, kB=kB, n_sigma=7.0, Nvperp=400, Nvpar=600)

    VPERP, VPAR = np.meshgrid(vperp, vpar, indexing="xy")
    f = bi_maxwellian(VPERP, VPAR, n=n0, T_perp=T_perp0, T_par=T_par0, m=m, kB=kB, u_par=u_par)

    n_calc, P_perp, P_par = compute_moments_pressure(vperp, vpar, f, m=m, u_par=u_par)

    ratios_T.append(a)
    ratios_P.append(P_perp / P_par)
    densities.append(n_calc)

ratios_T = np.array(ratios_T)
ratios_P = np.array(ratios_P)
densities = np.array(densities)

plt.figure(figsize=(6.2, 4.8))
plt.plot(ratios_T, ratios_P, marker="o", label=r"numerical: $P_\perp/P_\parallel$")
plt.plot(ratios_T, ratios_T, linestyle="--", label=r"reference: $P_\perp/P_\parallel = T_\perp/T_\parallel$")
plt.xscale("log")
plt.yscale("log")
plt.xlabel(r"$T_\perp/T_\parallel$")
plt.ylabel(r"$P_\perp/P_\parallel$")
plt.title("Bi-Maxwellian anisotropy: Pressure ratio\ntracks temperature ratio")
plt.grid(True, which="both", alpha=0.3)
plt.legend(frameon=False)
plt.tight_layout()
plt.savefig("bi_maxwellian_pressure_vs_temperature_ratio.png", dpi=300)
plt.close()


# sanity printout:
print("\nDensity conservation across anisotropy sweep:")
print(f"min(n) = {densities.min()}, max(n) = {densities.max()}, target n = {n0}")
# %% END
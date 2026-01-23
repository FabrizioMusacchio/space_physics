"""
krook_relaxation_velocity_space.py

Krook (BGK-type) collision operator as velocity-space relaxation (0D in space).

We solve:
    ∂_t f(v,t) = ν ( f_m(v) - f(v,t) )

Exact solution:
    f(v,t) = f_m(v) + (f_0(v) - f_m(v)) * exp(-ν t)

This script:
* constructs a non-Maxwellian initial distribution f0(v)
* defines a Maxwellian target distribution fm(v)
* evolves f(v,t) either via the exact solution or via explicit time stepping
* compares different collision frequencies ν
* visualizes relaxation in velocity space and tracks moment relaxation

author: Fabrizio Musacchio
date: Aug 2020
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

# distribution definitions:

def maxwell_1d(v, n=1.0, u=0.0, vt=1.0):
    """
    1D Maxwellian with thermal speed vt (standard deviation):
        f(v) = n / (sqrt(2π) vt) * exp(-(v-u)^2 / (2 vt^2))
    """
    return n / (np.sqrt(2.0 * np.pi) * vt) * np.exp(-0.5 * ((v - u) / vt) ** 2)


def core_plus_beam(v, eps=0.05, n=1.0, u_core=0.0, vt_core=1.0, u_beam=4.0, vt_beam=0.3):
    """
    Non-Maxwellian mixture distribution:
        f0(v) = (1-eps) f_core(v) + eps f_beam(v)
    """
    f_core = maxwell_1d(v, n=n, u=u_core, vt=vt_core)
    f_beam = maxwell_1d(v, n=n, u=u_beam, vt=vt_beam)
    return (1.0 - eps) * f_core + eps * f_beam


def normalize_pdf(v, f):
    """
    Normalize f(v) such that ∫ f dv = 1 (or desired n if you rescale later).
    """
    Z = np.trapezoid(f, v)
    if Z <= 0 or not np.isfinite(Z):
        raise ValueError("Normalization failed.")
    return f / Z


def velocity_moments(v, f):
    """
    Compute n, u, variance (temperature proxy) for 1D distributions.
        n = ∫ f dv
        u = (1/n) ∫ v f dv
        var = (1/n) ∫ (v-u)^2 f dv
    """
    n = np.trapezoid(f, v)
    u = np.trapezoid(v * f, v) / n
    var = np.trapezoid(((v - u) ** 2) * f, v) / n
    return n, u, var



# Krook / BGK relaxation:

def krook_exact(f0, fm, nu, t):
    """
    Exact solution at time t:
        f(t) = fm + (f0 - fm) exp(-nu t)
    """
    return fm + (f0 - fm) * np.exp(-nu * t)


def krook_euler(f0, fm, nu, t_grid):
    """
    Explicit Euler time stepping as an optional numerical check:
        f^{n+1} = f^n + dt * nu (fm - f^n)
    """
    f = f0.copy()
    out = [f.copy()]
    for i in range(1, len(t_grid)):
        dt = t_grid[i] - t_grid[i - 1]
        f = f + dt * nu * (fm - f)
        out.append(f.copy())
    return np.array(out)
# %% MAIN RUN

# Velocity grid
vmin, vmax = -8.0, 8.0
Nv = 4000
v = np.linspace(vmin, vmax, Nv)

# Target Maxwellian fm(v)
# Choose an equilibrium state you want the collisions to enforce.
# Here: zero drift, vt=1.
fm = maxwell_1d(v, n=1.0, u=0.0, vt=1.0)
fm = normalize_pdf(v, fm)

# Initial non-Maxwellian f0(v)
# Here: weak beam added to a core, then normalized.
f0 = core_plus_beam(v, eps=0.05, n=1.0, u_core=0.0, vt_core=1.0, u_beam=4.0, vt_beam=0.25)
f0 = normalize_pdf(v, f0)

# Collision frequencies to compare
nu_list = [0.2, 1.0, 5.0]

# Time axis for plotting
tmax = 6.0
Nt = 7
t_samples = np.linspace(0.0, tmax, Nt)

# for moment relaxation curves:
t_curve = np.linspace(0.0, tmax, 300)

# Plot 1: snapshots in time for each nu:
for nu in nu_list:
    plt.figure(figsize=(7.2, 4.6))
    for t in t_samples:
        ft = krook_exact(f0, fm, nu=nu, t=t)
        plt.plot(v, ft, label=f"t = {t:.2f}")
    plt.plot(v, fm, linewidth=2.0, label="f_m (target)")
    plt.xlabel("v")
    plt.ylabel("f(v,t)")
    plt.title(f"Krook relaxation in velocity space (nu = {nu:g})")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(f"krook_relaxation_snapshots_nu_{nu:g}.png", dpi=300)
    plt.close()


# plot 2: relaxation of moments vs time (for each nu):
# Compute moments for f_m once
n_m, u_m, var_m = velocity_moments(v, fm)

plt.figure(figsize=(7.2, 7.2))
ax1 = plt.subplot(311)
ax2 = plt.subplot(312)
ax3 = plt.subplot(313)

for nu in nu_list:
    n_list = []
    u_list = []
    var_list = []
    for t in t_curve:
        ft = krook_exact(f0, fm, nu=nu, t=t)
        n_t, u_t, var_t = velocity_moments(v, ft)
        n_list.append(n_t)
        u_list.append(u_t)
        var_list.append(var_t)

    ax1.plot(t_curve, n_list, label=f"nu = {nu:g}")
    ax2.plot(t_curve, u_list, label=f"nu = {nu:g}")
    ax3.plot(t_curve, var_list, label=f"nu = {nu:g}")

ax1.axhline(n_m, linestyle="--", label="target")
ax2.axhline(u_m, linestyle="--")
ax3.axhline(var_m, linestyle="--")

ax1.set_ylabel("n")
ax2.set_ylabel("u")
ax3.set_ylabel("var ~ T")
ax3.set_xlabel("t")
ax1.set_title("Relaxation of velocity moments under Krook operator")
ax1.legend(frameon=False)
ax1.grid(True, alpha=0.3)
ax2.grid(True, alpha=0.3)
ax3.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("krook_relaxation_moments.png", dpi=300)
plt.close()


# plot 3: L2 distance to equilibrium vs time:
# This provides a scalar measure of how far f is from f_m.
plt.figure(figsize=(7.2, 4.6))
for nu in nu_list:
    dist = []
    for t in t_curve:
        ft = krook_exact(f0, fm, nu=nu, t=t)
        dist.append(np.sqrt(np.trapezoid((ft - fm) ** 2, v)))
    plt.semilogy(t_curve, dist, label=f"nu = {nu:g}")
plt.xlabel("t")
plt.ylabel(r"$||f - f_m||_2$")
plt.title("Krook relaxation: distance to equilibrium")
plt.legend(frameon=False)
plt.grid(True, which="both", alpha=0.3)
plt.tight_layout()
plt.savefig("krook_relaxation_distance.png", dpi=300)
plt.close()

# optional: compare exact vs Euler for one nu as a numerical check
nu_test = nu_list[1]
t_grid = np.linspace(0.0, tmax, 200)
f_euler = krook_euler(f0, fm, nu=nu_test, t_grid=t_grid)
f_exact_end = krook_exact(f0, fm, nu=nu_test, t=t_grid[-1])

err = np.sqrt(np.trapezoid((f_euler[-1] - f_exact_end) ** 2, v))
print("Saved figures:")
for nu in nu_list:
    print(f"  krook_relaxation_snapshots_nu_{nu:g}.png")
print("  krook_relaxation_moments.png")
print("  krook_relaxation_distance.png")
print(f"\nEuler sanity check (nu={nu_test:g}) L2 error at t=tmax: {err:.3e}")

# %% END
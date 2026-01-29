"""
This script visualizes charged particle motion along a dipole field 
line + adiabatic mirror model in the meridional plane (x,z)

We use: 

Normalized units:
  Earth radius R_E = 1
  Equatorial surface field B_E = 1

Dipole magnitude along an L-shell:
  B(λ,L) = (1/L^3) * sqrt(1 + 3 sin^2 λ) / cos^6 λ

1st adiabatic invariant:
  μ = m v_perp^2 / (2 B) = const
so:
  v_perp(λ) = sqrt(2 μ B(λ) / m)

Energy conservation:
  v^2 = v_perp^2 + v_par^2

Mirror point where v_par = 0  <=>  v_perp = v

As integrator, we use explicit Euler for the guiding center motion along the
field line and for the gyro phase evolution. At the mirror points, we reflect
the parallel velocity.

author: Fabrizio Musacchio
date: Jul 2020 / Jan 2026
"""
# %% IMPORTS
import numpy as np
import matplotlib.pyplot as plt
# remove spines right and top for better aesthetics
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.left'] = False
plt.rcParams['axes.spines.bottom'] = False
plt.rcParams.update({'font.size': 12})
# %% FUNCTIONS TO CALCULATE FIELD, POSITIONS, ETC.
def r_of_lambda(L, lam):
    return L * (np.cos(lam)**2)

def dipole_Bmag(L, lam):
    c = np.cos(lam)
    s = np.sin(lam)
    return (1.0 / L**3) * np.sqrt(1.0 + 3.0*s*s) / (c**6)

def xyz_on_fieldline(L, lam):
    # Meridional plane y=0, magnetic latitude λ
    # colatitude θ = π/2 - λ
    # x = r sinθ = r cosλ
    # z = r cosθ = r sinλ
    r = r_of_lambda(L, lam)
    x = r * np.cos(lam)
    z = r * np.sin(lam)
    return x, z

def ds_dlambda(L, lam):
    # Arc length along field line in meridional plane:
    # ds^2 = dr^2 + (r dθ)^2, with θ = π/2 - λ => dθ = -dλ
    r = r_of_lambda(L, lam)
    dr = -2.0 * L * np.cos(lam) * np.sin(lam)
    return np.sqrt(dr*dr + r*r)

def unit_tangent_and_normal(L, lam):
    # Numerical tangent on the field line and in-plane normal
    eps = 1e-6
    x1, z1 = xyz_on_fieldline(L, lam - eps)
    x2, z2 = xyz_on_fieldline(L, lam + eps)
    tx, tz = (x2 - x1), (z2 - z1)
    tnorm = np.sqrt(tx*tx + tz*tz) + 1e-15
    tx, tz = tx/tnorm, tz/tnorm
    nx, nz = -tz, tx
    return (tx, tz), (nx, nz)

def find_mirror_latitude(L, alpha_eq, lam_max=1.2):
    # Condition from μ const:
    # B(λ_m) = B_eq / sin^2(alpha_eq)
    Beq = dipole_Bmag(L, 0.0)
    target = Beq / (np.sin(alpha_eq)**2)

    lams = np.linspace(0.0, lam_max, 20001)
    Bvals = dipole_Bmag(L, lams)
    idx = np.argmax(Bvals >= target)

    if Bvals[idx] < target:
        raise RuntimeError("No mirror point found: pitch angle too small or lam_max too low.")

    lam1, lam2 = lams[idx-1], lams[idx]
    B1, B2 = Bvals[idx-1], Bvals[idx]
    return lam1 + (target - B1) * (lam2 - lam1) / (B2 - B1 + 1e-15)

# %% MAIN SCRIPT

# parameters we can set:
L = 4.0                          # L-shell
alpha_eq = np.deg2rad(15.0)      # equatorial pitch angle; larger => mirror point closer to equator
v_total = 1.0                    # total speed (normalized)
m = 3.0
q = 300.0                       # large q/m => small gyroradius in normalized units

# derived constants (do not change!; calculates initial mu and mirror latitude):
Beq = dipole_Bmag(L, 0.0)
vperp_eq = v_total * np.sin(alpha_eq)
mu = 0.5 * m * vperp_eq**2 / Beq
lam_m = find_mirror_latitude(L, alpha_eq, lam_max=np.deg2rad(70))

# initial conditions:
lam = np.deg2rad(2.0)            # start near equator
sgn = +1.0                       # initial direction along field
phi = 0.0                        # gyro phase

# time integration:
dt = 0.002
steps = 9150 # 5000
""" 
Increase the number of steps to get a longer trajectory. However, note that
the explicit Euler integrator is not very accurate, so for very long
simulations the results may become inaccurate due to numerical errors/drift.
"""

# time-stepping loop (numerical integration):
xs, zs, dirs = [], [], []
for _ in range(steps):
    # Euler explicit integrator:
    B = dipole_Bmag(L, lam)
    vperp = np.sqrt(2.0 * mu * B / m)
    vpar = sgn * np.sqrt(max(v_total**2 - vperp**2, 0.0))

    # Guiding center evolution along the field line
    dlam_dt = vpar / (ds_dlambda(L, lam) + 1e-15)
    lam_new = lam + dt * dlam_dt

    # Gyro phase evolution
    dphi_dt = (q * B / m)
    phi_new = phi + dt * dphi_dt

    # mirror reflection at |λ| = λ_m:
    if abs(lam_new) >= lam_m:
        lam_new = np.sign(lam_new) * (2*lam_m - abs(lam_new))
        sgn *= -1.0

    # guiding center position:
    xg, zg = xyz_on_fieldline(L, lam_new)

    # Local normal direction for an in-plane "gyration wiggle"
    (_, _), (nx, nz) = unit_tangent_and_normal(L, lam_new)

    # gyroradius ρ = m v_perp / (q B):
    Bn = dipole_Bmag(L, lam_new)
    vperp_n = np.sqrt(2.0 * mu * Bn / m)
    rho = m * vperp_n / (q * Bn + 1e-15)

    # in-plane wiggle (visualization):
    x = xg + rho * np.cos(phi_new) * nx
    z = zg + rho * np.cos(phi_new) * nz

    xs.append(x)
    zs.append(z)
    dirs.append(sgn)

    lam, phi = lam_new, phi_new

xs = np.array(xs)
zs = np.array(zs)
dirs = np.array(dirs)

# calculate field line for reference for plotting:
lam_foot = np.arccos(1.0 / np.sqrt(L))          # footpoint latitude (r=1)
lam_grid = np.linspace(-lam_foot, lam_foot, 2000)
xf, zf = xyz_on_fieldline(L, lam_grid)

## %% PLOTS
# Plot
fig = plt.figure(figsize=(7.85, 4.5))
plt.plot(xf, zf, linewidth=2, label=f"field line (L={L:g})")
# plt.plot(xs[dirs > 0], zs[dirs > 0], linewidth=1.2, label="forward")
# plt.plot(xs[dirs < 0], zs[dirs < 0], linewidth=1.2, linestyle="--", label="backward")
# get the indices, where the direction changes:
cuts = np.where(np.diff(dirs) != 0)[0] + 1
# define segment boundaries:
starts = np.r_[0, cuts]
ends   = np.r_[cuts, len(xs)]
# now plot each segment separately:
for s, e in zip(starts, ends):
    if dirs[s] > 0:
        plt.plot(xs[s:e], zs[s:e], linewidth=1.2, label=None,
                 c="tab:green")              # forward
    else:
        plt.plot(xs[s:e], zs[s:e], linewidth=1.2, linestyle="--", 
                 label=None, c="tab:orange")  # backward
# dummy plots for legend:
plt.plot([], [], c="tab:green", linewidth=1.2, label="forward motion")
plt.plot([], [], c="tab:orange", linewidth=1.2, linestyle="--", label="backward motion")

# earth:
theta = np.linspace(0, 2*np.pi, 600)
plt.plot(np.cos(theta), np.sin(theta), linewidth=2, label="Earth",
         c="tab:red")

# Mirror points
xM, zM = xyz_on_fieldline(L, lam_m)
plt.scatter([xM, xM], [zM, -zM], s=35, label="mirror points")

plt.gca().set_aspect("equal", adjustable="box")
plt.xlabel("x ($R_E$)")
plt.ylabel("z ($R_E$)")
plt.title("Magnetic mirror motion on a dipole field line (2D projection)\nAdiabatic model with μ const and energy const\n")
plt.legend(loc="upper right", bbox_to_anchor=(1.50, 1.0))
plt.xlim(-1.1, L*1.1)
plt.ylim(-2, 2)

# set x and y ticks:
plt.xticks(np.arange(-1, L+1, 1))
#plt.yticks(np.array([-1.5,-0.5, 0, 0.5, 1.5]))
plt.yticks(np.arange(-2, 2.5, 1))

plt.tight_layout()

out_path = f"magnetic_mirror_2d_L{L}_alpha{np.rad2deg(alpha_eq):.1f}deg.png"
plt.savefig(out_path, dpi=200)
#plt.close(fig)

rho_eq = m * vperp_eq / (q * Beq)
print(f"mirror latitude λ_m = {np.rad2deg(lam_m):.2f} deg")
print(f"equatorial gyroradius ρ_eq ≈ {rho_eq:.3f} R_E")
# %% END

"""
Toy demo for the blog post on "What velocity moments miss: Core plus beam distributions"
---------------------------------------------------
Goal
* Construct a 1D velocity distribution consisting of a thermal core plus a weak beam:
    f(v) = (1 - eps) f_core(v) + eps f_beam(v)
* Compute fluid moments (density n, bulk velocity u, temperature T) from velocity moments
* Vary eps and show that moments can remain very similar while f(v) changes visibly

This is a deliberately minimal, pedagogical script.

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

# Helpers: distributions

# 1D Maxwellian:
def maxwell_1d(v, n=1.0, u=0.0, vt=1.0):
    """
    1D Maxwellian:
        f(v) = n / (sqrt(2*pi)*vt) * exp( - (v-u)^2 / (2 vt^2) )
    Here vt is the 1D thermal speed (standard deviation).
    """
    return n / (np.sqrt(2.0 * np.pi) * vt) * np.exp(-0.5 * ((v - u) / vt) ** 2)

# Mixture: core plus beam:
def core_plus_beam(v, eps=0.02, n=1.0, u_core=0.0, vt_core=1.0, u_beam=4.0, vt_beam=0.3):
    """
    Mixture model:
        f = (1-eps) f_core + eps f_beam
    Both components are normalized so that integral f dv = n.
    """
    f_core = maxwell_1d(v, n=n, u=u_core, vt=vt_core)
    f_beam = maxwell_1d(v, n=n, u=u_beam, vt=vt_beam)
    return (1.0 - eps) * f_core + eps * f_beam


# moments from distribution:
def velocity_moments(v, f):
    """
    Compute density n, bulk velocity u, and 1D temperature-like variance T
    from velocity moments.

    n = ∫ f dv
    u = (1/n) ∫ v f dv
    T ~ (1/n) ∫ (v-u)^2 f dv

    Note:
    * In kinetic theory, pressure p = m ∫ (v-u)^2 f dv, and T = p/(n k_B).
      Here we return the variance sigma^2 = (1/n) ∫ (v-u)^2 f dv as a proxy for temperature.
    """
    dv = v[1] - v[0]
    n = np.sum(f) * dv
    u = np.sum(v * f) * dv / n
    var = np.sum(((v - u) ** 2) * f) * dv / n
    return n, u, var
# %% MAIN RUN

# demo configuration:

# velocity grid:
vmin, vmax = -10.0, 10.0
Nv = 40001  # large enough to resolve a narrow beam; reduce if needed for speed
v = np.linspace(vmin, vmax, Nv)

# base parameters:
params = dict(
    n=1.0,
    u_core=0.0,
    vt_core=1.0,
    u_beam=4.0,
    vt_beam=0.25)

# beam strengths to compare;
eps_list = [0.0, 1e-3, 5e-3, 1e-2, 2e-2, 5e-2]



# compute distributions and moments:
moments = []
fs = []
for eps in eps_list:
    f = core_plus_beam(v, eps=eps, **params)
    n, u, Tvar = velocity_moments(v, f)
    moments.append((eps, n, u, Tvar))
    fs.append(f)
moments = np.array(moments, dtype=float)  # columns: eps, n, u, Tvar

# %% PLOTS

# plot 1: f(v) for different eps
plt.figure(figsize=(7, 4))
for f, eps in zip(fs, eps_list):
    plt.plot(v, f, label=f"eps = {eps:g}")
plt.xlim(-6, 10)
plt.xlabel("v")
plt.ylabel("f(v)")
plt.title("Core plus weak beam: Distribution changes\nstrongly, moments can change weakly")
plt.legend(frameon=False)
plt.tight_layout()
plt.savefig("velocity_distribution_vs_eps.png", dpi=300)
plt.close()



# plot 2: zoom-in on the beam region
plt.figure(figsize=(7, 4))
for f, eps in zip(fs, eps_list):
    plt.plot(v, f, label=f"eps = {eps:g}")
plt.xlim(1.5, 7.0)
plt.ylim(0, None)
plt.xlabel("v")
plt.ylabel("f(v)")
plt.title("Zoom near the beam: Small eps introduces clear\nkinetic structure")
plt.legend(frameon=False)
plt.tight_layout()
plt.savefig("velocity_distribution_zoom_vs_eps.png", dpi=300)
plt.close()


# plot 3: moments vs eps
eps = moments[:, 0]
n = moments[:, 1]
u = moments[:, 2]
Tvar = moments[:, 3]

fig = plt.figure(figsize=(7, 7))

ax1 = fig.add_subplot(311)
ax1.plot(eps, n, marker="o")
ax1.set_xlabel("eps")
ax1.set_ylabel("n")
ax1.set_title("Velocity moments vs beam strength eps")
ax1.grid(True, alpha=0.3)

ax2 = fig.add_subplot(312)
ax2.plot(eps, u, marker="o")
ax2.set_xlabel("eps")
ax2.set_ylabel("u")
ax2.grid(True, alpha=0.3)

ax3 = fig.add_subplot(313)
ax3.plot(eps, Tvar, marker="o")
ax3.set_xlabel("eps")
ax3.set_ylabel("T (variance)")
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("velocity_moments_vs_eps.png", dpi=300)
plt.close()



# optional: We demonstrate "nearly identical moments"
# by choosing a beam that minimally perturbs u and T.
# One way is to add a symmetric pair of beams at ±u_beam:
def symmetric_beams(v, eps=0.02, n=1.0, u_core=0.0, vt_core=1.0, u_beam=4.0, vt_beam=0.25):
    """
    Symmetric two-beam mixture:
        f = (1-eps) f_core + (eps/2) f_beam(+u) + (eps/2) f_beam(-u)
    This keeps bulk velocity closer to zero while still adding strong kinetic structure.
    """
    f_core = maxwell_1d(v, n=n, u=u_core, vt=vt_core)
    f_bp = maxwell_1d(v, n=n, u=+u_beam, vt=vt_beam)
    f_bm = maxwell_1d(v, n=n, u=-u_beam, vt=vt_beam)
    return (1.0 - eps) * f_core + 0.5 * eps * f_bp + 0.5 * eps * f_bm


eps_list2 = [0.0, 1e-3, 5e-3, 1e-2, 2e-2, 5e-2]
moments2 = []
fs2 = []

for eps_ in eps_list2:
    f2 = symmetric_beams(v, eps=eps_, **params)
    n2, u2, T2 = velocity_moments(v, f2)
    moments2.append((eps_, n2, u2, T2))
    fs2.append(f2)

moments2 = np.array(moments2, dtype=float)

plt.figure(figsize=(7, 4))
for f2, eps_ in zip(fs2, eps_list2):
    plt.plot(v, f2, label=f"eps = {eps_:g}")
plt.xlim(-7, 7)
plt.xlabel("v")
plt.ylabel("f(v)")
plt.title("Symmetric weak beams: strong kinetic structure with u ~ 0")
plt.legend(frameon=False)
plt.tight_layout()
plt.savefig("symmetric_velocity_distribution_vs_eps.png", dpi=300)
plt.close()

fig = plt.figure(figsize=(7, 7))
ax1 = fig.add_subplot(311)
ax1.plot(moments2[:, 0], moments2[:, 1], marker="o")
ax1.set_xlabel("eps")
ax1.set_ylabel("n")
ax1.set_title("Moments vs eps (symmetric beams)")
ax1.grid(True, alpha=0.3)

ax2 = fig.add_subplot(312)
ax2.plot(moments2[:, 0], moments2[:, 2], marker="o")
ax2.set_xlabel("eps")
ax2.set_ylabel("u")
ax2.grid(True, alpha=0.3)

ax3 = fig.add_subplot(313)
ax3.plot(moments2[:, 0], moments2[:, 3], marker="o")
ax3.set_xlabel("eps")
ax3.set_ylabel("T (variance)")
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("symmetric_velocity_moments_vs_eps.png", dpi=300)
plt.close()



# print a concise numeric table to the console:
print("\nAsymmetric single-beam mixture:")
print("eps        n            u            T(var)")
for row in moments:
    print(f"{row[0]:.4g}   {row[1]:.8f}   {row[2]:.8f}   {row[3]:.8f}")
print("\nSymmetric two-beam mixture:")
print("eps        n            u            T(var)")
for row in moments2:
    print(f"{row[0]:.4g}   {row[1]:.8f}   {row[2]:.8f}   {row[3]:.8f}")
# %% END

""" 
Script to visualize schematic anisotropic velocity distributions in (v_perp, v_par):
 - bi-Maxwellian
 - drifting Maxwellian
 - loss-cone distribution
 
author: Fabrizio Musacchio
date: June 2020
"""
# %% IMPORTS
import numpy as np
import matplotlib.pyplot as plt
# %% ANISOTROPIC DISTRIBUTIONS

# helper: normalized 2D distributions in (v_perp, v_par)
# We plot schematic shapes, not absolute normalization.
def bi_maxwell(vperp, vpar, T_perp=2.0, T_par=1.0):
    # f ~ exp(- v_perp^2/(2 T_perp) - v_par^2/(2 T_par))
    return np.exp(-0.5 * (vperp**2 / T_perp + vpar**2 / T_par))

def drifting_maxwell(vperp, vpar, T=1.0, u_par=1.5):
    # isotropic Maxwell in (v_perp, v_par) with drift in v_par
    return np.exp(-0.5 * (vperp**2 + (vpar - u_par)**2) / T)

def loss_cone(vperp, vpar, T_perp=1.5, T_par=1.0, alpha_lc_deg=25.0, smooth=0.08):
    """
    Simple schematic loss-cone:
    start from a bi-Maxwell core and suppress phase-space density for pitch angles |alpha| < alpha_lc
    where alpha = arctan(v_perp/|v_par|).
    We implement a smooth cutoff with a logistic/tanh.
    """
    f = bi_maxwell(vperp, vpar, T_perp=T_perp, T_par=T_par)

    alpha_lc = np.deg2rad(alpha_lc_deg)
    # pitch angle alpha in [0, pi/2]
    alpha = np.arctan2(np.abs(vperp), np.abs(vpar) + 1e-12)

    # suppression factor ~ 0 inside loss cone, ~ 1 outside
    # tanh transition around alpha_lc
    s = 0.5 * (1.0 + np.tanh((alpha - alpha_lc) / smooth))

    return f * s


# grid in velocity space:
vmax = 4.0
Nv = 500
vperp = np.linspace(-vmax, vmax, Nv)   # symmetric for schematic style
vpar  = np.linspace(-vmax, vmax, Nv)
VPERP, VPAR = np.meshgrid(vperp, vpar, indexing="xy")

# compute distributions
F_aniso = bi_maxwell(VPERP, VPAR, T_perp=2.2, T_par=0.9)
F_drift = drifting_maxwell(VPERP, VPAR, T=1.0, u_par=1.6)
F_lc    = loss_cone(VPERP, VPAR, T_perp=1.8, T_par=1.0, alpha_lc_deg=27.0, smooth=0.06)

# levels for contour lines (choose relative levels)
def contour_levels(F, n=6):
    fmax = np.max(F)
    # geometric spacing gives nicer nested contours
    return fmax * np.geomspace(0.15, 0.95, n)

levels_aniso = contour_levels(F_aniso, n=6)
levels_drift = contour_levels(F_drift, n=6)
levels_lc    = contour_levels(F_lc, n=6)


# plotting:
fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.0), constrained_layout=True)

# common settings
for ax in axes:
    ax.set_aspect("equal")
    ax.set_xlim(-vmax, vmax)
    ax.set_ylim(-vmax, vmax)
    ax.set_xlabel(r"$v_\perp$")
    ax.set_ylabel(r"$v_\parallel$")
    ax.axhline(0.0, lw=1.0, color="black")
    ax.axvline(0.0, lw=1.0, color="black")
    ax.set_xticks([])
    ax.set_yticks([])

# panel 1: anisotropic
axes[0].contourf(VPERP, VPAR, F_aniso, levels=80)
axes[0].contour(VPERP, VPAR, F_aniso, levels=levels_aniso, colors="black", linewidths=1.2)
axes[0].set_title("anisotropic distribution")

# panel 2: drifting Maxwellian
axes[1].contourf(VPERP, VPAR, F_drift, levels=80)
axes[1].contour(VPERP, VPAR, F_drift, levels=levels_drift, colors="black", linewidths=1.2)
axes[1].set_title("drifting Maxwellian")

# panel 3: loss cone
axes[2].contourf(VPERP, VPAR, F_lc, levels=80)
axes[2].contour(VPERP, VPAR, F_lc, levels=levels_lc, colors="black", linewidths=1.2)
axes[2].set_title("loss cone distribution")

# draw loss-cone boundary lines (schematic)
alpha_lc_deg = 27.0
alpha_lc = np.deg2rad(alpha_lc_deg)
m = np.tan(alpha_lc)   # v_perp = m |v_par|
vp = np.linspace(0.0, vmax, 200)
axes[2].plot( m * vp,  vp, ls="--", lw=1.5, color="black")
axes[2].plot(-m * vp,  vp, ls="--", lw=1.5, color="black")
axes[2].plot( m * vp, -vp, ls="--", lw=1.5, color="black")
axes[2].plot(-m * vp, -vp, ls="--", lw=1.5, color="black")
axes[2].text(0.05 * vmax, -0.85 * vmax, rf"$\alpha_{{lc}}={alpha_lc_deg:.0f}^\circ$", fontsize=11)

plt.tight_layout()
plt.savefig("anisotropic_distributions.png", dpi=300)
#plt.show()
plt.close()
# %% END
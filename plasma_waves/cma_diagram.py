# %% IMPORTS
import numpy as np
import matplotlib.pyplot as plt
# %% FUNCTIONS

def stix_params_cma_xy(xp, Y, mu=2.5, Z=1.0):
    """
    CMA in Stix-like coordinates:
      x' = (ω_pe^2 + ω_pi^2)/ω^2
      Y  = |Ω_e|/ω  >= 0
    with mu = m_i/m_e, Z ion charge.

    We express species parameters in terms of x', Y.
    """

    xp = np.asarray(xp, dtype=float)
    Y  = np.asarray(Y, dtype=float)

    # For quasi-neutral two-component plasma:
    # ω_pi^2/ω_pe^2 = Z * m_e/m_i = Z/mu
    # Thus: x' = X + X_i = X*(1 + Z/mu)
    # -> X = x' / (1 + Z/mu), X_i = x' - X
    fac = 1.0 + Z / mu
    X  = xp / fac
    Xi = xp - X

    # Signed gyrofrequency ratios Ω_s/ω:
    # electron: Ω_e = -|Ω_e| -> Ω_e/ω = -Y
    # ion:     Ω_i = +Z e B0/m_i = (Z/mu)|Ω_e| -> Ω_i/ω = +(Z/mu) Y
    Ye = -Y
    Yi = (Z / mu) * Y

    # Cold-plasma Stix parameters:
    # R = 1 - Σ X_s / (1 + Ω_s/ω)
    # L = 1 - Σ X_s / (1 - Ω_s/ω)
    R = 1.0 - (X  / (1.0 + Ye) + Xi / (1.0 + Yi))
    L = 1.0 - (X  / (1.0 - Ye) + Xi / (1.0 - Yi))
    P = 1.0 - xp
    S = 0.5 * (R + L)
    D = 0.5 * (R - L)

    return R, L, S, D, P

def plot_cma(mu=2.5, Z=1.0, xp_max=2.0, Y_max=3.5, N=1200):
    xp = np.linspace(0.0, xp_max, N)
    Y  = np.linspace(0.0, Y_max, N)
    XX, YY = np.meshgrid(xp, Y)

    R, L, S, D, P = stix_params_cma_xy(XX, YY, mu=mu, Z=Z)

    fig = plt.figure(figsize=(6.1, 6.1))
    ax = plt.gca()

    # Boundary lines: P=0, R=0, L=0, S=0, and RL=PS (often drawn dotted)
    ax.contour(XX, YY, P, levels=[0.0], linewidths=2.5)
    ax.contour(XX, YY, R, levels=[0.0], linewidths=2.5)
    ax.contour(XX, YY, L, levels=[0.0], linewidths=2.5)
    ax.contour(XX, YY, S, levels=[0.0], linewidths=2.5)

    F = R * L - P * S
    ax.contour(XX, YY, F, levels=[0.0], linewidths=2.0, linestyles=":")

    # Key resonance markers in Stix plot:
    # |Ω_e| = ω  -> Y = 1
    # Ω_i = ω    -> Y = mu/Z
    ax.axhline(1.0, linewidth=2.0)
    ax.axhline(mu / Z, linewidth=2.0)

    ax.set_xlim(0.0, xp_max)
    ax.set_ylim(0.0, Y_max)

    ax.set_xlabel(r"$x' = (\omega_{pe}^2 + \omega_{pi}^2)/\omega^2$")
    ax.set_ylabel(r"$Y = |\Omega_e|/\omega$")
    ax.set_title(f"Clemmow–Mullaly–Allis diagram (mu={mu:g}, Z={Z:g})")

    # Optional: add a few text labels similar to Stix
    #ax.text(1.02, 0.15, r"$P=0$", transform=ax.transAxes)
    #ax.text(1.02, 0.15, r"$RL=PS$", transform=ax.transAxes)
    #ax.text(0.78, 0.60, r"$RL=PS$", transform=ax.transAxes)
    
    
    # after plotting all boundary curves, before plt.tight_layout()/savefig:

    # --- Stix-style annotations (boundary labels, as in Stix) ---

    ax = plt.gca()

    # helper for consistent text styling
    txt_kw = dict(fontsize=12)

    # Label key boundaries. Adjust x,y slightly if labels overlap with curves.
    ax.text(1.18, 1.38, r"$RL=PS$", **txt_kw)
    ax.text(1.52, 2.08, r"$S=0$", rotation=-8, **txt_kw)

    # horizontal resonance lines:
    #ax.text(1.52, 1.03, r"$|\Omega_e|=\omega$", **txt_kw)
    #ax.text(1.52, mu + 0.03, r"$\Omega_i=\omega$", **txt_kw)
    

    # Optional: label the horizontal lines also with "S=0" etc if you have them as distinct boundaries.
    # In Stix, S=0 is a boundary, often drawn as a slanted curve. Put label near the corresponding curve if present.
    # Example placement (only if you actually plotted S=0 as a separate curve):
    # ax.text(1.68, 2.02, r"$S=0$", rotation=-10, **txt_kw)

    # Vertical line at x'=1 is a prominent divider in your plot
    ax.text(1.02, 3.25, r"$P=0$", rotation=90, va="center", **txt_kw)  # if x'=1 corresponds to P=0 in your chosen x'

    # Labels for R=0, L=0 near the lower left branch family (place near where those boundaries lie)
    ax.text(0.40, 0.58, r"$R=0$", rotation=-30, **txt_kw)
    ax.text(0.52, 0.68, r"$S=0$", rotation=-25, **txt_kw)
    ax.text(1.25, 0.65, r"$L=0$", rotation=0, **txt_kw)


    # Labels for R=∞, L=∞ near the upper right (as in Stix)
    ax.text(1.62, 1.05, r"$R=\infty$", **txt_kw)
    ax.text(1.62, mu + 0.05, r"$L=\infty$", **txt_kw)
    #ax.text(1.52, mu + 0.03, r"$\Omega_i=\omega$", **txt_kw)

    # If you also plotted S=0 explicitly, label it
    # ax.text(1.65, 2.15, r"$S=0$", rotation=-8, **txt_kw)

    # Axis-side arrows, similar to Stix figure (optional, stylistic)
    ax.annotate(r"$|\Omega_e|=\omega$", xy=(0.0, 1.0), xytext=(-0.18, 1.0),
                textcoords="data", ha="right", va="center",
                arrowprops=dict(arrowstyle="->", lw=1.5), fontsize=12)

    ax.annotate(r"$\Omega_i=\omega$", xy=(0.00, mu), xytext=(-0.18, mu),
                textcoords="data", ha="right", va="center",
                arrowprops=dict(arrowstyle="->", lw=1.5), fontsize=12)

    # make sure labels are not clipped
    for t in ax.texts:
        t.set_clip_on(False)
    

    plt.tight_layout()
    plt.savefig("cma_diagram_two_component_plasma.png", dpi=300)
    plt.close()

## %% MAIN
# didactic mass ratio similar to the screenshot could be mu ~ 2.5
# physical proton-electron would be mu=1836
plot_cma(mu=2.5, Z=1.0, xp_max=2.0, Y_max=3.5, N=1200)
# %% END
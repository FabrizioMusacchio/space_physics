"""
Compare Maxwellian vs kappa energy distributions f(W) in 3D,
visualize suprathermal tails, and quantify tail fractions
F(W > Wc) = ∫_{Wc}^{∞} f(W) dW.

This is a static, one variable demonstration in energy space W ≥ 0.
It is meant as a kinetic space physics showcase: kappa distributions
retain a power law tail in collisionless plasmas where Maxwellians fail.

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

def normalize_on_grid(W: np.ndarray, f_unnorm: np.ndarray) -> np.ndarray:
    """
    Normalize a nonnegative PDF defined on a 1D grid W >= 0:
        f = f_unnorm / ∫ f_unnorm dW
    Uses trapezoidal rule.
    """
    Z = np.trapezoid(f_unnorm, W)
    if Z <= 0 or not np.isfinite(Z):
        raise ValueError("Normalization failed: nonpositive or nonfinite integral.")
    return f_unnorm / Z


def maxwell_energy_pdf(W: np.ndarray, kT: float) -> np.ndarray:
    """
    Maxwellian energy distribution (3D) for kinetic energy W:
        f_M(W) ∝ sqrt(W) * exp(-W / kT), W >= 0

    Here kT is an energy scale. For plotting and tail fractions, we normalize on the grid.
    """
    W = np.asarray(W)
    f = np.zeros_like(W, dtype=float)
    mask = W >= 0
    f[mask] = np.sqrt(W[mask]) * np.exp(-W[mask] / kT)
    return f


def kappa_energy_pdf(W: np.ndarray, kappa: float, W0: float) -> np.ndarray:
    """
    Kappa energy distribution (simple standard form in energy space):
        f_k(W) ∝ sqrt(W) * (1 + W/(kappa*W0))^{-(kappa+1)}, W >= 0

    This mirrors the common idea: Maxwellian core with power law tail.
    We normalize on the grid.

    Notes:
    - For meaningful suprathermal tails, choose kappa in roughly [2, 10].
    - W0 is an energy scale that sets the core width.
    """
    if kappa <= 1.5:
        # For many kappa conventions, moments can diverge for small kappa.
        # We keep a conservative guardrail.
        raise ValueError("Choose kappa > 1.5 for a well behaved demonstration.")
    W = np.asarray(W)
    f = np.zeros_like(W, dtype=float)
    mask = W >= 0
    f[mask] = np.sqrt(W[mask]) * (1.0 + W[mask] / (kappa * W0)) ** (-(kappa + 1.0))
    return f


def tail_fraction(W: np.ndarray, f: np.ndarray, Wc: float) -> float:
    """
    Compute tail fraction:
        F(W > Wc) = ∫_{Wc}^{∞} f(W) dW
    Numerically on a finite grid using trapezoidal rule.
    Assumes f is already normalized on the full grid.
    """
    if Wc <= W[0]:
        return 1.0
    if Wc >= W[-1]:
        return 0.0
    idx = np.searchsorted(W, Wc, side="left")
    return float(np.trapezoid(f[idx:], W[idx:]))


# %% MAIN RUN
# -----------------------------
# User adjustable parameters
# -----------------------------
kT = 1.0          # energy unit for Maxwellian
kappa_list = [2.5, 3.0, 5.0, 10.0]  # compare several kappa values
W0 = 1.0          # energy scale for kappa distribution
Wmax = 80.0       # max energy shown
NW = 20000        # resolution in W
Wc_values = [3.0, 5.0, 10.0, 20.0]  # thresholds for tail fractions


# Energy grid. Start slightly above 0 to avoid sqrt singular visuals.
W = np.linspace(0.0, Wmax, NW)

# Compute PDFs
fM = normalize_on_grid(W, maxwell_energy_pdf(W, kT=kT))

fk_dict = {}
for kap in kappa_list:
    fk = normalize_on_grid(W, kappa_energy_pdf(W, kappa=kap, W0=W0))
    fk_dict[kap] = fk

# -----------------------------
# plot 1: linear scale
# -----------------------------
plt.figure(figsize=(6, 4.5))
plt.plot(W, fM, label="Maxwell")
for kap in kappa_list:
    plt.plot(W, fk_dict[kap], label=f"kappa = {kap:g}")
plt.xlim(0, min(Wmax, 25.0))
plt.xlabel("W")
plt.ylabel("f(W)")
plt.title("Energy distribution f(W): Maxwell vs kappa\n(linear scale)")
plt.legend(frameon=False)
plt.tight_layout()
plt.savefig("kappa_vs_maxwell_linear.png", dpi=300)
plt.close()

# -----------------------------
# plot 2: Log scale to expose tails
# -----------------------------
plt.figure(figsize=(6, 4.5))
plt.semilogy(W, fM, label="Maxwell")
for kap in kappa_list:
    plt.semilogy(W, fk_dict[kap], label=f"kappa = {kap:g}")
plt.xlim(0, Wmax)
plt.ylim(1e-12, None)
plt.xlabel("W")
plt.ylabel("f(W) (log)")
plt.title("Energy distribution f(W): Suprathermal tails\n(log scale)")
for Wc in Wc_values:
    plt.axvline(Wc, linewidth=1.0, alpha=0.5)
plt.legend(frameon=False, loc="upper right")
plt.tight_layout()
plt.savefig("kappa_vs_maxwell_log.png", dpi=300)
plt.close()

# -----------------------------
# Tail fractions for selected thresholds
# -----------------------------
print("\nTail fractions F(W > Wc) for selected thresholds")
print(f"Grid: W in [0, {Wmax}] with NW={NW}")
print(f"Maxwell parameter: kT={kT}")
print(f"Kappa parameters: W0={W0}, kappa in {kappa_list}")
print("")
header = "Wc".ljust(8) + "Maxwell".rjust(14)
for kap in kappa_list:
    header += f"  kappa={kap:g}".rjust(14)
print(header)
print("-" * len(header))

for Wc in Wc_values:
    row = f"{Wc:>6.2f}  "
    row += f"{tail_fraction(W, fM, Wc):>12.6e}  "
    for kap in kappa_list:
        row += f"{tail_fraction(W, fk_dict[kap], Wc):>12.6e}  "
    print(row)

# -----------------------------
# plot 3: tail fraction curve F(W>Wc) vs Wc
# -----------------------------
Wc_grid = np.linspace(0.0, Wmax, 600)
FM_tail = np.array([tail_fraction(W, fM, wc) for wc in Wc_grid])

plt.figure(figsize=(6, 4.5))
plt.semilogy(Wc_grid, FM_tail, label="Maxwell")
for kap in kappa_list:
    Fk_tail = np.array([tail_fraction(W, fk_dict[kap], wc) for wc in Wc_grid])
    plt.semilogy(Wc_grid, Fk_tail, label=f"kappa = {kap:g}")
plt.ylim(1e-12, 1.0)
plt.xlim(0, Wmax)
plt.xlabel("Wc")
plt.ylabel("F(W > Wc) (log)")
plt.title("Tail fraction above threshold energy")
plt.legend(frameon=False)
plt.tight_layout()
plt.savefig("kappa_vs_maxwell_tail_fraction.png", dpi=300)
plt.close()
# %% END
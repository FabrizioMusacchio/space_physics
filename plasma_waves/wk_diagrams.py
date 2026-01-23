# %% IMPORTS
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

RESULTS_FOLDER = "plots"
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# remove spines right and top for better aesthetics:
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.left'] = False
plt.rcParams['axes.spines.bottom'] = False
plt.rcParams.update({'font.size': 12})
# %% PHYSICAL CONSTANTS
# define required physical constants (SI):
c = 299_792_458.0
eps0 = 8.8541878128e-12
mu0 = 4e-7 * np.pi
e = 1.602176634e-19
me = 9.1093837015e-31
mp = 1.67262192369e-27
# %% FUNCTIONS
# ----------------------------
# Helper: plasma and cyclotron frequencies
# ----------------------------
def omega_p(n, q, m):
    """Plasma frequency (rad/s). n in m^-3."""
    return np.sqrt(n * q**2 / (eps0 * m))

def Omega_c(B0, q, m):
    """Cyclotron frequency (rad/s). Sign follows q. B0 in Tesla."""
    return q * B0 / m

# ----------------------------
# Cold plasma Stix parameters (R, L, S, D, P)
# ----------------------------
def stix_params(omega, species):
    """
    species: list of dicts with keys {"n","q","m"}
    omega: array (rad/s), positive
    Returns R,L,S,D,P arrays
    """
    omega = np.asarray(omega)

    # Initialize sums
    sum_P = np.zeros_like(omega, dtype=np.float64)
    sum_R = np.zeros_like(omega, dtype=np.float64)
    sum_L = np.zeros_like(omega, dtype=np.float64)

    for sp in species:
        n = sp["n"]
        q = sp["q"]
        m = sp["m"]
        B0 = sp["B0"]

        wps2 = omega_p(n, q, m)**2
        Ocs = Omega_c(B0, q, m)

        # P = 1 - Σ (ω_ps^2 / ω^2)
        sum_P += wps2 / (omega**2)

        # R = 1 - Σ (ω_ps^2 / (ω(ω + Ω_cs)))
        # L = 1 - Σ (ω_ps^2 / (ω(ω - Ω_cs)))
        # Note the sign in Ω_cs is already in Ocs
        sum_R += wps2 / (omega * (omega + Ocs))
        sum_L += wps2 / (omega * (omega - Ocs))

    P = 1.0 - sum_P
    R = 1.0 - sum_R
    L = 1.0 - sum_L
    S = 0.5 * (R + L)
    D = 0.5 * (R - L)
    return R, L, S, D, P

# ----------------------------
# Appleton-Hartree n^2 for oblique propagation
# ----------------------------
def appleton_hartree_n2(omega, theta, species):
    """
    Returns two branches n2_plus, n2_minus from the quadratic in n^2.
    omega: array (rad/s)
    theta: propagation angle in radians
    """
    R, L, S, D, P = stix_params(omega, species)

    sin2 = np.sin(theta)**2
    cos2 = np.cos(theta)**2

    A = S * sin2 + P * cos2
    B = R * L * sin2 + P * S * (1.0 + cos2)
    C = P * R * L

    disc = B**2 - 4.0 * A * C

    # Numerical safety: allow tiny negative due to rounding
    disc = np.where(disc < 0.0, np.nan, disc)
    sqrt_disc = np.sqrt(disc)

    n2_plus = (B + sqrt_disc) / (2.0 * A)
    n2_minus = (B - sqrt_disc) / (2.0 * A)
    return n2_plus, n2_minus

# ----------------------------
# Convert n^2(omega) -> k(omega) and filter physical branches
# ----------------------------
def omega_k_curve(omega, n2):
    """
    Given omega array and n^2 array, return k and omega filtered to real propagating solutions.
    We keep points where n^2 is positive and finite.
    """
    omega = np.asarray(omega)
    n2 = np.asarray(n2)

    mask = np.isfinite(n2) & (n2 > 0.0)
    n = np.sqrt(n2[mask])
    k = (omega[mask] / c) * n
    return k, omega[mask]

# helper to plot sorted curves
def plot_sorted_k(k, omega_hz, **kwargs):
    idx = np.argsort(k)
    plt.plot(k[idx], omega_hz[idx], **kwargs)
    
def plot_monotonic_segments(k, omega_hz, min_len=8, color=None, ls_first="-", ls_rest="--", **kwargs):
    k = np.asarray(k)
    omega_hz = np.asarray(omega_hz)

    if len(k) < min_len:
        return None

    dk = np.diff(k)
    sgn = np.sign(dk)
    sgn[sgn == 0] = 1

    cut = np.where(np.diff(sgn) != 0)[0] + 1
    cuts = np.concatenate(([0], cut, [len(k)]))

    used_color = color
    first = True
    for a, b in zip(cuts[:-1], cuts[1:]):
        if (b - a) < min_len:
            continue
        if first:
            line, = plt.plot(k[a:b], omega_hz[a:b], color=used_color, ls=ls_first, **kwargs)
            if used_color is None:
                used_color = line.get_color()
            first = False
        else:
            plt.plot(k[a:b], omega_hz[a:b], color=used_color, ls=ls_rest)

    return used_color
# %% MAIN RUN

# example parameters: solar-wind-like
parameter_set = "Solar Wind-like"
B0 = 50e-9          # Tesla
ne_cm3 = 5.0        # cm^-3
ne = ne_cm3 * 1e6   # m^-3

# quasi-neutral proton-electron plasma:
species = [
    {"n": ne, "q": -e, "m": me, "B0": B0},  # electrons
    {"n": ne, "q": +e, "m": mp, "B0": B0},  # protons
]

""" # example parameters: magnetosheath-like (Earth)
parameter_set = "Magnetosheath (Earth)"
B0 = 20e-9           # Tesla (often 10–30 nT)
ne_cm3 = 30.0        # cm^-3  (often 10–50 cm^-3)
ne = ne_cm3 * 1e6    # m^-3

# quasi-neutral proton-electron plasma
species = [
    {"n": ne, "q": -e, "m": me, "B0": B0},
    {"n": ne, "q": +e, "m": mp, "B0": B0}] """

""" # example parameters: auroral acceleration region-like (Earth)
parameter_set = "Auroral Region (Earth)"
B0 = 50e-6            # Tesla (50 µT ~ few thousand nT; order-of-magnitude in auroral region)
ne_cm3 = 0.1          # cm^-3 (0.01–1 cm^-3 plausible)
ne = ne_cm3 * 1e6     # m^-3

# quasi-neutral proton-electron plasma
species = [
    {"n": ne, "q": -e, "m": me, "B0": B0},
    {"n": ne, "q": +e, "m": mp, "B0": B0}] """


# characteristic frequencies to guide omega-range:
wpe = omega_p(ne, e, me)
wce = abs(Omega_c(B0, -e, me))
wci = abs(Omega_c(B0, +e, mp))

print(f"omega_pe / (2π) = {wpe/(2*np.pi):.2e} Hz")
print(f"|Omega_ce| / (2π) = {wce/(2*np.pi):.2e} Hz")
print(f"Omega_ci / (2π) = {wci/(2*np.pi):.2e} Hz")

# frequency grid (rad/s):
# use log spacing to capture ion -> electron regimes.
omega_min = 0.05 * wci
if parameter_set == "Auroral Region (Earth)":
    omega_max = np.min([0.5 * wce, 5.0 * wpe])
    if omega_max <= omega_min:
        omega_max = 10.0 * omega_min
else:
    omega_max = 0.5 * wce
omega = np.logspace(np.log10(omega_min), np.log10(omega_max), 6000)


# plots:
# choose angles
thetas_deg = [0.0, 30.0, 60.0, 85.0]

plt.figure(figsize=(5.5, 5.5))
legend_handles = {}
# use Matplotlib's default color cycle, but assign ONE color per theta:
cycle = plt.rcParams['axes.prop_cycle'].by_key().get('color', [])
if len(cycle) == 0:
    cycle = ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9"]
for i, th_deg in enumerate(thetas_deg):
    th = np.deg2rad(th_deg)
    base_color = cycle[i % len(cycle)]  # fixed per theta

    n2p, n2m = appleton_hartree_n2(omega, th, species)
    kp, op = omega_k_curve(omega, n2p)
    km, om = omega_k_curve(omega, n2m)

    # + branch: solid
    label_p = f"+ branch, θ={th_deg:.0f}°"
    plot_monotonic_segments(
        kp, op/(2*np.pi),
        color=base_color,
        ls_first="-",
        ls_rest="-",
        label=label_p
    )

    # − branch: dashed
    label_m = f"− branch, θ={th_deg:.0f}°"
    plot_monotonic_segments(
        km, om/(2*np.pi),
        color=base_color,
        ls_first="--",
        ls_rest="--",
        label=label_m
    )

    # legend proxies: same color, different linestyle:
    if label_p not in legend_handles:
        legend_handles[label_p] = Line2D([0], [0], color=base_color, lw=2, ls="-", label=label_p)
    if label_m not in legend_handles:
        legend_handles[label_m] = Line2D([0], [0], color=base_color, lw=2, ls="--", label=label_m)

plt.xscale("log")
plt.yscale("log")
plt.xlabel("k [m$^{-1}$]")
plt.ylabel("frequency f = $\\omega/(2\\pi)$ [Hz]")
plt.title(f"Cold-plasma electromagnetic dispersion:\n$(\\omega)/(k)$ via Appleton–Hartree\nParameter set: {parameter_set}")
plt.legend(handles=list(legend_handles.values()), fontsize=8, ncols=2)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_FOLDER, f"cold_plasma_appleton_hartree_dispersion_{parameter_set.replace(' ', '_')}.png"), dpi=300)
plt.close()
# %% IO-PLASMA TORUS-like

# example parameters: Io torus-like (Jupiter)
B0 = 2_000e-9         # Tesla (2000 nT = 2 µT; order-of-magnitude at Io torus)
ne_cm3 = 2000.0       # cm^-3 (10^3–10^4 cm^-3 plausible)
ne = ne_cm3 * 1e6     # m^-3

# choose a simple composition:
# electrons balance total positive charge
# ions: 5% H+, 50% O+, 45% S+ (number fractions; adjust as you like)
n_H = 0.05 * ne
n_O = 0.50 * ne
n_S = 0.45 * ne

m_O = 16.0 * mp
m_S = 32.0 * mp

species = [
    {"n": ne,   "q": -e, "m": me, "B0": B0},  # electrons
    {"n": n_H,  "q": +e, "m": mp, "B0": B0},  # protons (minor)
    {"n": n_O,  "q": +e, "m": m_O,"B0": B0},  # O+
    {"n": n_S,  "q": +e, "m": m_S,"B0": B0},  # S+
]
# characteristic frequencies to guide omega-range:
wpe = omega_p(ne, e, me)
wce = abs(Omega_c(B0, -e, me))
wci_H = abs(Omega_c(B0, +e, mp))
wci_O = abs(Omega_c(B0, +e, m_O))
wci_S = abs(Omega_c(B0, +e, m_S))

print(f"\nIo torus-like parameters:")
print(f"omega_pe / (2π) = {wpe/(2*np.pi):.2e} Hz")
print(f"|Omega_ce| / (2π) = {wce/(2*np.pi):.2e} Hz")
print(f"Omega_ci (H+) / (2π) = {wci_H/(2*np.pi):.2e} Hz")
print(f"Omega_ci (O+) / (2π) = {wci_O/(2*np.pi):.2e} Hz")
print(f"Omega_ci (S+) / (2π) = {wci_S/(2*np.pi):.2e} Hz") 

# frequency grid (rad/s):
# use log spacing to capture ion -> electron regimes.
omega_min = 0.05 * wci_S
omega_max = 0.5 * wce
omega = np.logspace(np.log10(omega_min), np.log10(omega_max), 6000)

# plots:
# choose angles
thetas_deg = [0.0, 30.0, 60.0, 85.0]
legend_handles = {}
# use Matplotlib's default color cycle, but assign ONE color per theta:
cycle = plt.rcParams['axes.prop_cycle'].by_key().get('color', [])
if len(cycle) == 0:
    cycle = ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9"]
plt.figure(figsize=(5.5, 5.5))
for i, th_deg in enumerate(thetas_deg):
    th = np.deg2rad(th_deg)
    base_color = cycle[i % len(cycle)]  # fixed per theta

    n2p, n2m = appleton_hartree_n2(omega, th, species)

    kp, op = omega_k_curve(omega, n2p)
    km, om = omega_k_curve(omega, n2m)
    
    if f"+ branch, θ={th_deg:.0f}°" not in legend_handles:
        label_p = f"+ branch, θ={th_deg:.0f}°"
        plot_monotonic_segments(
            kp, op/(2*np.pi),
            color=base_color,
            ls_first="-",
            ls_rest="-",
            label=label_p)
        legend_handles[label_p] = Line2D([0], [0], color=base_color, lw=2, ls="-", label=label_p)
    if f"− branch, θ={th_deg:.0f}°" not in legend_handles:
        label_m = f"− branch, θ={th_deg:.0f}°"
        plot_monotonic_segments(
            km, om/(2*np.pi),
            color=base_color,
            ls_first="--",
            ls_rest="--",
            label=label_m)
        legend_handles[label_m] = Line2D([0], [0], color=base_color, lw=2, ls="--", label=label_m)

    # Plot omega(k) for both branches
    # Convert omega to Hz for readability
    #plt.plot(kp, op/(2*np.pi), label=f"+ branch, θ={th_deg:.0f}°")
    #plt.plot(km, om/(2*np.pi), label=f"− branch, θ={th_deg:.0f}°")
    #plot_sorted_k(kp, op/(2*np.pi), label=f"+ branch, θ={th_deg:.0f}°")
    #plot_sorted_k(km, om/(2*np.pi), label=f"− branch, θ={th_deg:.0f}°")
    # plot_monotonic_segments(kp, op/(2*np.pi), label=f"+ branch, θ={th_deg:.0f}°")
    # plot_monotonic_segments(km, om/(2*np.pi), label=f"− branch, θ={th_deg:.0f}°")
plt.xscale("log")
plt.yscale("log")
plt.xlabel("k [m$^{-1}$]")
plt.ylabel("frequency f = $\\omega/(2\\pi)$ [Hz]")
plt.title(f"Cold-plasma electromagnetic dispersion:\n$(\omega)/(k)$ via Appleton–Hartree\nParameter set: Io torus-like")
plt.legend(handles=legend_handles.values(), fontsize=8, ncols=2)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_FOLDER, "cold_plasma_appleton_hartree_dispersion_Io_torus_like.png"), dpi=300)
plt.close()
# %% END
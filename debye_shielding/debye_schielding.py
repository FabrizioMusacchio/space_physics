""" 
This script calculates and plots the Debye shielding potential around a test
charge in a plasma. It compares the screened potential to the unscreened
Coulomb potential.

This script is based on scipython.com's tutorial "The Debye length"
(https://scipython.com/blog/the-debye-length/), with modifications.

author: Fabrizio Musacchio
date: Jun 2020 / Jan 2026
"""
# %% IMPORTS
import numpy as np
from scipy.constants import k as kB, epsilon_0, e
from matplotlib import rc
import matplotlib.pyplot as plt

# set up LaTeX-style plotting.
# rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'size': 16})
# rc('text', usetex=True)

# remove spines right and top for better aesthetics
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.left'] = False
plt.rcParams['axes.spines.bottom'] = False
# set global font site zu 12:
plt.rcParams.update({'font.size': 12})
#rc('text.latex', preview=True)
# %% DEBYE LENGTH AND POTENTIAL FUNCTIONS
def calc_debye_length(Te, n0):
    """Return the Debye length for a plasma characterized by Te, n0.

    The electron temperature Te should be given in eV and density, n0
    in cm-3. The debye length is returned in m.

    """

    return np.sqrt(epsilon_0 * Te / e / n0 / 1.e-6)

def calc_unscreened_potential(r, qT):
   return qT * e / 4 / np.pi / epsilon_0 / r
def calc_e_potential(r, lam_De, qT):
    return calc_unscreened_potential(r, qT) * np.exp(-r / lam_De)

# %% MAIN SCRIPT
# plasma electron temperature (eV) and density (cm-3) for a typical tokamak.
Te, n0 = 1.e8 * kB / e, 1.e26
lam_De = calc_debye_length(Te, n0)
print(lam_De)

# range of distances to plot phi for, in m.
rmin = lam_De / 10
rmax = lam_De * 5
r = np.linspace(rmin, rmax, 100)
qT = 1
phi_unscreened = calc_unscreened_potential(r, qT)
phi = calc_e_potential(r, lam_De, qT)

# Plot the figure. Apologies for the ugly and repetitive unit conversions from
# m to µm and from V to mV.
fig, ax = plt.subplots(figsize=(6,4))
ax.plot(r*1.e6, phi_unscreened * 1000,
                label=r'Unscreened: $\phi = \frac{e}{4\pi\epsilon_0 r}$')
ax.plot(r*1.e6, phi * 1000,
                label=r'Screened: $\phi = \frac{e}{4\pi\epsilon_0 r}'
                      r'e^{-r/\lambda_\mathrm{D}}$')
ax.axvline(lam_De*1.e6, ls='--', c='k')
ax.annotate(text=r'$\lambda_\mathrm{D} = %.1f \mathrm{\mu m}$' % (lam_De*1.e6),
            xy=(lam_De*1.1*1.e6, max(phi_unscreened)/2 * 1000))
ax.legend()
ax.set_xlabel(r'$r \; [\mathrm{\mu m}]$')
ax.set_ylabel(r'$\phi ; [\mathrm{mV}]$')
ax.set_title('Debye shielding potential around a test charge')
plt.tight_layout()
plt.savefig('debye_length.png', dpi=300)
#plt.show()
plt.close()
# %% END
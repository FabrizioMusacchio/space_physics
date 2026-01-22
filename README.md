# Space Physics. A collection of Python scripts

This repository contains Python scripts for different examples from the field of space plasma physics. Each scripts belongs to one of the following blog posts, which provide detailed explanations of the implemented physics and numerical methods:


* Space Physics: A definitional perspective (Overview article)
* Characteristics of a plasma: Collective behavior, shielding, and intrinsic time scales
  * ⟶ `debye_shielding/`
  * ⟶ `plasma_types/`
* Single-particle description of plasmas: Equation of motion, gyration, and ExB drift
  * ⟶ `gyromotion/`
* Adiabatic invariants and magnetic mirrors
  * ⟶ `magnetic_mirrors/`
* Magnetohydrodynamics (MHD): A theoretical overview with a numerical toy example
  * ⟶ `mhd_simulation/`
* The solar wind and the Parker model
  * ⟶ `parker_spiral/`
* Magnetic reconnection: Theory and a simple numerical model
  * ⟶ `magnetic_reconnection/`
* Magnetic reconnection via X-point collapse
  * ⟶ `magnetic_reconnection/`
* Plasma waves in space plasmas
  * ⟶ `plasma_waves/`
* The Alfvén wave as a fundamental mode of magnetized plasmas
  * ⟶ `plasma_waves/`
* Plasma instabilities as dynamical departures from equilibrium (overview article)
* Kinetic plasma theory: From distribution functions to the Vlasov equation
  * ⟶ `vlasov_equation/`
* Vlasov–Poisson dynamics: Landau damping and the two-stream instability
  * ⟶ `vlasov_equation/`
* What velocity moments miss: Core plus beam distributions
  * ⟶ `vlasov_equation/`
* Bi-Maxwellian distributions and anisotropic pressure
  * ⟶ `vlasov_equation/`
* Kappa versus Maxwell distributions: Suprathermal tails in collisionless plasmas
  * ⟶ `vlasov_equation/`
* Krook collision operator as velocity-space relaxation
  * ⟶ `vlasov_equation/`
* Particle-in-Cell methods in kinetic plasma simulations
  * ⟶ `particle_in_cell/`
* A spectral (FFT) Poisson solver for 1D electrostatic PIC
  * ⟶ `particle_in_cell/`

## Installation
For reproducibility, it is recommended to create a new conda environment with the following packages:

```bash
conda create -n plasmaphysics python=3.12 mamba -y
conda activate plasmaphysics
mamba install -y numpy matplotlib scipy imageio ffmpeg ipykernel ipython
```

## Usage
Each script can be run directly using Python. In particular, they are written in such a way, that they can be interactively executed cell-by-cell, e.g., in VSCode's interactive window. 

# Contact and Support
For questions or suggestions, please open an issue on GitHub or contact the author via email: [Fabrizio Musacchio](mailto:fabrizio.musacchio@posteo.de)



# Space Physics 
**A collection of educational Python scripts**

![Particle in Cell Simulation](particle_in_cell/pic_evolution.gif)

This repository contains Python scripts for different examples from the field of space plasma physics. Each scripts belongs to one of the following blog posts, which provide detailed explanations of the implemented physics and numerical methods:


* [Space Physics: A definitional perspective](https://www.fabriziomusacchio.com/blog/2020-08-23-space_physics/) (Overview article)
* [Characteristics of a plasma: Collective behavior, shielding, and intrinsic time scales](https://www.fabriziomusacchio.com/blog/2020-06-05-plasma_characteristics/)
  * ⟶ `debye_shielding/`
  * ⟶ `plasma_types/`
* [Single-particle description of plasmas: Equation of motion, gyration, and ExB drift](https://www.fabriziomusacchio.com/blog/2020-06-06-single_particle_description_of_plasma/)
  * ⟶ `gyromotion/`
* [Adiabatic invariants and magnetic mirrors](https://www.fabriziomusacchio.com/blog/2020-08-18-magnetic_mirrors/)
  * ⟶ `magnetic_mirrors/`
* [Magnetohydrodynamics (MHD): A theoretical overview with a numerical toy example](https://www.fabriziomusacchio.com/blog/2020-08-19-mhd/)
  * ⟶ `mhd_simulation/`
* [The solar wind and the Parker model](https://www.fabriziomusacchio.com/blog/2020-08-21-solar_wind_and_parker_spiral/)
  * ⟶ `parker_spiral/`
* [Magnetic reconnection: Theory and a simple numerical model](https://www.fabriziomusacchio.com/blog/2020-08-22-magnetic_reconnection/)
  * ⟶ `magnetic_reconnection/`
* [Magnetic reconnection via X-point collapse](https://www.fabriziomusacchio.com/blog/2020-08-22-magnetic_reconnection_y_point_collapse/)
  * ⟶ `magnetic_reconnection/`
* [Planetary aurorae](https://www.fabriziomusacchio.com/blog/2020-08-30-planetary_aurorae/) (Overview article)
* [Plasma waves in space plasmas](https://www.fabriziomusacchio.com/blog/2020-09-01-plasma_waves/)
  * ⟶ `plasma_waves/`
* [The Alfvén wave as a fundamental mode of magnetized plasmas](https://www.fabriziomusacchio.com/blog/2020-09-01-alfven_wave/)
  * ⟶ `plasma_waves/`
* [Plasma instabilities as dynamical departures from equilibrium](https://www.fabriziomusacchio.com/blog/2020-09-01-plasma_instabilities/) (overview article)
* [Kinetic plasma theory: From distribution functions to the Vlasov equation](https://www.fabriziomusacchio.com/blog/2020-09-05-kinetic_plasma_theory/)
  * ⟶ `vlasov_equation/`
* [Vlasov–Poisson dynamics: Landau damping and the two-stream instability](https://www.fabriziomusacchio.com/blog/2020-09-06-vlasov_poisson_dynamics/)
  * ⟶ `vlasov_equation/`
* [What velocity moments miss: Core plus beam distributions](https://www.fabriziomusacchio.com/blog/2020-09-10-velocity_moments/)
  * ⟶ `vlasov_equation/`
* [Bi-Maxwellian distributions and anisotropic pressure](https://www.fabriziomusacchio.com/blog/2020-09-11-bi_maxwellian_distribution/)
  * ⟶ `vlasov_equation/`
* [Kappa versus Maxwell distributions: Suprathermal tails in collisionless plasmas](https://www.fabriziomusacchio.com/blog/2020-09-12-kappa_vs_maxwellian_distribution/)
  * ⟶ `vlasov_equation/`
* [Krook collision operator as velocity-space relaxation](https://www.fabriziomusacchio.com/blog/2020-09-18-krook_collision_operator/)
  * ⟶ `vlasov_equation/`
* [Particle-in-Cell methods in kinetic plasma simulations](https://www.fabriziomusacchio.com/blog/2020-09-23-particle_in_cell/)
  * ⟶ `particle_in_cell/`
* [A spectral (FFT) Poisson solver for 1D electrostatic PIC](https://www.fabriziomusacchio.com/blog/2020-09-26-particle_in_cell_with_fft_poisson_solver/)
  * ⟶ `particle_in_cell/`

The scripts are intended as didactic and conceptual examples. They prioritize clarity and physical transparency over numerical efficiency or large-scale applicability. The focus is on illustrating fundamental plasma-physical mechanisms and standard modeling approaches rather than providing optimized or fully general simulation frameworks.

Many scripts deliberately rely on reduced models, idealized geometries, or simplified boundary conditions. These choices are made to keep the connection between equations, numerical implementation, and physical interpretation as direct as possible.

The repository reflects the state of the accompanying blog series and may evolve over time. Backward compatibility is not guaranteed, but changes are typically driven by conceptual clarification rather than feature expansion.

## Installation
For reproducibility, create a new conda environment with the following packages:

```bash
conda create -n plasmaphysics python=3.12 mamba -y
conda activate plasmaphysics
mamba install -y numpy matplotlib scipy imageio ffmpeg ipykernel ipython
```

## Usage
Each script can be run directly using the Python environment described above. In particular, they are written in such a way, that they can be interactively executed cell-by-cell, e.g., in VS Code's interactive window. You can also place them in a Jupyter notebook for step-by-step execution.

## Citation
If you use code from this repository for your own research, teaching material, or derived software, please consider citing the Zenodo archive associated with this repository. Proper citation helps acknowledge the original source, provides context for the implemented physical models and numerical assumptions, and supports reproducibility.

When appropriate, citing the specific blog post that discusses the underlying physics and numerical methods in detail is encouraged in addition to the repository itself.

If you use substantial parts of the code in an academic publication, a reference to both the repository and the associated blog article is recommended.

Here is the suggested citation format for the repository:

> Musacchio, F. (2026). *Space Physics: A collection of educational Python scripts*. Zenodo. https://doi.org/10.5281/zenodo.18344030

```bibtex
@software{musacchio_space_physics_2026,
  author       = {Musacchio, Fabrizio},
  title        = {Space Physics: A collection of educational Python scripts},
  year         = {2026},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.18344030},
  url          = {https://doi.org/10.5281/zenodo.18344030}
}
```


Thank you for considering proper citation practices.

## Contact and support
For questions or suggestions, please open an issue on GitHub or contact the author via email: [Fabrizio Musacchio](mailto:fabrizio.musacchio@posteo.de)



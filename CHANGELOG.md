# Release Nnotes for space physics educational script collection

# 🚀 Release v1.0.0

This release marks the first complete and stable publication of the *Space Physics* script collection. It consolidates all Python scripts used throughout the accompanying space plasma physics blog series into a single, citable repository.

Version v1.0.0 establishes a coherent reference point for teaching, reuse, and reproducible exploration of fundamental concepts in space plasma physics.

## 📦 Scope and content
This release includes Python scripts covering a broad range of core topics in space plasma physics, from single-particle motion to kinetic simulations. Each script is directly associated with a dedicated blog post that provides detailed physical and mathematical context.

Included topics are:

* plasma characteristics and collective behavior
* single-particle dynamics, gyromotion, and ExB drift
* adiabatic invariants and magnetic mirrors
* ideal and resistive magnetohydrodynamics
* solar wind theory and the Parker spiral
* magnetic reconnection and X-point collapse models
* plasma waves and Alfvén waves
* kinetic plasma theory and the Vlasov equation
* Landau damping and two-stream instability
* non-Maxwellian velocity distributions
* Krook-type collision operators
* particle-in-cell methods, including spectral Poisson solvers

The repository structure reflects this thematic organization and mirrors the progression of the blog series.

## 🧠 Conceptual focus
The scripts in this repository are designed as **didactic and conceptual examples**. Emphasis is placed on:

* physical transparency
* direct correspondence between equations and code
* minimal numerical and algorithmic overhead
* clarity over computational performance

Many models deliberately rely on reduced geometries, simplified boundary conditions, or idealized assumptions to keep the physical mechanisms explicit.

## 🔬 Reproducibility and usage
All scripts are compatible with a lightweight Python environment based on NumPy, SciPy, and Matplotlib. They are written to support both direct execution and interactive, cell-by-cell exploration in development environments such as VS Code or Jupyter.

This release provides a stable baseline for reuse in:

* teaching and coursework
* self-study
* illustrative figures and animations
* methodological extensions

Backward compatibility across future releases is not guaranteed, but changes will primarily serve conceptual clarification rather than feature expansion.

## 📖 Citation and archiving
This release is archived on Zenodo and assigned a DOI, making it citable in academic contexts.

Users of this repository are encouraged to cite the Zenodo record and, where appropriate, the corresponding blog posts that document the physical background and numerical choices in detail.

## 🔖 Versioning note
Version v1.0.0 supersedes the initial v0.0.1 placeholder release, which primarily established the repository structure. No prior code is deprecated by this release.

## 📝 License
All code is released under the GPL-3.0 License.

## ✨ Outlook
Future releases may expand individual examples, refine numerical implementations, or add complementary scripts aligned with new blog posts. Any such extensions will build on the conceptual baseline established with this release.
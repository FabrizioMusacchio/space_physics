# %% IMPORTS
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, FancyArrowPatch, Wedge
from matplotlib.path import Path

# Optional: match your aesthetic choices
plt.rcParams["axes.spines.right"] = False
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.left"] = False
plt.rcParams["axes.spines.bottom"] = False
plt.rcParams.update({"font.size": 12})
# %% FUNCTIONS

plt.rcParams.update({"font.size": 14})


def save_pic_grid_particles(
    outpath="pic_grid_particles.png",
    *,
    Nx=8,
    Ny=6,
    dx=1.0,
    seed=2,
):
    """
    PIC cartoon: Eulerian grid and Lagrangian macro-particles with velocity arrows.
    Clean, no ticks, no dense text.
    """
    rng = np.random.default_rng(seed)

    Lx = (Nx - 1) * dx
    Ly = (Ny - 1) * dx

    fig, ax = plt.subplots(figsize=(7.8, 6))
    ax.set_xlim(-0.6 * dx, Lx + 0.6 * dx)
    ax.set_ylim(-0.6 * dx, Ly + 0.9 * dx)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Grid lines
    for i in range(Nx):
        x = i * dx
        ax.plot([x, x], [0, Ly], lw=2, color="0.75")
    for j in range(Ny):
        y = j * dx
        ax.plot([0, Lx], [y, y], lw=2, color="0.75")

    # Grid nodes as small dark points
    for i in range(Nx):
        for j in range(Ny):
            ax.scatter(i * dx, j * dx, s=14, color="0.25", zorder=3)

    # Macro-particles
    n_particles = 10
    pos = np.column_stack([
        rng.uniform(0.7 * dx, Lx - 0.7 * dx, size=n_particles),
        rng.uniform(0.7 * dx, Ly - 0.7 * dx, size=n_particles)
    ])

    # Two species coloring
    colors = np.array(["#3B82F6"] * n_particles)
    colors[: n_particles // 2] = "#EF4444"
    rng.shuffle(colors)

    # Random velocities for arrows
    vel = rng.normal(0, 1.0, size=(n_particles, 2))
    vel /= np.maximum(np.linalg.norm(vel, axis=1, keepdims=True), 1e-12)
    vel *= 0.65 * dx

    # Draw particles as soft disks and arrows
    for (x, y), (vx, vy), c in zip(pos, vel, colors):
        circ = Circle((x, y), radius=0.28 * dx, facecolor=c, edgecolor="none", alpha=0.35, zorder=4)
        ax.add_patch(circ)
        ax.scatter(x, y, s=25, color=c, zorder=5)
        ax.arrow(
            x, y, vx, vy,
            width=0.02 * dx, head_width=0.18 * dx, head_length=0.18 * dx,
            length_includes_head=True, color="0.15", alpha=0.8, zorder=6
        )

    # small scale markers:
    ax.annotate("", xy=(6.0, -0.35 * dx), xytext=(7.0*dx, -0.35 * dx),
                arrowprops=dict(arrowstyle="<->", lw=2, color="0.2"))
    ax.text(6.5 * dx, -0.52 * dx, r"$\Delta x$", ha="center", va="top")

    ax.text(0.07, 0.92, "Lagrangian macro-particles", transform=ax.transAxes, ha="left", va="top")
    ax.text(0.07, 0.02, "Eulerian fields on a grid", transform=ax.transAxes, ha="left", va="bottom")

    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def save_pic_processing_loop(
    outpath="pic_processing_loop.png",
):
    """
    PIC processing loop diagram: Push, Deposit, Field solve, Gather, loop with dt.
    """
    fig, ax = plt.subplots(figsize=(9.75, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Box helper
    def box(x, y, w, h, text, *, fc="#E5E7EB", ec="0.25"):
        r = Rectangle((x, y), w, h, facecolor=fc, edgecolor=ec, lw=2, joinstyle="round")
        ax.add_patch(r)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center")
        return r

    # Arrow helper
    def arrow(x0, y0, x1, y1, *, lw=3):
        a = FancyArrowPatch(
            (x0, y0), (x1, y1),
            arrowstyle="-|>", mutation_scale=18,
            lw=lw, color="0.15", connectionstyle="arc3"
        )
        ax.add_patch(a)

    # Place boxes around a loop
    w, h = 2.8, 0.9
    b_push   = box(3.6, 4.6, w, h, "Push particles\n(Boris/leapfrog)")
    b_depos  = box(6.9, 2.9, w, h, "Deposit charge/current\n(particle → grid)")
    b_solve  = box(3.6, 1.2, w, h, f"Field solve\n(Poisson or Maxwell)")
    b_gather = box(0.3, 2.9, w, h, "Gather fields\n(grid → particle)")


    # Loop arrows
    arrow(5.0, 4.6, 7.0, 3.8)   # push -> deposit
    arrow(8.3, 2.9, 5.0, 2.1)   # deposit -> solve
    arrow(3.6, 1.65, 2.0, 2.9)  # solve -> gather
    arrow(1.7, 3.8, 3.6, 4.6)   # gather -> push

    # Central dt
    ax.text(4.9, 3.2, r"$\Delta t$", ha="center", va="center", fontsize=16, color="0.2")

    # Minimal side annotations, not dense
    #ax.text(6.95, 5.7, r"$\frac{d\mathbf{x}}{dt}=\mathbf{v}$", ha="left", va="top", color="0.2")
    #ax.text(6.95, 5.25, r"$\frac{d\mathbf{v}}{dt}=\frac{q}{m}\left(\mathbf{E}+\mathbf{v}\times\mathbf{B}\right)$", ha="left", va="top", color="0.2")
    ax.text(6.55, 5.55,
            "Particle push (Lorentz ODE):\n"
            r"$\frac{d\mathbf{x}}{dt}=\mathbf{v}$" "\n"
            r"$\frac{d\mathbf{v}}{dt}=\frac{q}{m}\left(\mathbf{E}+\mathbf{v}\times\mathbf{B}\right)$",
            ha="left", va="top", color="0.2")

    ax.text(3.65, 0.55,
            r"Electrostatic: Poisson solve ($\nabla^2\phi=\rho$)" "\n"
            r"Electromagnetic: Maxwell update",
            ha="left", va="bottom", color="0.2")
    

    ax.text(0.5, 1.02, "Particle in Cell time step", transform=ax.transAxes, 
            ha="center", va="top", fontweight="bold", fontsize=16)

    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


# %% MAIN
os.makedirs("pic_diagrams", exist_ok=True)
save_pic_grid_particles(os.path.join("pic_diagrams", "pic_grid_particles.png"))
save_pic_processing_loop(os.path.join("pic_diagrams", "pic_processing_loop.png"))
# %% END
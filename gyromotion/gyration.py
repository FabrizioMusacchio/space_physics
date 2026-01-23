"""
Single-particle motion in prescribed electromagnetic and magnetic fields
(Lorentz equation). Numerical integration with a fixed-step 4th-order 
Runge–Kutta (RK4) method.

This script is based on scipython.com's tutorial "Gyromotion of a charged 
particle in a magnetic field"
(https://scipython.com/blog/gyromotion-of-a-charged-particle-in-a-magnetic-field/),
with modifications and extensions.

author: Fabrizio Musacchio
date: Jun 2020
"""
# %% IMPORTS
import numpy as np
from matplotlib import animation, rc
rc('animation', html='html5')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from IPython.display import HTML

# remove spines right and top for better aesthetics
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.left'] = False
plt.rcParams['axes.spines.bottom'] = False
# set grid style:
plt.rcParams['grid.color'] = (0.8, 0.8, 0.8)
plt.rcParams['grid.linestyle'] = '-'
#plt.rcParams['grid.linewidth'] = 0.5
# %% FUNCTIONS

# define a simple RK4 integrator:
def rk4_integrate(rhs, X0, t, args=()):
    """
    Fixed step RK4 integrator for a system X' = rhs(X,t,*args).
    t must be an array of time points with constant spacing.
    Returns array X of shape (len(t), len(X0)).
    """
    t = np.asarray(t, dtype=float)
    dt = t[1] - t[0]
    if not np.allclose(np.diff(t), dt):
        raise ValueError("Time grid must be equally spaced for fixed-step RK4.")

    X = np.empty((len(t), len(X0)), dtype=float)
    X[0] = np.asarray(X0, dtype=float)

    for i in range(len(t) - 1):
        ti = t[i]
        Xi = X[i]

        k1 = rhs(Xi, ti, *args)
        k2 = rhs(Xi + 0.5*dt*k1, ti + 0.5*dt, *args)
        k3 = rhs(Xi + 0.5*dt*k2, ti + 0.5*dt, *args)
        k4 = rhs(Xi + dt*k3, ti + dt, *args)

        X[i+1] = Xi + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)

    return X

def lorentz(X, t, q_over_m, E=np.array((0,0,0)), B=np.array((0,0,1))):
        """
        The equations of motion for the Lorentz force on a particle with
        q/m given by q_over_m. X=[x,y,z,vx,vy,vz] defines the particle's
        position and velocity at time t: F = ma = (q/m)[E + v×B].
        
        """
        
        v = X[3:]
        drdt = v
        dvdt = q_over_m * (E + np.cross(v, B))
        return np.hstack((drdt, dvdt))

def larmor(q, m, v0, B=np.array((0,0,1))):
    """Calculate the Larmor (cyclotron) radius.

    rho = m.v_perp / (|q|B) where m is the particle's mass,
    q is its charge, B is the magnetic field strength and
    v_perp is its instantaneous speed perpendicular to the
    magnetic field, **B**.

    """

    B_sq = B @ B
    v0_par = (v0 @ B) * B / B_sq
    v0_perp = v0 - v0_par
    v0_perp_abs = np.sqrt( v0_perp @ v0_perp)
    return m * v0_perp_abs / np.abs(q) / np.sqrt(B_sq)

def calc_trajectory(q, m, r0=None, v0=np.array((1,0,1)), E=np.array((0,0,0)), B=np.array((0,0,1))):
    """Calculate the particle's trajectory.
    
    q, m are the particle charge and mass;
    r0 and v0 are its initial position and velocity vectors.
    If r0 is not specified, it is calculated to be the Larmor
    radius of the particle, and particles with different q, m
    are placed so as have a common guiding centre (for E=0).
    
    """
    if r0 is None:
        rho = larmor(q, m, v0, B)
        vp = np.array((v0[1],-v0[0],0))
        r0 = -np.sign(q) * vp * rho
    # final time, number of time steps, time grid.
    tf = 50
    N = 10 * tf
    # create time array (time discretization):
    t = np.linspace(0.0, tf, int(N))
    # fixed time step dt = tf / N (sufficiently small for stable gyromotion)
    # initial position and velocity components:
    X0 = np.hstack((r0, v0))
    # numerical integration of the equation of motion:
    X = rk4_integrate(lorentz, X0, t, args=(q/m, E, B))
    return t, X

def setup_axes(ax):
    """Style the 3D axes the way we want them."""
    
    # change the axis juggling to have z vertical, x horizontal:
    ax.yaxis._axinfo['juggled'] = (1,1,2)
    ax.zaxis._axinfo['juggled'] = (1,2,0)
    
    # remove axes ticks and labels, the grey panels:
    for axis in ax.xaxis, ax.yaxis, ax.zaxis:
        for e in axis.get_ticklines() + axis.get_ticklabels():
            e.set_visible(False) 
        axis.pane.set_visible(False)
        #axis.gridlines.set_visible(False)
    # label the x and z axes only:
    ax.set_xlabel('x', labelpad=-10, size=16)
    ax.set_zlabel('z', labelpad=-10, size=16)
    
def plot_trajectories(trajectories, title="Particle Trajectories", plotname=None):
    """Produce a static plot of the trajectories.
    
    trajectories should be a sequence of (n,3) arrays representing
    the n timepoints over which the (x,y,z) coordinates of the
    particle are given.
    
    """
        
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111, projection='3d')
    for X in trajectories:
        ax.plot(*X.T[:3])
    setup_axes(ax)
    # Plot a vertical line through the origin parallel to the z axis:
    # (equals the magnetic field direction here if B has only z component):
    zmax = np.max(np.max([X.T[2] for X in trajectories]))
    ax.plot([0,0],[0,0],[0,zmax],lw=1.5,c='gray', alpha=0.8)
    plt.title(title)
    
    # set equal aspect ratio for all axes:
    XYZ = np.vstack([X[:, :3] for X in trajectories])
    mins = XYZ.min(axis=0)
    maxs = XYZ.max(axis=0)
    marg = 0.05 * (maxs - mins + 1e-12)
    xmin, ymin, zmin = mins - marg
    xmax, ymax, zmax = maxs + marg
    if np.abs(zmax) < 1e-12 and np.abs(zmin) < 1e-12:
        zmax = np.max([np.abs(xmin), np.abs(xmax)])
        zmin = -np.max([np.abs(xmin), np.abs(xmax)])
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_zlim(zmin, zmax)
    #print(xmin, xmax, ymin, ymax, zmin, zmax)
    ax.set_box_aspect((xmax - xmin, ymax - ymin, zmax - zmin))
    
    
    plt.tight_layout()
    if plotname is None:
        plotname = title.replace(" ", "_").lower() + ".png"
    plt.savefig(plotname, dpi=200)
    #plt.show()

# %% SOME BASIC EXAMPLES (W/O ANIMATION)
# define magnetic and electric field directions and strengths:
B = np.array((0,0,1))
# E = np.array((0,0.0,0)) # no E field -> pure gyration motion in xy plane
E = np.array((0,0.01,0)) # small E field in y direction

""" 
Since E has a y-component only, and B is along z, the particles will have an
E×B drift in the x direction with V_{E×B} = E×B / B^2 = (E_y * B_z - E_z * B_y, E_z * B_x - E_x * B_z, E_x * B_y - E_y * B_x) / B^2
= (0.01*1 - 0*0, 0*0 - 0*1, 0*0 - 0.01*0) / 1^2 = (0.01, 0, 0).
I.e., the drift will not propagate along in z direction. The particle trajectories
have only components:
- Gyromotion around the magnetic field lines
- E×B drift perpendicular to both E and B fields
- no parallel motion along the magnetic field lines (since v0_z=0 here).
"""

# define electron mass and charge (here assumed normalized units):
me1, qe1 = 1, -1
me2, qe2 = 0.5*me1, qe1
v0 = np.array((0.1,0,0.0)) # initial velocity only in x direction; change this later to (0.1,0,0.1) to have parallel motion too

t1, X1 = calc_trajectory(   qe1, me1, r0=(0,0,0), v0=v0, E=E, B=B)
t3, X3 = calc_trajectory(  -qe2, me2, r0=(0,0,0), v0=v0, E=E, B=B)
plot_trajectories([X1, X3], title=f"Particle Trajectories for q and -q w/ v0={v0}",
                  plotname=f"particle_trajectories_v0_{v0[0]}_{v0[1]}_{v0[2]}.png")
plt.close()

fig = plt.figure(figsize=(6,4))
plt.plot(X1[:,0], X1[:,1], label=f'q={qe1}')
plt.plot(X3[:,0], X3[:,1], label=f'q={-qe2}')
plt.axis('equal')
plt.title("Particle Trajectories Projection in xy Plane")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.tight_layout()
plt.savefig("particle_trajectories_v0_0.1_0_0_xy_projection.png", dpi=200)
plt.close()
# %% ANIMATION OF PARTICLE GYRATION 
""" 
Now, we plot the particle trajectories which have three components:
- Gyromotion around the magnetic field lines
- Parallel motion along the magnetic field lines (as we choose v0_z ≠ 0 here!)
- E×B drift perpendicular to both E and B fields
"""

# charges and masses of the electron (e) and ion (i); we assume normalized units:
qe, me = -1, 1
qi, mi = 1, 3
v0=np.array((1,0,1))

E = np.array((0,0.00,0))
#E = np.array((0,0.05,0)) # try different E field values; here: particles will have an E×B drift
#E = np.array((0,0.15,0))
# B = np.array((0,0.15,1)) # or induce an oblique B field
B = np.array((0,0.00,1))
""" 
In case E≠0, the particles will have an E×B drift and the guiding centers
will move with V_{E\times_B} = E×B / B^2 .
"""

# calculate the trajectories for the particles (which don't interact):
te, Xe = calc_trajectory(qe, me, E=E, B=B, v0=v0)
ti, Xi = calc_trajectory(qi, mi, E=E, B=B, v0=v0)

# make a static 3D plot of the trajectories:
plot_trajectories([Xe, Xi])
plt.close()

# and a 2D plot of the xy projection:
fig = plt.figure(figsize=(6,4))
plt.plot(Xe[:,0], Xe[:,1], label=f"q={qe} ('electron')")
plt.plot(Xi[:,0], Xi[:,1], label=f"q={qi} ('ion')")
# make x-y axes equal:
plt.axis('equal')
plt.legend()
plt.title("Particle gyration in xy plane")
plt.xlabel("x")
plt.ylabel("y")
plt.tight_layout()
plt.savefig("particle_trajectories_xy_projection_e_and_i.png", dpi=200)
plt.close()

# precompute E×B drift (if any) guiding center (perpendicular GC, shown here at z=0):
B_sq = float(B @ B)
V_EB = np.cross(E, B) / B_sq # constant drift velocity
# initial guiding center position (at t=0) and guiding center trajectory:
Rgc0 = np.array([0.0, 0.0, 0.0])
Rgc_e = Rgc0[None, :] + te[:, None] * V_EB[None, :]
Rgc_i = Rgc0[None, :] + ti[:, None] * V_EB[None, :]

# calculate axis limits:
XYZ = np.vstack((Xe[:, :3], Xi[:, :3]))
mins = XYZ.min(axis=0)
maxs = XYZ.max(axis=0)
marg = 0.05 * (maxs - mins + 1e-12)
xmin, ymin, zmin = mins - marg
xmax, ymax, zmax = maxs + marg

# calculate the helix axis:
bhat = B / np.linalg.norm(B) # # unit vector along B
# parallel velocities (constant in homogeneous fields):
vpar_e = float(np.dot(Xe[0, 3:], bhat))
vpar_i = float(np.dot(Xi[0, 3:], bhat))
# helix axes: perpendicular guiding center drift + parallel transport
Raxis_e = Rgc_e + (te[:, None] * vpar_e) * bhat[None, :]
Raxis_i = Rgc_i + (ti[:, None] * vpar_i) * bhat[None, :]


def init():
    """
    Function to initialize the trajectory animation. Used by FuncAnimation and
    constructs the initial plot elements. 
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_zlim(zmin, zmax)
    ax.set_box_aspect((xmax-xmin, ymax-ymin, zmax-zmin))
    ax.view_init(elev=25, azim=-60)

    # reference B direction line through origin; when E=0, 
    # this is also guiding center line of the gyro-motion; if E≠0, it's at the GC
    # at t=0; along z):
    ax.plot([0,0],[0,0],[zmin, zmax], lw=2, c="gray", alpha=0.6,
            label="magnetic field direction")

    # start empty, not full trajectories for electron motion, ion motion:
    lne, = ax.plot([], [], [], c="tab:blue", lw=2, label="electron")
    lni, = ax.plot([], [], [], c="tab:orange", lw=2, label="ion")
    dGC  = ax.plot([], [], [], "ok", label="guiding center") # dummy plot for legend of guiding center

    # plot the particles instantaneous positions as a scatter plot, scaled
    # according to particle mass:
    SCALE = 40
    particles = ax.scatter(*np.vstack((Xe[0][:3], Xi[0][:3])).T,
                           s=(me*SCALE, mi*SCALE),
                           c=("tab:blue", "tab:orange"), depthshade=0)

    # plot guiding center (GC) points (two black dots):
    gc_scatter = ax.scatter([Rgc_e[0,0], Rgc_i[0,0]],
                            [Rgc_e[0,1], Rgc_i[0,1]],
                            [Rgc_e[0,2], Rgc_i[0,2]],
                            s=25, c="k", depthshade=0)

    # plot helix axis traces (also time evolving):
    axis_e_line, = ax.plot([], [], [], "--", lw=1.5, alpha=0.6, c="tab:blue",
                           label="helix axis (electron)")
    axis_i_line, = ax.plot([], [], [], "--", lw=1.5, alpha=0.6, c="tab:orange",
                           label="helix axis (ion)")
    # add current point on each helix axis (x markers):
    axis_pts = ax.scatter([Raxis_e[0,0], Raxis_i[0,0]],
                          [Raxis_e[0,1], Raxis_i[0,1]],
                          [Raxis_e[0,2], Raxis_i[0,2]],
                          s=20, c=("tab:blue", "tab:orange"), marker="o", depthshade=0,
                          alpha=0.6)

    # add some title info including E and B fields:
    ax.set_title("Numerical integration of particle motion\n"
                 f"E = ({E[0]:.2f}, {E[1]:.2f}, {E[2]:.2f}), "
                 f"B = ({B[0]:.2f}, {B[1]:.2f}, {B[2]:.2f})\n")
    
    # add a legend:
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.0))
    
    # apply custom axes function:
    setup_axes(ax)

    return fig, ax, lne, lni, particles, gc_scatter, axis_e_line, axis_i_line, axis_pts

def animate(i):
    """ 
    The main animation function, called for each frame by FuncAnimation.
    Updates the plot elements to show the particle trajectories up to
    time step i.
    """
    
    # update electron and ion trajectories up to step i:
    lne.set_data(Xe[:i,0], Xe[:i,1])
    lne.set_3d_properties(Xe[:i,2])

    lni.set_data(Xi[:i,0], Xi[:i,1])
    lni.set_3d_properties(Xi[:i,2])

    # update particle positions:
    particles._offsets3d = np.vstack((Xe[i][:3], Xi[i][:3])).T

    # also indicate the current guiding center position of each particle
    # on the xy plane (z=0); actually two points, but usually overlapping:
    gc_scatter._offsets3d = np.array([[Rgc_e[i,0], Rgc_i[i,0]],
                                      [Rgc_e[i,1], Rgc_i[i,1]],
                                      [Rgc_e[i,2], Rgc_i[i,2]]])
    
    # plot helix axis lines up to current time:
    axis_e_line.set_data(Raxis_e[:i, 0], Raxis_e[:i, 1])
    axis_e_line.set_3d_properties(Raxis_e[:i, 2])

    axis_i_line.set_data(Raxis_i[:i, 0], Raxis_i[:i, 1])
    axis_i_line.set_3d_properties(Raxis_i[:i, 2])

    # indicate current helix-axis points (also two points, but usually overlapping):
    axis_pts._offsets3d = np.array([[Raxis_e[i,0], Raxis_i[i,0]],
                                    [Raxis_e[i,1], Raxis_i[i,1]],
                                    [Raxis_e[i,2], Raxis_i[i,2]]])

    # add a time stamp:
    time_text = f"t = {te[i]:.2f}" # unitless time (!)
    if not hasattr(animate, "time_annotation"):
        animate.time_annotation = ax.text2D(0.05, 0.95, time_text, transform=ax.transAxes)
    else:
        animate.time_annotation.set_text(time_text)

print("animating...")
fig, ax, lne, lni, particles, gc_scatter, axis_e_line, axis_i_line, axis_pts = init()
anim = animation.FuncAnimation(fig, animate, frames=len(te), interval=10, blit=False)
HTML(anim.to_html5_video())
plt.tight_layout()
# save animation as mp4 video file:
anim.save(f'gyration_animation_E{E[1]:.2f}_B{B[1]:.2f}.mp4', writer='ffmpeg', fps=30, dpi=200)
plt.close()
print("animation complete.")

# %% PITCH ANGLE EXPERIMENTS
"""
In the previous examples, the initial velocity implicitly fixed the pitch angle.
Here we explicitly parametrize the initial velocity by the pitch angle α, defined
as the angle between the velocity vector and the magnetic field direction.
"""

alpha = np.radians(150) # pitch angle in degrees
v0_mag = 0.1 # magnitude of initial velocity
v0 = v0_mag * np.array([np.sin(alpha), 0.0, np.cos(alpha)])

# define some charges and masses of the electron (e) and ion (i):
qe, me = -1, 1
qi, mi = 1, 3

E = np.array((0,0.00,0))
B = np.array((0,0.00,1))

# calculate the trajectories for the particles:
te, Xe = calc_trajectory(qe, me, E=E, B=B, v0=v0)
ti, Xi = calc_trajectory(qi, mi, E=E, B=B, v0=v0)

# make a static 3D plot of the trajectories:
plot_trajectories([Xe, Xi], title=f"Particle Trajectories with Pitch Angle {np.degrees(alpha):.1f}°",
                  plotname=f"particle_trajectories_pitch_angle_{np.degrees(alpha):.1f}_deg.png")
plt.close()


# %% END
    
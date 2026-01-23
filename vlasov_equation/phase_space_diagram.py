# %% IMPORTS
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
# %% PLOT PHASE-SPACE SCHEMATIC
# -----------------------------
# phase-space schematic (x, v)
# -----------------------------
fig, ax = plt.subplots(figsize=(5, 5))

# Coordinate limits
ax.set_xlim(0, 10)
ax.set_ylim(0, 8)

# Hide default spines and ticks for a clean schematic look
for spine in ax.spines.values():
    spine.set_visible(False)
ax.set_xticks([])
ax.set_yticks([])

# Draw axes with arrowheads
ax.annotate("", xy=(10, 0), xytext=(0, 0),
            arrowprops=dict(arrowstyle="-|>", lw=2, color="black"))
ax.annotate("", xy=(0, 8), xytext=(0, 0),
            arrowprops=dict(arrowstyle="-|>", lw=2, color="black"))

# Axis labels
ax.text(10.15, -0.15, "x", fontsize=18, ha="left", va="center")
ax.text(-0.2, 8.15, "v", fontsize=18, ha="center", va="bottom")

# Choose a volume element location and size
x0, v0 = 6.8, 5.0
dx, dv = 1.2, 1.2

# Dashed guide lines for dv (horizontal) and dx (vertical)
# Horizontal dashed lines at v0 and v0+dv, from x=0 to x0
ax.plot([0, x0], [v0, v0], ls=(0, (8, 6)), lw=2, color="black")
ax.plot([0, x0], [v0 + dv, v0 + dv], ls=(0, (8, 6)), lw=2, color="black")

# Vertical dashed lines at x0 and x0+dx, from v=0 to v0
ax.plot([x0, x0], [0, v0], ls=(0, (8, 6)), lw=2, color="black")
ax.plot([x0 + dx, x0 + dx], [0, v0], ls=(0, (8, 6)), lw=2, color="black")

# Draw the volume element (rectangle in phase space)
rect = Rectangle((x0, v0), dx, dv, facecolor="lightgray", edgecolor="black", lw=2)
ax.add_patch(rect)

# Draw the "exact particle position" as a dot inside the rectangle
dot_x = x0 + 0.65 * dx
dot_v = v0 + 0.55 * dv
ax.plot(dot_x, dot_v, "o", color="black", markersize=10)

# Label "volume element" with an arrow pointing to the rectangle
ax.annotate("volume element",
            xy=(x0 + 0.55 * dx, v0 + dv),
            xytext=(x0 + 0.2 * dx, v0 + dv + 1.2),
            fontsize=16,
            arrowprops=dict(arrowstyle="->", lw=1.8, color="black"),
            ha="center", va="bottom")

# Label "exact particle position" with an arrow pointing to the dot
ax.annotate("exact\nparticle\nposition",
            xy=(dot_x+0.15, dot_v),
            xytext=(x0 + dx + 1.0, v0 + 0.2 * dv),
            fontsize=16,
            arrowprops=dict(arrowstyle="->", lw=1.8, color="black"),
            ha="left", va="center")

# Bracket-like markers and labels for dv (left side) and dx (bottom)
# dv bracket on the left near the y-axis
x_br = 0.6
ax.plot([x_br, x_br], [v0, v0 + dv], lw=2, color="black")
ax.plot([x_br - 0.25, x_br + 0.25], [v0, v0], lw=2, color="black")
ax.plot([x_br - 0.25, x_br + 0.25], [v0 + dv, v0 + dv], lw=2, color="black")
ax.text(x_br - 0.75, v0 + 0.5 * dv, "dv", fontsize=16, ha="right", va="center")

# dx bracket on the bottom below the dashed verticals
y_br = -0.3
ax.plot([x0, x0 + dx], [y_br, y_br], lw=2, color="black", clip_on=False)
ax.plot([x0, x0], [y_br - 0.18, y_br + 0.18], lw=2, color="black", clip_on=False)
ax.plot([x0 + dx, x0 + dx], [y_br - 0.18, y_br + 0.18], lw=2, color="black", clip_on=False)
ax.text(x0 + 0.5 * dx, y_br - 0.35, "dx", fontsize=16, ha="center", va="top", clip_on=False)

plt.tight_layout()
plt.savefig("phase_space_schematic.png", dpi=300)
#plt.show()
plt.close()
# %% END
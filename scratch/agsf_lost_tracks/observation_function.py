"""Plot of the observation function g of mtt (simulators/mtt.py): bearing arctan2(x2, x1) and range |(x1, x2)|.
Debug view only (rule 5). The function is taken from the module, not written again.
Panels: (1) bearing over the plane, with the branch cut on the negative x1-axis; (2) range; (3) bearing along vertical
lines x1 = const that cross the cut, far from and near the origin, with a 2-sigma band of a component (sd 0.02).
Usage (repository root, jaxenv):  python scratch/agsf_lost_tracks/observation_function.py
"""
import os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "gaussfiltax")); os.chdir(ROOT)
import jax.numpy as jnp
from jax import vmap
from simulators import mtt

g = lambda x: mtt.g(x, jnp.zeros(2), 0, None)      # noise-free observation [bearing, range]
L = 15.0
x1, x2 = jnp.meshgrid(jnp.linspace(-L, L, 601), jnp.linspace(-L, L, 601))
X = jnp.stack([x1.ravel(), jnp.zeros(x1.size), x2.ravel(), jnp.zeros(x1.size)], axis=1)
Y = vmap(g)(X)
bearing, rng = Y[:, 0].reshape(x1.shape), Y[:, 1].reshape(x1.shape)

fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
im = ax[0].pcolormesh(x1, x2, bearing, cmap="RdBu_r", shading="auto", vmin=-jnp.pi, vmax=jnp.pi)
ax[0].plot([-L, 0], [0, 0], "r--", lw=1.5, label="branch cut (jump of $2\\pi$)")
fig.colorbar(im, ax=ax[0], label="bearing $g_1(x)$ (rad)")
ax[0].set_title("Bearing $g_1 = \\mathrm{arctan2}(x_2, x_1)$")

cs = ax[1].contour(x1, x2, rng, levels=jnp.arange(2, 22, 2), cmap="viridis")
ax[1].clabel(cs, fontsize=7)
ax[1].set_title("Range $g_2 = \\sqrt{x_1^2 + x_2^2}$")

s = jnp.linspace(-1.0, 1.0, 2001)
for c, col in [(-12.0, "C0"), (-1.0, "C1"), (-0.1, "C2")]:
    b = vmap(g)(jnp.stack([jnp.full_like(s, c), jnp.zeros_like(s), s, jnp.zeros_like(s)], axis=1))[:, 0]
    ax[2].plot(s, b, color=col, label=f"$x_1 = {c}$")
ax[2].axvspan(-0.04, 0.04, color="grey", alpha=0.3, label="$\\pm 2$ sd of a component (sd 0.02)")
ax[2].set_xlabel("$x_2$"); ax[2].set_ylabel("bearing (rad)")
ax[2].set_title("Bearing across the cut ($x_1 < 0$)")
ax[2].legend(fontsize=8, loc="center right")

for a in ax[:2]:
    a.set_xlabel("$x_1$"); a.set_ylabel("$x_2$"); a.set_aspect("equal")
ax[0].legend(fontsize=8, loc="lower left")
fig.suptitle("Observation function of mtt (Exp. A): $y = g(x) + r$, noise added to the raw bearing (no wrap)")
fig.tight_layout()
out = os.path.join(os.path.dirname(__file__), "figures", "observation_function")
for ext in ("png", "pdf"):
    fig.savefig(f"{out}.{ext}", dpi=170, bbox_inches="tight")
print("saved", out + ".{png,pdf}")

"""The mtt observation with a raw bearing (now) and with a wrapped bearing (option 3 of D-30). Debug view only (rule 5).
g(x) does not change. The wrap changes the residual and thus the likelihood:
  raw:      p(y|x) = N(y1 - g1(x); 0, s2) N(y2 - g2(x); 0, s2)
  wrapped:  p(y|x) = N(wrap(y1 - g1(x)); 0, s2) N(y2 - g2(x); 0, s2),  wrap(d) = (d + pi) mod 2 pi - pi
Panels: (1) bearing as a point on the unit circle, (cos g1, sin g1): no jump; (2) and (3) the likelihood of one
observation near the cut (bearing pi - 0.05, range 12, s2 = 0.025) over the plane, raw and wrapped;
(4) the bearing residual along vertical lines that cross the cut, raw (dashed) and wrapped (solid).
Usage (repository root, jaxenv):  python scratch/agsf_lost_tracks/observation_function_wrapped.py
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

g = lambda x: mtt.g(x, jnp.zeros(2), 0, None)       # noise-free [bearing, range]
wrap = lambda d: (d + jnp.pi) % (2 * jnp.pi) - jnp.pi
S2 = 0.025
Y = jnp.array([jnp.pi - 0.05, 12.0])                 # one observation just above the cut

L = 15.0
x1, x2 = jnp.meshgrid(jnp.linspace(-L, L, 601), jnp.linspace(-L, L, 601))
X = jnp.stack([x1.ravel(), jnp.zeros(x1.size), x2.ravel(), jnp.zeros(x1.size)], axis=1)
G = vmap(g)(X)
b = G[:, 0].reshape(x1.shape)
d_raw = Y[0] - G[:, 0]
d_wrap = wrap(d_raw)
loglik = lambda d1: (-0.5 * (d1 ** 2 + (Y[1] - G[:, 1]) ** 2) / S2).reshape(x1.shape)

fig, ax = plt.subplots(1, 4, figsize=(21, 4.8))

# (1) the bearing on the unit circle: the two coordinates are smooth across the cut
im = ax[0].pcolormesh(x1, x2, jnp.sin(b), cmap="RdBu_r", shading="auto", vmin=-1, vmax=1)
ax[0].contour(x1, x2, jnp.cos(b), levels=[-0.9, -0.5, 0, 0.5, 0.9], colors="k", linewidths=0.6)
ax[0].plot([-L, 0], [0, 0], "r--", lw=1.2, label="old cut (no jump now)")
fig.colorbar(im, ax=ax[0], label="$\\sin g_1$ (colors), $\\cos g_1$ (lines)")
ax[0].set_title("Bearing as an angle: $(\\cos g_1, \\sin g_1) = (x_1, x_2)/|x|$", fontsize=10)
ax[0].legend(fontsize=8, loc="lower left")

# (2), (3) likelihood of one observation near the cut
for k, (d1, name) in enumerate([(d_raw, "raw bearing (now)"), (d_wrap, "wrapped bearing")]):
    a = ax[1 + k]
    ll = loglik(d1)
    im = a.pcolormesh(x1, x2, jnp.exp(ll - ll.max()), cmap="magma", shading="auto", vmin=0, vmax=1)
    a.plot([-L, 0], [0, 0], "c--", lw=1.0)
    a.set_title(f"$p(y\\,|\\,x)$, {name}\n$y$ = (bearing $\\pi$ - 0.05, range 12), $\\sigma^2$ = {S2}", fontsize=10)
    a.set_xlim(-15, -5); a.set_ylim(-5, 5)
    fig.colorbar(im, ax=a, label="likelihood / max")

# (4) bearing residual across the cut, relative to the observation y1 = pi - 0.05
s = jnp.linspace(-1.0, 1.0, 2001)
for c, col in [(-12.0, "C0"), (-1.0, "C1"), (-0.1, "C2")]:
    gb = vmap(g)(jnp.stack([jnp.full_like(s, c), jnp.zeros_like(s), s, jnp.zeros_like(s)], axis=1))[:, 0]
    ax[3].plot(s, Y[0] - gb, "--", color=col, lw=1.0)
    ax[3].plot(s, wrap(Y[0] - gb), "-", color=col, label=f"$x_1 = {c}$")
ax[3].set_xlabel("$x_2$"); ax[3].set_ylabel("$y_1 - g_1(x)$ (rad)")
ax[3].set_title("Bearing residual across the cut\nraw (dashed), wrapped (solid)", fontsize=10)
ax[3].legend(fontsize=8)

for a in ax[:3]:
    a.set_xlabel("$x_1$"); a.set_ylabel("$x_2$"); a.set_aspect("equal")
fig.tight_layout()
out = os.path.join(os.path.dirname(__file__), "figures", "observation_function_wrapped")
for ext in ("png", "pdf"):
    fig.savefig(f"{out}.{ext}", dpi=170, bbox_inches="tight")
print("saved", out + ".{png,pdf}")

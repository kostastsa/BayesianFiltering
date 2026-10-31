import sys
sys.path.insert(0, '/Users/kostastsampourakis/Desktop/code/Python/projects/gaussfiltax/gaussfiltax')
from types import SimpleNamespace
import numpy as onp
import jax
import jax.numpy as jnp
import jax.random as jr
from jax import lax
from simulators.ssm import NonlinearSSM
from filters import pf

leaf = lambda v: SimpleNamespace(value=jnp.asarray(v))
F = jnp.array([[1.0, 1.0], [0.0, 1.0]])
H = jnp.array([[1.0, 0.0]])
Q = jnp.diag(jnp.array([0.05, 0.05]))
R = jnp.array([[float(sys.argv[1]) if len(sys.argv) > 1 else 0.5]])
m0, P0 = jnp.zeros(2), jnp.eye(2)

params = SimpleNamespace(
    initial=SimpleNamespace(mean=leaf(m0), cov=leaf(P0)),
    dynamics=SimpleNamespace(bias=leaf(jnp.zeros(2)), cov=leaf(Q)),
    emissions=SimpleNamespace(bias=leaf(jnp.zeros(1)), cov=leaf(R)),
)
f = lambda x, q, u, p: F @ x + q
g = lambda x, r, u, p: H @ x + r
ssm = NonlinearSSM(2, 2, 1, 1, f, g)

T = 50
key = jr.PRNGKey(1)
k1, k2 = jr.split(key)
qs = jr.multivariate_normal(k1, jnp.zeros(2), Q, (T,))
rs = jr.multivariate_normal(k2, jnp.zeros(1), R, (T,))
x = jr.multivariate_normal(jr.PRNGKey(2), m0, P0)
ys = []
for t in range(T):
    x = F @ x + qs[t]
    ys.append(H @ x + rs[t])
ys = jnp.stack(ys)


def kf(ys):
    def step(c, y):
        m, P, ll = c
        m = F @ m
        P = F @ P @ F.T + Q
        S = H @ P @ H.T + R
        K = P @ H.T @ jnp.linalg.inv(S)
        e = y - H @ m
        ll += -0.5 * (e @ jnp.linalg.solve(S, e) + jnp.log(jnp.linalg.det(2 * jnp.pi * S)))
        m = m + K @ e
        P = (jnp.eye(2) - K @ H) @ P
        return (m, P, ll), m
    (_, _, ll), ms = lax.scan(step, (m0, P0, 0.0), ys)
    return ms, ll


ms_kf, ll_kf = kf(ys)
print('KF ll', float(ll_kf))

for name, cls in [('BPF', pf.BPF), ('APF', pf.APF)]:
    for M in [100, 1000, 10000]:
        errs, lls = [], []
        for seed in range(5):
            out, ll = cls(None, ssm).run(params, ys, M, jr.PRNGKey(seed), None, 0.5)
            w, xs = out['weights'], out['particles']
            mean = jnp.einsum('it,itd->td', w, xs)
            errs.append(float(jnp.sqrt(jnp.mean((mean - ms_kf) ** 2))))
            lls.append(float(ll) - float(ll_kf))
        print(f'{name} M={M:6d}  RMSE(mean vs KF)={onp.mean(errs):.4f}  ll-ll_KF={onp.mean(lls):+.3f} (sd {onp.std(lls):.3f})')

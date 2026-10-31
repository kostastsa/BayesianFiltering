import sys
sys.path.insert(0, '/Users/kostastsampourakis/Desktop/code/Python/projects/gaussfiltax/gaussfiltax')
import numpy as onp
import jax.numpy as jnp
import jax.random as jr
from simulators import mtt
from util.param import sample_prior
import util.numerics as un
from filters import pf

M = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
n_seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 4
T = 500

out = mtt.setup(4, 2, 0)
ssm, props = out['ssm'], out['props']
true_ps = sample_prior(jr.PRNGKey(0), props, 1)[0]
inputs = mtt.get_inputs(None, T)

rows = {'bpf': [], 'apf': []}
for seed in range(n_seeds):
    states, obs = ssm.simulate(jr.PRNGKey(100 + seed), true_ps, T, inputs)
    for name, cls in [('bpf', pf.BPF), ('apf', pf.APF)]:
        res = cls(props, ssm).run(true_ps, obs, M, jr.PRNGKey(seed), inputs, 0.5)
        mse_repo, _ = un.mse(res, name, states)
        w, x = res[0]['weights'], res[0]['particles']
        w = w / jnp.sum(w, axis=0, keepdims=True)
        mean_w = jnp.einsum('it,itd->td', w, x)
        mse_w = jnp.mean(jnp.sum((mean_w - states) ** 2, axis=-1))
        ess = 1.0 / jnp.sum(w ** 2, axis=0)
        rows[name].append((float(mse_repo), float(mse_w), float(jnp.mean(ess)), float(jnp.min(ess))))
        print(f'seed {seed} {name}: MSE(repo, unweighted)={rows[name][-1][0]:.3f}  MSE(weighted)={rows[name][-1][1]:.3f}  '
              f'mean ESS={rows[name][-1][2]:.1f}  min ESS={rows[name][-1][3]:.1f}', flush=True)

for name in rows:
    a = onp.array(rows[name])
    print(f'{name.upper()} M={M}: MSE repo={a[:,0].mean():.3f}  weighted={a[:,1].mean():.3f}  mean ESS={a[:,2].mean():.1f}')

from jax import numpy as jnp
import jax.random as jr
import numpy as np
from jax import jacfwd, jit, jacrev
from jax.tree_util import tree_map
import utils
from typing import NamedTuple
from jaxtyping import Float32
from itertools import count
from tensorflow_probability.substrates.jax.distributions import MultivariateNormalFullCovariance as MVN
import tensorflow_probability.substrates.jax.distributions as tfd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse



class MixtureModel(NamedTuple):
    means : jnp.ndarray
    covs : jnp.ndarray
    weights : jnp.ndarray

class AugmentedJointApproximation():

    

    def __init__(self, num_comp, f, dim_in, dim_out, mu, cov, cov_tol, Delta):
        self.num_comp = num_comp
        self.f = f
        self.jacobian = jit(jacfwd(f))
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.mu = mu
        self.cov = cov
        self.cov_tol = cov_tol
        self.Delta = Delta

    def _sample_particles(self, keys):
        particles = np.zeros((self.num_comp, self.dim_in))
        for i in range(self.num_comp):
            particles[i] = MVN(self.mu, self.cov-self.Delta).sample(seed=next(keys))
        self.particles = particles


    
    def return_posterior(self, y0, seed)->MixtureModel:
        self._sample_particles(seed)
        means = np.zeros((self.num_comp, self.dim_in))
        covs = np.zeros((self.num_comp, self.dim_in, self.dim_in))
        weights = np.zeros(self.num_comp)
        lls = np.zeros(self.num_comp)
        grads = np.zeros((self.num_comp, self.dim_in, self.dim_out))
        for n in range(self.num_comp):
            Jn = np.expand_dims(self.jacobian(self.particles[n]), axis=1)
            mu_y = np.float32(self.f(self.particles[n]))
            Sy = self.cov_tol + Jn.T @ self.Delta @ Jn
            Omy = jnp.linalg.inv(Sy)
            means[n] = self.particles[n] + self.Delta @ Jn @ Omy @ (y0 - mu_y)
            covs[n] = self.cov - self.Delta @ Jn @ Omy @ Jn.T @ self.Delta
            lls[n] = MVN(loc=mu_y, covariance_matrix=Sy).prob(y0)
            grads[n] = Jn
        weights = jnp.exp(lls - jnp.max(lls))
        weights /= jnp.sum(weights)
        _posterior = MixtureModel(means, covs, weights)
        return _posterior, grads

def plot_cov_ellipse(cov, pos, nstd=2, ax=None, **kwargs):
    """
    Plots an `nstd` sigma error ellipse based on the specified covariance
    matrix (`cov`). Additional keyword arguments are passed on to the 
    ellipse patch artist.
    Parameters
    ----------
        cov : The 2x2 covariance matrix to base the ellipse on
        pos : The location of the center of the ellipse. Expects a 2-element
            sequence of [x0, y0].
        nstd : The radius of the ellipse in numbers of standard deviations.
            Defaults to 2 standard deviations.
        ax : The axis that the ellipse will be plotted on. Defaults to the 
            current axis.
        Additional keyword arguments are pass on to the ellipse patch.
    Returns
    -------
        A matplotlib ellipse artist
    """
    def eigsorted(cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:,order]

    if ax is None:
        ax = plt.gca()

    vals, vecs = eigsorted(cov)
    theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))

    # Width and height are "full" widths, not radius
    width, height = 2 * nstd * np.sqrt(vals)
    ellip = Ellipse(xy=pos, width=width, height=height, angle=theta, **kwargs)

    ax.add_artist(ellip)
    return ellip


# f = lambda x : jnp.dot(x, x) / 2.0

# dx = 2
# Nprt = 100
# mu = jnp.array([0.01, 0.01])
# cov = 10 * jnp.eye(2)
# Q = 0.01 * jnp.eye(1)
# y0 = 1.0
# hessian = jit(jacfwd(jacrev(f)))
# hess = hessian(mu)
# Delta = 0.9999 * cov #utils.sdp_opt(2, Nprt , 1, cov, cov, hess, 10, 0.01)
# print(Delta)
# model = AugmentedJointApproximation(Nprt, f, dx, 1, mu, cov, Q, Delta)
# # # posterior = model.return_posterior(1.0)
# keys = map(jr.PRNGKey, count(11))
# posterior = model.return_posterior([y0], keys)
# # print(posterior.weights.shape)
# # print(posterior.covs)

# ## Plots
# ax = plt.gca()
# p = posterior.weights
# p = np.asarray(p).astype('float64')
# p = p / np.sum(p)
# idx = np.random.choice(Nprt, 40, p=p)
# for i in idx:
#     plot_cov_ellipse(np.sqrt(p[i]) * posterior.covs[i], posterior.means[i], nstd=2, ax=ax, alpha=0.5)
# plt.scatter(posterior.means[:, 0], posterior.means[:, 1], s=5, c='r')
# plt.show()



# # key = jr.PRNGKey(0)
# # inputs = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
# # next_inputs = tree_map(lambda x: x[1:], inputs)
# # next_keys = jr.split(key, 3 - 1)
# # init = 0.0
# # out = lax.scan(lambda x: x*x, init, next_inputs)
# # print(out)


# # Initialize a single 3-variate Gaussian.
# # mu = jnp.array([1., 2, 3])
# # cov = [[ 0.36,  0.12,  0.06],
# #        [ 0.12,  0.29, -0.13],
# #        [ 0.06, -0.13,  0.26]]
# # mvn = tfd.MultivariateNormalFullCovariance(
# #     loc=jnp.array([1., 2, 3]),
# #     covariance_matrix=jnp.eye(3))

# # print(mvn.prob([-1., 0, 1]))

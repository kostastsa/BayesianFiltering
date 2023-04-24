from jax import numpy as jnp
import jax.random as jr
import numpy as np
from jax import jacfwd, jit, jacrev
from jax.tree_util import tree_map
import gaussfiltax.utils as utils
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


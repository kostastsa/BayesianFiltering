import jax
import jax.numpy as jnp
import tensorflow_probability.substrates.jax as tfp

tfd = tfp.distributions

class OscPrior(tfd.Distribution):
    def __init__(self, uniform_low, uniform_high, gaussian_loc, gaussian_scale, validate_args=False, allow_nan_stats=True, name="OscPrior"):
        parameters = dict(locals())  # Store parameters for TFP
        self.uniform_low = jnp.array(uniform_low)
        self.uniform_high = jnp.array(uniform_high)
        self.gaussian_loc = jnp.array(gaussian_loc)
        self.gaussian_scale = jnp.array(gaussian_scale)
        self.shape = (4,)

        super().__init__(
            dtype=jnp.float32,
            reparameterization_type=tfd.NOT_REPARAMETERIZED,
            validate_args=validate_args,
            allow_nan_stats=allow_nan_stats,
            parameters=parameters,
            name=name
        )

    # @property
    # def event_shape(self):
    #     """Defines the event shape as (4,) since each sample is a 4D vector."""
    #     return tuple([4])

    # def _event_shape_tensor(self):
    #     """Returns a tensor representation of the event shape."""
    #     return jnp.array([4])

    def _log_prob(self, value):
        """Computes log probability of given values under OscPrior distribution."""
        uniform_prob = jnp.all((value >= self.uniform_low) & (value <= self.uniform_high), axis=-1) / jnp.prod(self.uniform_high - self.uniform_low)
        gaussian_prob = jnp.exp(-0.5 * jnp.sum(((value - self.gaussian_loc) / self.gaussian_scale) ** 2, axis=-1)) / jnp.prod(self.gaussian_scale * jnp.sqrt(2 * jnp.pi))
        return jnp.log(uniform_prob * gaussian_prob + 1e-10)  # Small epsilon to prevent log(0)

    def _sample_n(self, n, seed=None):
        """Samples from the distribution using importance sampling (Uniform as proposal)."""
        if seed is None:
            seed = jax.random.PRNGKey(0)
        key1, key2 = jax.random.split(seed)

        # Sample from Uniform proposal
        # uniform_samples = jax.random.uniform(key1, shape=(n, 4), minval=self.uniform_low, maxval=self.uniform_high)
        gaussian_samples = jax.random.normal(key2, shape=(n, 4)) * self.gaussian_scale + self.gaussian_loc

        # Compute importance weights: Gaussian density over uniform samples
        probs = jnp.where(jnp.all((gaussian_samples >= self.uniform_low) & (gaussian_samples <= self.uniform_high), axis=-1), 1.0, 0.0)
        weights = probs / jnp.sum(probs)  # Normalize weights
        
        # gaussian_probs = jnp.exp(-0.5 * jnp.sum(((uniform_samples - self.gaussian_loc) / self.gaussian_scale) ** 2, axis=-1))
        # weights = gaussian_probs / jnp.sum(gaussian_probs)  # Normalize weights

        # Resample based on importance weights
        resampled_indices = jax.random.choice(key2, n, shape=(n,), p=weights, replace=True)
        resampled_samples = gaussian_samples[resampled_indices]

        return resampled_samples


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    # Example usage
    key = jax.random.PRNGKey(42)

    # Define 4D parameters
    uniform_low = [-5.0] * 4
    uniform_high = [2.0] * 4
    gaussian_loc = [jnp.log(0.01), jnp.log(0.5), jnp.log(1), jnp.log(0.01)]
    gaussian_scale = [0.5] * 4

    osc_prior = OscPrior(uniform_low, uniform_high, gaussian_loc, gaussian_scale)

    # Sample from the distribution
    samples = osc_prior.sample(seed=key, sample_shape=(1000,))

    plt.scatter(samples[:, 0], samples[:, 1], alpha=0.5)
    plt.scatter(gaussian_loc[0], gaussian_loc[1], color='red')
    plt.show()
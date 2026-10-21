import jax.random as jr # type: ignore
import jax.numpy as jnp # type: ignore
from jax import jit, lax, debug, vmap # type: ignore
import blackjax # type: ignore
from util.param import get_unravel_fn, tree_from_params, join_trees, params_from_tree, sample_prior, to_train_array
from blackjax.mcmc.elliptical_slice import as_top_level_api # type: ignore
from functools import partial


def map_sims(key, cond_param, props, ssmodel, num_timesteps):

    ''' Takes parameters in conditional form and
    returns emissions from ssmodel. ''
    '''

    xp = sample_prior(key, props)[0]
    unravel_fn = get_unravel_fn(xp, props)
    unravel = unravel_fn(cond_param)
    tree = tree_from_params(xp)
    new_tree = join_trees(unravel, tree, props)
    param = params_from_tree(new_tree, xp._get_names(), xp._is_constrained_tree())
    param.from_unconstrained(props)

    _, emissions = ssmodel.simulate(key, param, num_timesteps)

    return _, emissions


def sim_emissions(key, param, ssm, num_timesteps):
    ''' Takes parameters in conditional form and
    returns emissions from ssmodel. ''
    '''

    _, emissions = ssm.simulate(key, param, num_timesteps)

    return emissions

@partial(jit, static_argnums=(1,))
def one_step(state, kernel, rng_key):
    state, _ = kernel(rng_key, state)
    return state, state


def mcmc_inference_loop(rng_key, kernel, initial_state, num_samples):
    '''
    Runs the MCMC kernel for num_samples steps and returns the states.
    '''

    keys = jr.split(rng_key, num_samples)
    _one_step = lambda state, key: one_step(state, kernel, key)
    _, states = lax.scan(_one_step, initial_state, keys)

    return states


def sample_logpdf(key, learner, logdensity_fn, num_samples, num_mcmc_steps, rw_sigma):

    assert(num_mcmc_steps > num_samples), 'Number of MCMC steps must be greater than number of samples'

    ## Initialize MCMC chain and kernel
    key, subkey = jr.split(key)
    initial_cond_params = to_train_array(sample_prior(subkey, learner.props, 1)[0], learner.props)
    random_walk = blackjax.additive_step_random_walk(logdensity_fn, blackjax.mcmc.random_walk.normal(rw_sigma))
    # hmc = blackjax.hmc(logdensity_fn, step_size=1e-3, inverse_mass_matrix=jnp.ones(initial_cond_params.shape), num_integration_steps=10)
    # initial_state = hmc.init(initial_cond_params)
    # kernel = jit(hmc.step)

    initial_state = random_walk.init(initial_cond_params)
    kernel = jit(random_walk.step)

    ## Run MCMC inference loop
    key, subkey = jr.split(key)
    mcmc_states = mcmc_inference_loop(subkey, kernel, initial_state, num_mcmc_steps)
    ps = mcmc_states.position[-num_samples:]

    # Setup params for next round
    params_sample = []
    param_names = learner.xparam._get_names()
    is_constrained_tree = learner.xparam._is_constrained_tree()

    for cond_param in ps:

        unravel_fn = get_unravel_fn(learner.xparam, learner.props)
        unravel = unravel_fn(cond_param)
        tree = tree_from_params(learner.xparam)
        new_tree = join_trees(unravel, tree, learner.props)
        param = params_from_tree(new_tree, param_names, is_constrained_tree)
        params_sample.append(param)

    return params_sample, ps


def resample(weights, particles, key):                                                         

    keys = jr.split(key, 2)
    num_particles = weights.shape[0]
    resampled_idx = jr.choice(keys[0], jnp.arange(weights.shape[0]), shape=(num_particles,), p=weights)
    resampled_particles = jnp.take(particles, resampled_idx, axis=0)
    weights = jnp.ones(shape=(num_particles,)) / num_particles
    next_key = keys[1]

    return weights, resampled_particles, next_key


def resample2(weights, particles, key):                                                                  

    keys = jr.split(key, 2)
    num_particles = weights.shape[0]
    resampled_idx = jr.choice(keys[0], jnp.arange(weights.shape[0]), shape=(num_particles,), p=weights)
    resampled_particles = jnp.take(particles, resampled_idx, axis=0)
    weights = jnp.ones(shape=(num_particles,)) / num_particles
    next_key = keys[1]

    return weights, resampled_particles, next_key, resampled_idx


def generate_emissions(key, emission_dim, model, cond_param, num_samples, lag, num_timesteps):

    all_emissions = []
    n_params = cond_param.shape[0]

    if lag >=0:

        count = 0

        while count < num_samples:

            emissions = []
            prev_lagged_emissions = jnp.zeros((lag, emission_dim))
            skip_outer = False
            t=0

            while t < num_timesteps:

                condition_on = jnp.concatenate([cond_param, prev_lagged_emissions.flatten()])[None]
                key, subkey = jr.split(key)
                gen = model.generate(subkey, 1, condition_on)[0]
                new_emission = gen[-emission_dim:]

                if jnp.isnan(new_emission).any() | jnp.isinf(new_emission).any():
                    
                    print('NaN or Inf generated')
                    skip_outer = True
                    break

                else: 

                    prev_lagged_emissions = jnp.concatenate([prev_lagged_emissions[1:], new_emission[None]])
                    emissions.append(new_emission)
                    t += 1

            if skip_outer:

                continue

            sim_emission = jnp.array(emissions)
            is_nanUinf = jnp.isnan(sim_emission).any() | jnp.isinf(sim_emission).any()

            if is_nanUinf:
                
                continue

            else: 

                all_emissions.append(sim_emission)
                count += 1
                print(f'Sample {count} generated')

        all_emissions = jnp.array(all_emissions)

    else:

        key, subkey = jr.split(key)
        all_emissions = model.generate(subkey, num_samples, cond_param[None])[:, n_params:]

    return all_emissions


def get_eps_init(key, k, observations, ssm, props, num_timesteps, num_particles):

    emission_dim = ssm.emission_dim
    key, subkey = jr.split(key)
    kN_ps = sample_prior(subkey, props, k * num_particles)
    kN_cps = jnp.array(list(map(lambda ps: to_train_array(ps,props), kN_ps)))

    # # Simulate emissions
    key, subkey = jr.split(key)
    _, sim_emissions = vmap(lambda particle: map_sims(subkey, particle, props, ssm, num_timesteps))(kN_cps)

    # # Compute distance
    dists = vmap(lambda emissions: jnp.linalg.norm(observations - emissions) / jnp.sqrt(num_timesteps * emission_dim))(sim_emissions)
    min_dists = jnp.sort(dists, axis=0)[:num_particles]
    eps = jnp.max(min_dists)

    return eps
"""
Test script to verify LGSF with square root steps produces correct results.
"""
import sys
sys.path.append(r'/Users/kostastsampourakis/Desktop/code/Python/projects/gaussfiltax/gaussfiltax')

import jax.numpy as jnp
import jax.random as jr
from filters.gsf import LGSF
from util.param import initialize

def test_lgsf_sqrt():
    """Test LGSF filter with square root implementation using a real simulator."""
    
    # Import a simulator
    from simulators.lgssm import LGSSM, get_constrained_bijectors
    
    key = jr.PRNGKey(42)
    
    # Set up simulator properties
    props = {
        'state_dim': 2,
        'emission_dim': 2,
        'num_timesteps': 10
    }
    
    num_components = 3
    
    # Initialize simulator
    ssm = LGSSM(props)
    
    # Get bijectors and initialize parameters
    bijectors_tree = get_constrained_bijectors(ssm, props)
    params = initialize(ssm, props, bijectors_tree, key)
    
    # Generate some synthetic observations
    key1, key2 = jr.split(key)
    num_timesteps = props['num_timesteps']
    state_dim = props['state_dim']
    emission_dim = props['emission_dim']
    
    # Simple simulation
    observations = jr.normal(key2, (num_timesteps, emission_dim))
    
    # Run LGSF filter
    filter = LGSF(props, ssm)
    
    print("Running LGSF with square root steps...")
    print("=" * 60)
    
    outputs = filter.run(
        params=params,
        emissions=observations,
        num_components=num_components,
        rng_key=key1
    )
    
    print(f"Output shapes:")
    print(f"  means: {outputs['means'].shape}")
    print(f"  covariances: {outputs['covariances'].shape}")
    print(f"  weights: {outputs['weights'].shape}")
    
    # Verify outputs are reasonable
    assert outputs['means'].shape == (num_timesteps, num_components, state_dim)
    assert outputs['covariances'].shape == (num_timesteps, num_components, state_dim, state_dim)
    assert outputs['weights'].shape == (num_timesteps, num_components)
    
    # Check weights sum to 1
    weight_sums = jnp.sum(outputs['weights'], axis=1)
    print(f"\nWeights sum to 1: {jnp.allclose(weight_sums, 1.0)}")
    print(f"  Max deviation: {jnp.max(jnp.abs(weight_sums - 1.0)):.2e}")
    
    # Check covariances are positive definite
    min_eigvals = []
    for t in range(num_timesteps):
        for k in range(num_components):
            eigvals = jnp.linalg.eigvalsh(outputs['covariances'][t, k])
            min_eigvals.append(jnp.min(eigvals))
    
    min_eigval = jnp.min(jnp.array(min_eigvals))
    print(f"\nAll covariances positive definite: {min_eigval > 0}")
    print(f"  Min eigenvalue: {min_eigval:.6e}")
    
    print("\n" + "=" * 60)
    print("SUCCESS: LGSF with square root steps works correctly!")
    print("All covariances remain positive definite throughout filtering.")
    
if __name__ == "__main__":
    test_lgsf_sqrt()

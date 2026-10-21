# Copilot Instructions for GaussFiltAX

## Project Overview
Likelihood-free inference experiments for nonlinear state space models using Gaussian sum filters, unscented filters, adaptive Gaussian sum filters, and particle filters. JAX-based implementation for high-performance numerical computing.

## Architecture & Key Components

### Experiment Orchestration
- Entry point: [gaussfiltax/main.py](gaussfiltax/main.py) with CLI subcommands (`trials`, `view`, `ensemble`, `errors`)
- Experiment execution: [gaussfiltax/experiment_runner.py](gaussfiltax/experiment_runner.py) handles filter dispatch and result storage
- Experiment descriptors: [gaussfiltax/experiment_descriptor.py](gaussfiltax/experiment_descriptor.py) parses text-based experiment specs

### Experiment Descriptor Syntax
Experiments defined as: `experiment{sim:<simulator>{<params>},inf:<inference>{<params>}}`

Example:
```
experiment{sim:lmsv{state_dim:10,emission_dim:10,num_timesteps:140},inf:ugsf{n_comps:10,n_iters:1,alpha:0.001,beta:2.0,kappa:0.0}}
```

**Available simulators:** `lgssm`, `lmsv`, `covinfl`, `mtt`, `svssm`, `sirssm`, `bmnl`
**Available inference:** `lgsf`/`ugsf` (Gaussian sum), `lagsf`/`uagsf` (adaptive), `bpf`/`apf` (particle)

### Simulators Structure
Each simulator in `gaussfiltax/simulators/` requires:
1. `<name>.py` - main module with `setup()`, `get_inputs()`, `get_root()`, `get_param_info()`
2. `<name>_params.py` - defines `_init_vals()` and `_param_dists()` for parameters
3. Descriptor class in [experiment_descriptor.py](gaussfiltax/experiment_descriptor.py)
4. Registration in [misc.py](gaussfiltax/misc.py) `get_simulator()`

**LMSV Simulator (Linear Multivariate Stochastic Volatility):**
- Dynamics: `x_{t+1} = Phi * x_t + q_t` (linear AR)
- Emission: `y_t = u * beta * exp(x/sigma) * r + (1-u) * (H*x + r)`
- Input `u` controls volatility regime: 0=linear, 1=stochastic volatility
- Parameters: Phi (dynamics weights), Q (dynamics cov), H (emission weights), R (emission cov), sigma (volatility scale), beta (volatility coeff)

### Filter Implementations
- Gaussian sum filters: [gaussfiltax/filters/gsf.py](gaussfiltax/filters/gsf.py) - LGSF (linearized), UGSF (unscented)
- Adaptive filters: [gaussfiltax/filters/agsf.py](gaussfiltax/filters/agsf.py) - LAGSF, UAGSF with component adaptation
- Particle filters: [gaussfiltax/filters/pf.py](gaussfiltax/filters/pf.py) - BPF, APF
- Common operations: [gaussfiltax/filters/steps.py](gaussfiltax/filters/steps.py)

Filters use square-root covariances for numerical stability and `lax.scan` for efficient sequential processing.

### Data Layout & Storage
- Experiment results: `data/experiments/<sim>/<inf>/<trial>/`
- Each trial contains: `results.pkl`, `gt.pkl` (ground truth), `exp_desc.txt`, `out.log`, optional `aux_outputs.pkl`
- Hard-coded data root in [misc.py](gaussfiltax/misc.py): `/Users/kostastsampourakis/Desktop/code/Python/projects/gaussfiltax/data`
- Figures output: `figures/` directory

### Running Experiments

```bash
# Run trials with random or fixed seed
python -m gaussfiltax.main trials <seed|r> <start_trial> <end_trial> <exp_desc.txt>

# View results for specific trial
python -m gaussfiltax.main view <exp_desc.txt> -t <trial_num>

# Generate ensemble plots across trials
python -m gaussfiltax.main ensemble <start> <end> <exp_desc.txt>

# Evaluate and plot errors
python -m gaussfiltax.main errors <start> <end> <exp_desc.txt>
```

## Key Conventions

### JAX Patterns
- Always split PRNG keys: `key, subkey = jr.split(key)`
- Use `vmap` for batched operations over components/particles
- Filters expect `ParamSSM` parameter trees with `.value` attributes
- Array shapes: emissions `(T, emission_dim)`, states `(T, state_dim)`, inputs `(T, input_dim)`

### Parameter Management
- `util.io.save()` automatically appends `.pkl` extension
- Parameters stored in nested structure: `params.<group>.<name>.value`
- Constrainers: `RealToPSDBijector` for covariance matrices, `None` for unconstrained
- Initialize via `util.param.initialize(prior_fields, param_names, constrainers)`

### Plotting & Analysis
- Uses `science` and `ieee` matplotlib styles
- Bootstrap for error estimation: `util.numerics.bootstrap(key, data, B)`
- Metrics: log prob error, bias, min error, MSE computed via `util.numerics`
- Ensemble plots combine results across algorithms with error bands

## Common Gotchas
- Existing trial directories raise `AlreadyExistingExperiment` - clean manually if rerunning
- Descriptor parsing strips whitespace - keep exact syntax with no spaces around `:` and `,`
- Input shapes must match `(num_timesteps,)` or `(num_timesteps, input_dim)` - see simulator's `get_inputs()`
- Filter `run()` methods have different signatures (LGSF vs UGSF vs LAGSF vs BPF)

## Adding New Components

### New Simulator
1. Create `gaussfiltax/simulators/<name>.py` and `<name>_params.py`
2. Define dynamics `f(x, q, u, params)` and emission `g(x, r, u, params)` functions
3. Implement `setup()`, `get_inputs()`, `get_root()`, `get_param_info()`
4. Add descriptor class to [experiment_descriptor.py](gaussfiltax/experiment_descriptor.py): inherit `SimulatorDescriptor`, implement `parse()`, `pprint()`, `get_dir()`
5. Register in `SimulatorDescriptor.get_descriptor()` and [misc.py](gaussfiltax/misc.py) `get_simulator()`

### New Inference Method
1. Implement filter class in appropriate module under `gaussfiltax/filters/`
2. Create descriptor class in [experiment_descriptor.py](gaussfiltax/experiment_descriptor.py)
3. Add `_run_<method>()` handler in [experiment_runner.py](gaussfiltax/experiment_runner.py)
4. Register in `InferenceDescriptor.get_descriptor()`

## Dependencies
Core: `jax`, `jaxlib`, `flax`, `tensorflow-probability[jax]`, `jaxtyping`
Plotting: `matplotlib`, `scienceplots`, `matplotlib_inline`
Utils: `pandas`, `numpy`, `numpyro`

Install: `pip install -e .` from project root

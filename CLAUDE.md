# CLAUDE.md

This file gives guidance to Claude Code (claude.ai/code) for work in this repository.

## Project Overview

**gaussfiltax** is a research framework for state estimation (filtering) in nonlinear state space models (SSMs). It compares six filters on four simulators. All computations use JAX.

| Short name | Filter |
|---|---|
| L-GSF, U-GSF | Gaussian sum filter, with linearized (L) or unscented (U) components |
| L-AGSF, U-AGSF | Adaptive Gaussian sum filter, with linearized (L) or unscented (U) components |
| BPF | Bootstrap particle filter |
| APF | Auxiliary particle filter |

The simulators are `lgssm`, `lmsv`, `covinfl`, and `mtt`. The filters run with the true model parameters. The parameters are fixed values, not random draws. Only the simulated states and observations change from trial to trial.

## Working rules

These rules apply to every task.

1. **Use the code that exists.** Before you write new code, search the repository for code that does the job. The filters, simulators, metrics, descriptors, plot classes, and I/O helpers are all implemented. Read them and call them. Do not write a second version.
2. **Fix old code that fails.** Some modules are from an older layout (see "Known stale code"). If code that you need raises an error, find the cause and correct the code in place. Do not copy it, do not work around it, and do not delete it. Keep the fix small. Tell the user what you changed.
3. **Run experiments only through `main.py`.** Describe the experiment in a descriptor file. Run it with `main.py trials`. Do not write a new script, notebook, or loop that runs a filter for a reported result. `standalone_filter_runner.py` is for quick debugging only.
4. **Report results only with the implemented metrics.** Use `main.py errors` and `main.py ensemble`. These use the functions in `util/numerics.py`. Do not define a new error measure in a script. If you need a new metric, add it to `util/numerics.py` with the same interface as the others (see "Metrics") and connect it to `main.py`.

## Commands

Run all commands from the repository root, in the conda environment `jaxenv` (`conda activate jaxenv`, JAX 0.5.2).

```bash
# Run trials 1 to 10 of every experiment in the descriptor file(s).
# The seed is an integer (fixed) or 'r' (random, from the clock).
python gaussfiltax/main.py trials r 1 10 "exps/lmsv/state_dim_4/lmsv_lgsf_scan_num_prt.txt"

# Compute and save the metrics of trials 1 to 10, with histograms. Run this before 'ensemble'.
python gaussfiltax/main.py errors 1 10 <descriptor files>

# Plot the metrics against the number of simulations, one line for each algorithm
python gaussfiltax/main.py ensemble 1 10 <descriptor files>

# Write new descriptor files (edit the values in the __main__ block first)
python gaussfiltax/create_exps.py

# Run trials, and start again after a crash, until all trials finish
./run_nocrash.zsh 1 10 "<one descriptor file>"

# Tests that run without pytest
python tests/test_pf_vs_kalman.py
python tests/test_numerics_weights.py
python test_sqrt_filter.py
```

- Use `python gaussfiltax/main.py`. The command `python -m gaussfiltax.main` fails with `No module named 'util'` because the modules use bare imports.
- Put quotes around file names. Some descriptor files have spaces and brackets in the name.
- Trial numbers start at 1. The value 0 raises `ValueError`.
- `tests/test_apf.py` needs pytest, which `jaxenv` does not have.
- `./gaussfilt_tmux.zsh` makes a tmux session `gaussfilt_exps` with four panes. Each pane has `jaxenv` active. Start long jobs there.
- `.github/copilot-instructions.md` is older. If it differs from this file, this file is correct.

## Architecture

### Experiment lifecycle

```
create_exps.py → exps/**/*.txt → main.py trials → experiment_runner.py → data.nosync/experiments/...
                                                                                  ↓
                         figures/ ← main.py ensemble ← main.py errors (metrics from util/numerics.py)
```

1. **Descriptor** (`experiment_descriptor.py`): Text configurations that regular expressions parse into typed objects. The descriptor gives the storage folder through `get_dir()`.
2. **Runner** (`experiment_runner.py`, class `ExperimentRunner`): Builds the simulator with `misc.get_simulator`, simulates states and observations, runs the filter, and saves the results. Each filter has a method `_run_<name>`.
3. **Main** (`main.py`): The command line with four subcommands: `trials`, `errors`, `ensemble`, `view`.
4. **Results** (`experiment_results.py`): Containers with the plot name, marker, and color of each algorithm.
5. **Create Exps** (`create_exps.py`): The function `create_descs(sim_desc, alg, scan_params)` writes descriptors for all combinations of the values in `scan_params`. The default values are in `DEFAULT_SCAN_PARAMS`. The hyperparameters live in this code, not in other configuration files.

### Descriptor syntax

A descriptor file holds one or more `experiment` blocks. The parser removes all white space. The keys must be in the order that this section shows.

```
experiment
{
    sim: lmsv
    {
        state_dim: 4,
        emission_dim: 4,
        num_timesteps: 200
    },

    inf: ugsf
    {
        n_comps: 10,
        n_iters: 1,
        alpha: 1.0,
        beta: 0.0,
        kappa: 0.0
    }
}
```

| `sim:` | Keys, in order |
|---|---|
| `lgssm`, `lmsv`, `mtt` | `state_dim`, `emission_dim`, `num_timesteps` |
| `covinfl` | `emission_dim`, `num_timesteps`, `a`, `b` (the state dimension is always 1) |

| `inf:` | Keys, in order |
|---|---|
| `lgsf` | `n_comps`, `n_iters` |
| `ugsf` | `n_comps`, `n_iters`, `alpha`, `beta`, `kappa` |
| `lagsf` | `M`, `N`, `L`, `n_iters`, `ac1`, `r1`, `ac2`, `r2` |
| `uagsf` | `M`, `N`, `L`, `n_iters`, `alpha`, `beta`, `kappa`, `ac1`, `r1`, `ac2`, `r2` |
| `bpf`, `apf` | `num_prt`, `ess` |

- `n_comps`: number of Gaussian components. `n_iters`: number of relinearizations in the update step. `alpha`, `beta`, `kappa`: parameters of the unscented transform.
- AGSF: `M` components stay after each step. Each component splits into `N` points before the prediction and `L` points before the update. The filter then resamples the `M·N·L` candidates down to `M`. `ac1` is the split rule for the prediction and `ac2` is the rule for the update. The value is `prop` or `auto`. `r1` and `r2` set the covariance of the new components as a fraction of the covariance P (`prop`) or as the scale of the automatic rule (`auto`).
- BPF and APF: `num_prt` particles. The filter resamples when the effective sample size is below `ess × num_prt`.

### Simulators (`simulators/`)

| Name | Model |
|---|---|
| `lgssm` | Linear Gaussian SSM |
| `lmsv` | Linear dynamics `x' = Phi x + q`. The input `u` mixes a linear emission (`u = 0`) with a multiplicative stochastic-volatility emission (`u = 1`). The default input switches every 20 steps. |
| `covinfl` | Scalar state with dynamics `sin(c x) + q` and a quadratic emission |
| `mtt` | Maneuvering target tracking. The state is position and velocity in 2D. The input (0, 1, or 2) selects straight motion or a turn in one of two directions. The emission is bearing and range. |

Each simulator module has these functions:
- `setup(...)` returns a dictionary with `ssm`, `props` (parameter properties and values), and `exp_info`. The `ssm` is a `NonlinearSSM` or `LGSSM` from `simulators/ssm.py`.
- `get_inputs(key, num_timesteps)` returns the input sequence.
- `get_param_info()` returns the module `<name>_params.py`. This module has `_init_vals` and `_param_dists`.

The dynamics function is `f(x, q, u, params)`. The emission function is `g(x, r, u, params)`. The particle filters also need `ssm.emission_log_prob`. The module `lmsv` sets it explicitly.

To add a simulator:
1. Make `simulators/<name>.py` and `simulators/<name>_params.py`.
2. Add a descriptor class in `experiment_descriptor.py` with `parse`, `pprint`, `create_desc`, and `get_dir`. Register it in `SimulatorDescriptor.get_descriptor`.
3. Register the module in `misc.get_simulator`.
4. If `setup` needs more arguments than `(state_dim, emission_dim, input_dim)`, add a branch in `ExperimentRunner.run`, as for `covinfl`.

### Filters (`filters/`)

The filters use square-root covariances and `lax.scan` over time. The `run` methods have different signatures. Copy the call from the matching `_run_<name>` method in `experiment_runner.py`.

| Class | Module | Saved `results` |
|---|---|---|
| `LGSF`, `UGSF` | `gsf.py` | Dictionary with `means` (K, T, d), `covariances` (K, T, d, d), `weights` (K, T), and the predicted values |
| `LAGSF`, `UAGSF` | `agsf.py` | Same dictionary without the predicted values. A second output, `aux_outputs`, holds `Deltas`, `Lambdas`, `rho1s`, `rho2s`. |
| `BPF`, `APF` | `pf.py` | Tuple of a dictionary and the log likelihood. The dictionary has `particles` (N, T, d), `weights` (N, T), and `u_weights`. |

`steps.py` holds the Kalman and unscented prediction and update steps. The filters share them.

To add a filter, change all of these places:
1. The filter class in `filters/`.
2. A descriptor class in `experiment_descriptor.py`, registered in `InferenceDescriptor.get_descriptor`.
3. A method `_run_<name>` and a branch in `ExperimentRunner.run`.
4. A branch in `get_means_covs` and `get_num_sims` in `util/numerics.py`.
5. A results class in `experiment_results.py` and a branch in `main.view_ensemble`.

### Parameters

`ParamSSM` (in `parameters.py`) is a JAX pytree with three groups: `initial`, `dynamics`, `emissions`. Read a value as `params.dynamics.cov.value`. `util/param.py` builds the parameters (`initialize`) and samples them (`sample_prior`). Covariance matrices use the bijector `RealToPSDBijector` (`util/bijectors.py`).

Shapes: emissions `(T, emission_dim)`, states `(T, state_dim)`, inputs `(T,)` or `(T, input_dim)`.

### Storage

The root is `misc.get_root()`, which is `data.nosync/` in the repository. The folder does not exist on a fresh clone. The runner creates it. Results are not in git.

```
data.nosync/experiments/<sim.get_dir()>/<inf.get_dir()>/<trial>/
e.g. data.nosync/experiments/lmsv/state-dim_4_emission-dim_4_num-timesteps_200/gsf/ugsf/ncomps_10_iters_1_alpha_1.0_beta_0.0_kappa_0.0/1/
```

The names encode every descriptor field. If you change one field, the folder changes.

| File | Content | Written by |
|---|---|---|
| `results.pkl`, `gt.pkl` | Filter output. Ground truth: `(states, observations, true_params)`. | `trials` |
| `exp_desc.txt`, `out.log`, `runtime.txt`, `observations.png` | Descriptor, log, run time, plot of the simulated data | `trials` |
| `aux_outputs.pkl` | `aux_outputs` of the AGSF | `trials` |
| `lpe.pkl`, `bs.pkl`, `me.pkl`, `mse.pkl`, `num_sims.pkl` | Metrics of the trial (see "Metrics") | `errors` |
| `Error histograms.png`, `Errors vs Trials.png` | Plots over all trials. These are in the experiment folder, above the trial folders. | `errors` |

`ensemble` writes `figures/ensemble_<sim>_.pdf`. Use `util.io.save(data, path)` and `util.io.load(path)`. Both add `.pkl` to the path.

When `results.pkl` exists, `trials` skips the trial. If the trial folder exists but `results.pkl` does not (a crash), `trials` deletes the files and runs the trial again. To redo a finished trial, delete its folder by hand.

## Metrics

All metrics are in `util/numerics.py`. Each takes `(data, inf_name, true_states)`, where `data` is the saved `results` and `inf_name` is `inf_desc.name`. Each returns `(mean over time, array over time)`. The functions `get_means_covs` and `get_weights` read the output of each filter. For BPF and APF, `get_means_covs` builds a mixture with one component for each particle.

| Function | File | Meaning (lower is better) |
|---|---|---|
| `log_prob_error` | `lpe` | Negative log density of the true state under the filter mixture |
| `bias` | `bs` | Norm of the error of the weighted mean estimate |
| `min_error` | `me` | Norm of the error of the best single component mean |
| `mse` | `mse` | Squared norm of the error of the weighted mean estimate |
| `get_num_sims(exp_desc)` | `num_sims` | Number of simulations that the filter uses. This is the cost axis of the plots. |

`bootstrap(key, array, B)` returns `B` bootstrap means over trials. `main.py` uses `B = 100`. The plots show the mean and the standard deviation of these means.

## Key conventions

**Imports**: The code uses bare imports (`import util.io`, `from parameters import ParamSSM`, `from filters import gsf`). Run scripts that are in `gaussfiltax/` directly, or put `gaussfiltax/` on `sys.path`. Several modules append the hardcoded path to `sys.path`. `pip install -e .` does not make the bare imports work.

**RNG**: JAX PRNG keys are passed explicitly (`key, subkey = jr.split(key)`). `main.py trials` splits one key in sequence over the descriptors and trials. Two algorithms with the same trial number therefore see different simulated data. A repeat run gives the same data only with the same seed, file order, and set of existing trials. Compare algorithms through the average over many trials, not trial by trial.

**Memory cleanup**: After each trial, `jax.clear_caches()` and `gc.collect()` run. This is intentional. Keep it.

**Plotting style**: `scienceplots` with `['science', 'ieee']`. Use the classes in `experiment_results.py` for the name, marker, and color: L-GSF blue circle, U-GSF green square, L-AGSF red diamond, U-AGSF magenta cross, BPF cyan triangle, APF yellow triangle.

**NaN trials**: `main.py ensemble` removes NaN and inf values before the bootstrap. `main.py errors` computes the bootstrap mean and standard deviation first, then removes them for the histograms. A NaN trial makes the mean line in the `errors` plots NaN. Do not ignore a NaN trial. Find the cause.

**Git**: Python caches (`__pycache__`) are tracked in git. Do not stage them.

## Known stale code

The experiment code came from the `neuralssm` project and from older versions of this repository. These parts do not match the current layout. Rule 2 applies: correct them when you need them.

- **`main.py view` and `experiment_viewer.py`**: Copied from `neuralssm`. They read `info.txt` (the runner writes `exp_desc.txt`) and use descriptor classes that do not exist (`ed.ABC_Descriptor`, `ed.SNL_Descriptor`, and others). The default trial is 0, which is not valid. Use `errors` for plots until you fix the viewer.
- **`simulators/lgssm.py`**: It has no `get_inputs`, but the runner calls `sim.get_inputs`. An `lgssm` experiment raises `AttributeError`.
- **`gaussfilt.py`, `gausssumfilt.py`, `particlefilt.py`, `static_utils.py`, `mean_estimators.py`**: Older API. They import `gaussfiltax.utils` or `utils`, which moved to `util/`. The experiment code does not use them.
- **`ssm.py`, `models.py`, `_parameters.py`, `simulation.py`, `nonlinearities.py`** in `gaussfiltax/`: Older API. The experiments use `simulators/ssm.py` and `parameters.py`.
- **`util/numerics.py`**: `bootstrap` is defined twice. The second definition (the one in use) returns a vector of `B` means. `sdev` uses names that are not defined. It imports `jax.tree_map`, which is deprecated (use `jax.tree_util.tree_map`). `util/ml.py` imports `theano`. `misc.find_epsilon` is from `neuralssm` and is not used.
- **`test_lgsf_sqrt.py`** (repository root): It fails with `ImportError`, because `simulators/lgssm.py` has no `get_constrained_bijectors`.
- **`docs/`, `notebooks/`, `ALA/`, `build/`, `dist/`**: Older exploratory code. The API there can differ from the current one.
- **`exps/apf_study/`**: One-off scripts for one study (see its `README.md`). Do not use them as a template for new experiments.

## Hardcoded paths

These paths contain `/Users/kostastsampourakis/` and must change on another machine:
- `misc.get_root()`
- `fig_dir` in `main.py`
- `main_dir` in `experiment_descriptor.py`
- `exp_root` in the `__main__` block of `create_exps.py`
- The `sys.path.append(...)` lines in `simulators/`, and in `standalone_filter_runner.py`
- `WORKDIR` in `gaussfilt_tmux.zsh`

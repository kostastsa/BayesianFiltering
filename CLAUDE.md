# CLAUDE.md

This file gives guidance to Claude Code (claude.ai/code) for work in this repository.

## Project Overview

**gaussfiltax** is a research framework for state estimation (filtering) in nonlinear state space models (SSMs). It compares eight filters on four simulators. All computations use JAX.

| Short name | Filter |
|---|---|
| L-GSF, U-GSF | Gaussian sum filter, with linearized (L) or unscented (U) components |
| L-AGSF, U-AGSF | Adaptive Gaussian sum filter, with linearized (L) or unscented (U) components |
| BPF | Bootstrap particle filter |
| APF | Auxiliary particle filter |
| RPF | Regularized particle filter (Gaussian kernel after each resampling) |
| UPF | Unscented particle filter (UKF proposal over the process noise) |

The simulators are `lgssm`, `lmsv`, `covinfl`, and `mtt`. The filters run with the true model parameters. The parameters are fixed values, not random draws. Only the simulated states and observations change from trial to trial.

## Working rules

These rules apply to every task.

1. **Use the code that exists.** Before you write new code, search the repository for code that does the job. The filters, simulators, metrics, descriptors, plot classes, and I/O helpers are all implemented. Read them and call them. Do not write a second version.
2. **Fix old code that fails.** Some modules are from an older layout (see "Known stale code"). If code that you need raises an error, find the cause and correct the code in place. Do not copy it, do not work around it, and do not delete it. Keep the fix small. Tell the user what you changed.
3. **Run experiments only through `main.py`.** Describe the experiment in a descriptor file. Run it with `main.py trials`. Do not write a new script, notebook, or loop that runs a filter for a reported result. `standalone_filter_runner.py` is for quick debugging only.
4. **Report results only with the implemented metrics.** Use `main.py errors` and `main.py ensemble`. These use the functions in `util/numerics.py`. Do not define a new error measure in a script. If you need a new metric, add it to `util/numerics.py` with the same interface as the others (see "Metrics") and connect it to `main.py`.
5. **Put new code in one of two places.**
   - New experiment code (a filter, a simulator, a metric, a table, a plot): add it to the package `gaussfiltax/` and call it through `main.py`.
   - A custom script for a side question (a check, a diagnostic, a comparison with an external library): put it in a new directory `scratch/<topic>/`. Do not put it in `exps/`. Do not report its output as an experiment result.

## Commands

Run all commands from the repository root, in the conda environment `jaxenv` (`conda activate jaxenv`, JAX 0.5.2).

```bash
# Run trials 1 to 10 of every experiment in the descriptor file(s).
# The seed is an integer (fixed) or 'r' (random, from the clock).
python gaussfiltax/main.py trials r 1 10 "exps/lmsv/state_dim_4/lmsv_lgsf_scan_num_prt.txt"

# Compute and save the metrics of trials 1 to 10, with histograms. Run this before 'ensemble'.
# The option --no-plots skips the histograms. The summary file is written in both cases (see "Storage").
python gaussfiltax/main.py errors 1 10 <descriptor files>

# Plot the metrics against the number of simulations, one line for each algorithm
python gaussfiltax/main.py ensemble 1 10 <descriptor files>

# Write new descriptor files (edit the values in the __main__ block first)
python gaussfiltax/create_exps.py

# Run trials, and start again after a crash, until all trials finish
./run_nocrash.zsh 1 10 "<one descriptor file>"

# Exp. A, BPF and APF rows of Tables I and III of the paper (see "Large runs")
./run_tables_I_III.zsh 1 30 0.5      # Table I, a = 0.5

# Exp. A, GSF and AGSF rows of Tables I and III (task T1, 20 trials)
SEED_BASE=2000 ./run_tables_I_III.zsh 1 20 0.5 lgsf_table,ugsf_table,lagsf_table,uagsf_table

# Exp. A, RPF and UPF rows of Tables I and III (task A1, 30 trials)
SEED_BASE=3000 ./run_tables_I_III.zsh 1 30 0.5 rpf,upf

# Since D-30 (01-Nov-2026) the tables use the wrapped bearing: the same commands with the '*_wrap' files
SEED_BASE=2000 ./run_tables_I_III.zsh 1 20 0.5 lgsf_table_wrap,ugsf_table_wrap,lagsf_table_wrap,uagsf_table_wrap
SEED_BASE=1000 ./run_tables_I_III.zsh 1 30 0.5 bpf_wrap,apf_wrap
SEED_BASE=3000 ./run_tables_I_III.zsh 1 30 0.5 rpf_wrap,upf_wrap

# Markdown rows from the Error summary.txt files (run 'errors' first)
python gaussfiltax/main.py table <descriptor files> [-o out.md]

# Exp. B, sweep of a fixed rho2 (task A6), on the data of the adaptive Table II rows
SIM=lmsv SEED_BASE=7000 ./run_tables_I_III.zsh 1 100 B lagsf_rho2sweep
SIM=lmsv SEED_BASE=7100 ./run_tables_I_III.zsh 1 100 B uagsf_rho2sweep

# R runs of the BPF or APF on each of the 30 stored data sets (data block, see "Descriptor syntax"), for the check against
# the particles library in scratch/particles_check. Give each call its own SEED_BASE. Write sigma2 as in the folder names.
SEED_BASE=5000 ./run_samedata.zsh 20 0.5 0.025,2.5,2.5e-05 apf

# Tests that run without pytest
python tests/test_pf_vs_kalman.py
python tests/test_numerics_weights.py
python tests/test_gsf_weights.py
python tests/test_wrapped_bearing.py
python tests/test_nees.py
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
| `mtt` with two more keys | `state_dim`, `emission_dim`, `num_timesteps`, `a`, `sigma2` |
| `mtt` with the wrapped bearing | `state_dim`, `emission_dim`, `num_timesteps`, `a`, `sigma2`, `wrap` |
| `covinfl` | `emission_dim`, `num_timesteps`, `a`, `b` (the state dimension is always 1) |

| `inf:` | Keys, in order |
|---|---|
| `lgsf` | `n_comps`, `n_iters` |
| `ugsf` | `n_comps`, `n_iters`, `alpha`, `beta`, `kappa` |
| `lagsf` | `M`, `N`, `L`, `n_iters`, `ac1`, `r1`, `ac2`, `r2` |
| `uagsf` | `M`, `N`, `L`, `n_iters`, `alpha`, `beta`, `kappa`, `ac1`, `r1`, `ac2`, `r2` |
| `bpf`, `apf`, `rpf` | `num_prt`, `ess` |
| `upf` | `num_prt`, `ess`, `alpha`, `beta`, `kappa` |

- `mtt`: the keys `a` (turn parameter, default 0.05) and `sigma2` (variance of the observation noise, default 2.5) are optional. Write both or none. Each pair has its own results folder.
- `mtt`, `wrap: 1` (only after `a` and `sigma2`): the bearing is an angle (decision D-30). Tables I and III use this key. The simulator wraps the noisy bearing into [-pi, pi), and all filters use the residual `[wrap(y1 - y1_hat), y2 - y2_hat]`. The results folder ends in `_wrap`. Without the key, the bearing is the raw value of `arctan2` plus noise, with a jump of 2 pi on the negative x1-axis.
- `n_comps`: number of Gaussian components. `n_iters`: number of relinearizations in the update step. `alpha`, `beta`, `kappa`: parameters of the unscented transform.
- AGSF: `M` components stay after each step. Each component splits into `N` points before the prediction and `L` points before the update. The filter then resamples the `M·N·L` candidates down to `M`. `ac1` is the split rule for the prediction and `ac2` is the rule for the update. The value is `prop` or `auto`. `r1` and `r2` set the covariance of the new components as a fraction of the covariance P (`prop`) or as the scale of the automatic rule (`auto`).
- BPF, APF, RPF and UPF: `num_prt` particles. The filter resamples when the effective sample size is below `ess × num_prt`.
- RPF: after each resampling, each particle moves by `h D e`. `D` is the square root of the weighted covariance before the resampling, `e` is standard normal, and `h = (4 / (N (d + 2)))^(1 / (d + 4))` (decision D-29).
- UPF: for each particle, one square-root UKF update over the process noise `q` (prior `N(q0, Q)`) gives the proposal. The weight is `p(y|x) N(q; q0, Q) / N(q; m_q, P_q)`. Thus the UPF works when the noise enters through a singular matrix, as in `mtt` (decision D-26).
- Optional `data` block after `inf` (particle filters, L-AGSF and U-AGSF): the filter runs on the stored data of another experiment with the same `sim` block. The data is not simulated again. Only the filter key changes. Trial t uses the data of trial ((t - 1) mod `trials`) + 1 of the folder `from`. Thus trials 1 to R × `trials` are R runs on each data set. The results go to `<sim folder>/data_<from with _ for />/<inf folder>`. `create_exps.py` writes these files as `*_samedata_*` (particle filters) and `lmsv_*_rho2sweep_*` (AGSF, task A6).

```
    data: data
    {
        from: pf/apf/numprt_1000_ess_0.5,
        trials: 30
    }
```

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

`NonlinearSSM.emission_residual` is optional (default `None`, which means `y - y_hat`). Set it when an observation component is an angle. The likelihood `emission_log_prob` and the EKF and UKF update steps (`_sqrt_condition_on`, `_ukf_sqrt_condition_on_nonadditive`) then use it. With `None`, these functions use the old code path. Test: `tests/test_wrapped_bearing.py`.

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
| `BPF`, `APF`, `RPF`, `UPF` | `pf.py` | Tuple of a dictionary and the log likelihood. The dictionary has `particles` (N, T, d), `weights` (N, T), and `u_weights`. |

`steps.py` holds the Kalman and unscented prediction and update steps. The filters share them.

To add a filter, change all of these places:
1. The filter class in `filters/`.
2. A descriptor class in `experiment_descriptor.py`, registered in `InferenceDescriptor.get_descriptor`.
3. A method `_run_<name>` and a branch in `ExperimentRunner.run`.
4. A branch in `get_num_sims` in `util/numerics.py`. For a particle filter, add the name to `PF_NAMES`. For a mixture filter, add a branch in `get_means_covs`.
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
| `lpe.pkl`, `bs.pkl`, `me.pkl`, `mse.pkl`, `nees.pkl`, `num_sims.pkl` | Metrics of the trial (see "Metrics") | `errors` |
| `Error histograms.png`, `Errors vs Trials.png` | Plots over all trials. These are in the experiment folder, above the trial folders. | `errors` (not with `--no-plots`) |
| `Error summary.txt` | Number of trials, number of non-finite trials, and number of trials with MSE above 10. The bootstrap mean, standard deviation and median of each metric. The number of trials with more than 20 % of the steps outside the 99.9 % region (NEES). This file is in the experiment folder, above the trial folders. Use it for the tables of the paper. | `errors` |

`ensemble` writes `figures/ensemble_<sim>_.pdf`. Use `util.io.save(data, path)` and `util.io.load(path)`. Both add `.pkl` to the path.

When `results.pkl` exists, `trials` skips the trial. If the trial folder exists but `results.pkl` does not (a crash), `trials` deletes the files and runs the trial again. To redo a finished trial, delete its folder by hand.

### Large runs

One `results.pkl` of a BPF or APF run with 100000 particles and 500 steps has 1.2 GB. The runs with 10000 particles have 115 MB. Do not keep all of them.

- `run_tables_I_III.zsh FIRST LAST A [NAMES]` runs one trial number at a time: `trials` (seed `SEED_BASE` + trial number, default 1000), `errors --no-plots`, and then it deletes `results.pkl` of the particle filter runs with 10000 and 100000 particles. It deletes only the runs of its own descriptor files. Thus several calls can run at the same time. The metrics, `gt.pkl` and the other small files stay.
- If `results.pkl` is not there, `errors` reads the saved `lpe.pkl`, `bs.pkl`, `me.pkl`, `mse.pkl` and `nees.pkl` of the trial. A new metric needs `results.pkl`. Thus, for the runs with 10000 and 100000 particles, run the trials again with the same `SEED_BASE` (the same keys give the same results). Run `errors FIRST LAST <files>` at the end to write `Error summary.txt` for all trials.
- Do not run `trials` again for a trial that has no `results.pkl`. It deletes the trial folder and runs the trial again.
- `main.py trials` splits the key one time for each trial, also for a trial that it skips. Thus a trial gets the same key after a restart with the same seed. (Before 30-Oct-2026, the key was split only for trials that ran.)
- `main.py trials` splits the key again for each descriptor. Thus the BPF and APF runs of one trial number have different data. Use a data block to run a filter on the same data.
- A different seed for each trial number gives different data for each trial. Use a new seed base (environment variable `SEED_BASE`) if you add trials to a finished set. Otherwise the new trials repeat the data of the old trials.

## Metrics

All metrics are in `util/numerics.py`. Each takes `(data, inf_name, true_states)`, where `data` is the saved `results` and `inf_name` is `inf_desc.name`. Each returns `(mean over time, array over time)`. The functions `get_means_covs` and `get_weights` read the output of each filter. For BPF and APF, `get_means_covs` builds a mixture with one component for each particle.

| Function | File | Meaning (lower is better) |
|---|---|---|
| `log_prob_error` | `lpe` | Negative log density of the true state under the filter mixture |
| `bias` | `bs` | Norm of the error of the weighted mean estimate |
| `min_error` | `me` | Norm of the error of the best single component mean |
| `mse` | `mse` | Squared norm of the error of the weighted mean estimate |
| `nees` | `nees` | Fraction of the steps with NEES above the 0.999 quantile of chi2 (d degrees of freedom). The NEES uses the mean and covariance of the whole posterior, so it needs no kernel for the particle filters. A high value means that the filter is confident in a region where the target is not. |
| `get_num_sims(exp_desc)` | `num_sims` | Number of simulations that the filter uses. This is the cost axis of the plots. |

`bootstrap(key, array, B)` returns `B` bootstrap means over trials. `main.py` uses `B = 100`. The plots show the mean and the standard deviation of these means. Since 02-Nov-2026 the bootstrap key of each experiment is fixed: `BOOTSTRAP_SEED` (in `main.py`) and a hash of the results folder. Thus `errors` and `ensemble` give the same numbers at each run, in any order of the descriptor files.

## Key conventions

**Imports**: The code uses bare imports (`import util.io`, `from parameters import ParamSSM`, `from filters import gsf`). Run scripts that are in `gaussfiltax/` directly, or put `gaussfiltax/` on `sys.path`. Several modules append the hardcoded path to `sys.path`. `pip install -e .` does not make the bare imports work.

**RNG**: JAX PRNG keys are passed explicitly (`key, subkey = jr.split(key)`). `main.py trials` splits one key in sequence over the descriptors and trials. Two algorithms with the same trial number therefore see different simulated data. A repeat run gives the same data only with the same seed, file order, and set of existing trials. Compare algorithms through the average over many trials, not trial by trial.

**Memory cleanup**: After each trial, `jax.clear_caches()` and `gc.collect()` run. This is intentional. Keep it.

**Plotting style**: `scienceplots` with `['science', 'ieee']`. Use the classes in `experiment_results.py` for the name, marker, and color of each algorithm.
- GSF and AGSF: L-GSF blue circle, U-GSF green square, L-AGSF red diamond, U-AGSF magenta cross.
- Particle filters: BPF cyan triangle, APF yellow triangle, RPF black left triangle, UPF orange right triangle.

**NaN trials**: `main.py ensemble` removes NaN and inf values before the bootstrap. `main.py errors` computes the bootstrap mean and standard deviation first, then removes them for the histograms. The count of non-finite trials is in `Error summary.txt`. A NaN trial makes the mean line in the `errors` plots NaN. Do not ignore a NaN trial. Find the cause.

**Git**: Python caches (`__pycache__`) are tracked in git. Do not stage them.

## Known stale code

The experiment code came from the `neuralssm` project and from older versions of this repository. These parts do not match the current layout. Rule 2 applies: correct them when you need them.

- **`main.py view` and `experiment_viewer.py`**: Copied from `neuralssm`. They read `info.txt` (the runner writes `exp_desc.txt`) and use descriptor classes that do not exist (`ed.ABC_Descriptor`, `ed.SNL_Descriptor`, and others). The default trial is 0, which is not valid. Use `errors` for plots until you fix the viewer.
- **`simulators/lgssm.py`**: It has no `get_inputs`, but the runner calls `sim.get_inputs`. An `lgssm` experiment raises `AttributeError`.
- **`gaussfilt.py`, `gausssumfilt.py`, `particlefilt.py`, `static_utils.py`, `mean_estimators.py`**: Older API. They import `gaussfiltax.utils` or `utils`, which moved to `util/`. The experiment code does not use them.
- **`ssm.py`, `models.py`, `_parameters.py`, `simulation.py`, `nonlinearities.py`** in `gaussfiltax/`: Older API. The experiments use `simulators/ssm.py` and `parameters.py`.
- **`util/numerics.py`**: `bootstrap` is defined twice. The second definition (the one in use) returns a vector of `B` means. `sdev` uses names that are not defined. It imports `jax.tree_map`, which is deprecated (use `jax.tree_util.tree_map`). `util/ml.py` imports `theano`. `misc.find_epsilon` is from `neuralssm` and is not used.
- **`runtime.txt`**: Since 01-Nov-2026 the `_run_*` methods call `jax.block_until_ready` before the time measurement (task K1). Runs before this date measured only the dispatch of the call. The value also includes the JIT compile time, because the runner clears the caches after each trial. To compare run times (task A3), measure the compile time separately or use a warm-up call.
- **`test_lgsf_sqrt.py`** (repository root): It fails with `ImportError`, because `simulators/lgssm.py` has no `get_constrained_bijectors`.
- **`docs/`, `notebooks/`, `ALA/`, `build/`, `dist/`**: Older exploratory code. The API there can differ from the current one.
- **`scratch/apf_study/`**: Scratch scripts of the APF study of 21-Oct-2026 (see its `README.md`). They are checks and diagnostics, and some use the `particles` library. Do not use them as a template for new experiments.

## Hardcoded paths

These paths contain `/Users/kostastsampourakis/` and must change on another machine:
- `misc.get_root()`
- `fig_dir` in `main.py`
- `main_dir` in `experiment_descriptor.py`
- `exp_root` in the `__main__` block of `create_exps.py`
- The `sys.path.append(...)` lines in `simulators/`, and in `standalone_filter_runner.py`
- `WORKDIR` in `gaussfilt_tmux.zsh`

## Simplified Technical English (ASD-STE100, basic rules)

Write technical documentation and instructions (procedures, READMEs, warnings, how-to text) in STE. Chat replies and code comments are not covered unless I ask. Spec detail is in the `ste` skill: load it only to check one rule, look up a word, or review a text in full.

- **Words**: use approved dictionary words, each with one approved meaning and part of speech. Technical nouns and technical verbs of the subject field are allowed. Do not use a technical noun as a verb, or a technical verb as a noun. Use one term for one item. American spelling. Examples: start (not begin), use (not utilize), make (not manufacture), keep (not maintain).
- **Noun strings**: maximum three words. Write a longer technical noun in full.
- **Verbs**: only infinitive, imperative, simple present, simple past, simple future, and past participle as an adjective. No perfect or progressive tenses, no auxiliary constructions. "-ing" only in technical nouns. Use the active voice (passive only in descriptions when the agent is unknown). Use a verb for an action, not a noun. No phrasal verbs.
- **Sentences**: short and clear. No contractions. Do not omit words or articles. Use "the", "a", "an", "this", "these" before nouns. Use a vertical list for long or complex items. Connect related sentences with connecting words.
- **Procedures**: maximum 20 words per sentence. One instruction per sentence (two only if the actions occur at the same time). Imperative form. Put a condition first, then a comma, then the command. Notes give information, not instructions.
- **Descriptions**: maximum 25 words per sentence. One topic per paragraph. Maximum six sentences per paragraph. Give information gradually, with key words and phrases for structure.
- **Safety**: WARNING for risk of injury or death, CAUTION for risk of damage (WARNING if both). Start with a clear command or condition, then explain the risk or result.
- **Punctuation**: no semicolons. Hyphens only to join directly related words. Parentheses only for references, step or item identifiers, abbreviations, singular/plural, explanations, alternatives. A number, a number with its unit, an abbreviation, an identifier, quoted text, a label, or a proper noun counts as one word. Text in parentheses counts as one word.

## Tracking: tasks and decisions

Tracker: /Users/kostastsampourakis/Desktop/TSP Resubmission review (owner: exps)

The project has a tracker (the `track` skill): `TODO.md` (tasks, read **Now** first), `DECISIONS.md` (decisions, read **Open decisions** first) and `MAP.md` (reviewer points). The tracker is in the folder of the `Tracker:` line, not in this repository. Write all tracker text in STE and run the STE checker after each change. Change only the rows of this session (owner `exps`). Record each decision in the same turn in which the user decides.

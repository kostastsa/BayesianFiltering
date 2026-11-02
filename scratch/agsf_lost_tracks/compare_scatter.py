"""Comparison scatter of a T1 trial in which the L-AGSF loses the track.

Debug view only, not an experiment result (CLAUDE.md, rule 5).
Exp. A, a = 0.5, sigma2 = 2.5. The script loads gt.pkl of the trials of the
L-AGSF (M = 100) folder and runs the filters on these data with the
FilterRunner of standalone_filter_runner.py. The model is simulators/mtt.py
(the same as the experiments), not MTTConfig.

Usage (repository root, jaxenv):
    python scratch/agsf_lost_tracks/compare_scatter.py 11 2
The arguments are trial numbers (one row each). Default: 11 (lost) and 2 (not lost).
"""
import os, sys
import matplotlib
matplotlib.use("Agg")
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import jax.numpy as jnp
import standalone_filter_runner as sfr
from simulators import mtt
import util.io

A, SIGMA2, T = 0.5, 2.5, 500
SIM_DIR = f"data.nosync/experiments/mtt/state-dim_4_emission-dim_2_num-timesteps_{T}_a_{A}_sigma2_{SIGMA2}"
LAGSF_DIR = f"{SIM_DIR}/agsf/lagsf/M_100_N_5_L_5_iters_1_ac1_prop_r1_0.9_ac2_prop_r2_0.9"
OUT_DIR = os.path.join(os.path.dirname(__file__), "figures")


class ExpMTTConfig(sfr.SimulatorConfig):
    """The MTT model of the experiments (simulators/mtt.py)."""

    def __init__(self):
        super().__init__(state_dim=4, emission_dim=2, input_dim=1)

    def setup(self):
        return mtt.setup(4, 2, 1, A, SIGMA2)

    def get_inputs(self, key, num_timesteps):
        return mtt.get_inputs(key, num_timesteps)


def run_trial(trial):
    states, observations, true_ps = util.io.load(f"{LAGSF_DIR}/{trial}/gt")
    runner = sfr.FilterRunner(ExpMTTConfig(), num_timesteps=T, seed=trial)
    # Use the stored data of the trial instead of the new simulation.
    runner.states, runner.observations, runner.true_params = states, observations, true_ps

    res = {}
    res["L-AGSF (T1 run)"] = util.io.load(f"{LAGSF_DIR}/{trial}/results")
    res["L-AGSF (new run)"] = runner.run_lagsf(M=100, N=5, L=5, autocov1="prop", r1=0.9,
                                               autocov2="prop", r2=0.9, num_iterations=1)
    res["U-AGSF"] = runner.run_uagsf(M=100, N=5, L=5, alpha=1.0, beta=0.0, kappa=0.0,
                                     autocov1="prop", r1=0.9, autocov2="prop", r2=0.9,
                                     num_iterations=1)
    res["U-GSF"] = runner.run_ugsf(num_components=100, num_iterations=1, alpha=1.0, beta=0.0, kappa=0.0)
    res["BPF"], _ = runner.run_bpf(num_particles=10000, ess_threshold=0.5)

    stored = util.io.load(f"{LAGSF_DIR}/{trial}/mse")
    print(f"\n=== Trial {trial}: stored MSE of the T1 L-AGSF run = {float(stored[0]):.2f}")
    for k, r in res.items():
        runner.print_summary(r, k)
    return runner, res, states


if __name__ == "__main__":
    trials = [int(t) for t in sys.argv[1:]] or [11, 2]
    rows = [run_trial(t) for t in trials]
    order = list(rows[0][1].keys())
    os.makedirs(OUT_DIR, exist_ok=True)
    name = f"scatter_a{A}_sigma2{SIGMA2}_trials_" + "_".join(map(str, trials))
    for ext in ("pdf", "png"):
        rows[0][0].plot_three_panel_scatter(
            [r[1] for r in rows], dims=(0, 2), states_list=[r[2] for r in rows],
            row_titles=[f"trial {t}" for t in trials], algorithm_order=order,
            rasterize_points=True, save_path=os.path.join(OUT_DIR, f"{name}.{ext}"), show=False)
    print("saved", os.path.join(OUT_DIR, name + ".{pdf,png}"))

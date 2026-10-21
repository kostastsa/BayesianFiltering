#!/bin/zsh
# Runs the APF study in order. Output: out/logs/*.log and the npz/npy files in out/.
# Jax scripts use the jaxenv python. The particles-library scripts use ~/.venvs/particles.
cd "${0:A:h}/out" && mkdir -p logs || exit 1
PY=python3
PV=$HOME/.venvs/particles/bin/python
export PYTHONWARNINGS=ignore

# 1. Diagnostics of the first-stage weights (jax), M = 1000, 60 data sets, the six settings of Tables I and III
for a in 0.5 0.05; do
  QS=1e-6 RS=2.5,0.025,2.5e-5 $PY ../apf_regime.py 1000 60 500 $a > logs/regime_jax_a$a.log 2>&1
done
# 2. The same with the particles library (numpy)
for a in 0.5 0.05; do
  QS=1e-6 RS=2.5,0.025,2.5e-5 ACC=$a $PV ../regime_particles.py 1000 60 500 > logs/regime_particles_a$a.log 2>&1
done
# 3. Time alignment of the initial step
for a in 0.5 0.05; do
  QS=1e-6 RS=2.5,0.025,2.5e-5 ACC=$a $PV ../align_particles.py 1000 60 500 > logs/align_a$a.log 2>&1
done
# 4. BPF and APF in the settings of Tables I and III
for a in 0.5 0.05; do
  for r in 2.5 0.025 2.5e-5; do
    for M in 100 1000; do $PY ../expA_rerun.py $a $r $M 100 > logs/rerun_a${a}_r${r}_M$M.log 2>&1; done
    $PY ../expA_rerun.py $a $r 10000 50 > logs/rerun_a${a}_r${r}_M10000.log 2>&1
    FILTERS=bpf,apf,apf_every $PY ../expA_rerun.py $a $r 100000 20 > logs/rerun_a${a}_r${r}_M100000.log 2>&1
  done
done
echo finished > logs/DONE

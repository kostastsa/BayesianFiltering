#!/bin/zsh
# Final runs of the APF study with the corrected filters/pf.py (weights normalized in the log domain).
# 1. first-stage weight diagnostics (jax)  2. our BPF/APF against the particles library  3. the settings of Tables I and III
cd "${0:A:h}/out" && mkdir -p logs || exit 1
export PYTHONWARNINGS=ignore
for a in 0.5 0.05; do
  QS=1e-6 RS=2.5,0.025,2.5e-5 python3 ../apf_regime.py 1000 60 500 $a > logs/regime_jax_a$a.log 2>&1
done
export CONFIGS=1e-6:2.5,1e-6:0.025,1e-6:2.5e-5
for a in 0.5 0.05; do
  export ACC=$a SUF=_a$a
  python3 ../cmp_ours.py 1000 30 500 12 > logs/cmp_ours_a$a.log 2>&1
  ~/.venvs/particles/bin/python ../cmp_particles.py 1000 12 > logs/cmp_particles_a$a.log 2>&1
  ~/.venvs/particles/bin/python ../cmp_report.py > logs/cmp_report_a$a.log 2>&1
done
unset CONFIGS ACC SUF
echo finished > logs/CMP_DONE
../run_rerun.zsh

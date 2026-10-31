#!/bin/zsh
# Our BPF and APF against the particles library on the same data, at the six settings of Tables I and III (M = 1000).
cd "${0:A:h}/out" || exit 1
export PYTHONWARNINGS=ignore CONFIGS=1e-6:2.5,1e-6:0.025,1e-6:2.5e-5
for a in 0.5 0.05; do
  export ACC=$a SUF=_a$a
  python3 ../cmp_ours.py 1000 30 500 12 > logs/cmp_ours_a$a.log 2>&1
  ~/.venvs/particles/bin/python ../cmp_particles.py 1000 12 > logs/cmp_particles_a$a.log 2>&1
  ~/.venvs/particles/bin/python ../cmp_report.py > logs/cmp_report_a$a.log 2>&1
done
echo finished > logs/CMP_DONE

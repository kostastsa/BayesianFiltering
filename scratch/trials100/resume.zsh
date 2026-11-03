#!/bin/zsh
# Resume the 100-trial runs (D-34, task T3) and Table R1 (D-33, task T5) in the tmux session 'trials100'.
# Each job starts at its first incomplete trial (first_incomplete.py), so a stop at any time loses only the running trials.
# Usage (repository root):  ./scratch/trials100/resume.zsh [HH:MM]     HH:MM: wait until this time before the start (e.g. 23:00).
# Six parallel jobs at most (16 GB RAM: more BPF/APF jobs with 10^5 particles fill the swap). Stop: tmux kill-session -t trials100.
cd "${0:A:h}/../.."
R=$PWD; CONDA="$(conda info --base)/etc/profile.d/conda.sh"; L=data.nosync/logs
if [[ -n $1 ]]; then
    now=$(date +%s); tgt=$(date -j -f "%H:%M" "$1" +%s); (( tgt < now )) && tgt=$((tgt + 86400))
    echo "$(date +%T) waiting until $1"; sleep $((tgt - now))
fi
source "$CONDA" && conda activate jaxenv
fi_() { python scratch/trials100/first_incomplete.py "$@" 2>/dev/null }
G=lgsf_table_wrap,ugsf_table_wrap,lagsf_table_wrap,uagsf_table_wrap
job() {  # job NAME SIM A NAMES START KEEP SEED LOG  ->  the shell command of one run_tables job from its first incomplete trial
    local k=$(fi_ $2 $3 $4 $5)
    (( k > 100 )) && { echo "echo '$1 complete'"; return }
    local simvar=""; [[ $2 == lmsv ]] && simvar="SIM=lmsv"
    echo "$simvar KEEP_RESULTS=$6 SEED_BASE=$7 ./run_tables_I_III.zsh $k 100 $3 $4 >> $L/$8 2>&1"
}
pre="cd '$R' && source '$CONDA' && conda activate jaxenv"
win() { tmux new-window -t trials100 -n $1 "zsh -c \"$pre && $2; echo FINISHED; exec zsh\"" }
tmux kill-session -t trials100 2>/dev/null
tmux new-session -d -s trials100 -n A_gsf_a05 "zsh -c \"$pre && $(job a mtt 0.5 $G 21 20 2000 T100_A_gsf_a0.5.log); echo FINISHED; exec zsh\""
win A_gsf_a005 "$(job b mtt 0.05 $G 21 20 2000 T100_A_gsf_a0.05.log)"
win A_pf_a05 "$(job c mtt 0.5 bpf_wrap,apf_wrap 31 30 1000 T100_A_bpfapf_a0.5.log)"
win A_pf_a005 "$(job d mtt 0.05 bpf_wrap,apf_wrap 31 30 1000 T100_A_bpfapf_a0.05.log)"
win A_rpfupf "$(job e mtt 0.5 rpf_wrap,upf_wrap 31 30 3000 T100_A_rpfupf_a0.5.log) && $(job f mtt 0.05 rpf_wrap,upf_wrap 31 30 3000 T100_A_rpfupf_a0.05.log)"
# Exp. B, then Table R1 (light jobs, one after the other)
win B_then_R1 "$(job g lmsv B lgsf_tableII,lagsf_tableII 21 20 4000 T100_B_l.log) && $(job h lmsv B ugsf_tableII,uagsf_tableII 21 20 4000 T100_B_u.log) && $(job i lmsv B bpf_tableII,apf_tableII 31 30 4000 T100_B_bpfapf.log) && $(job j lmsv B rpf_tableII,upf_tableII 31 30 4000 T100_B_rpfupf.log) && ./scratch/particles_check/run_r1_wrap.zsh >> $L/T5_R1_wrap.log 2>&1"
echo "$(date +%T) started:"; tmux list-windows -t trials100 | cut -d' ' -f1-2

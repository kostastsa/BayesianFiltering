#!/bin/zsh
# Watchdog of the 100-trial runs (tmux session 'trials100'). Every 60 s:
# - it sets the priority of all job processes to nice 10, so that the system and the user's programs get the CPU first;
# - if the free disk is below 8 GB, the swap is above 6 GB, or the free memory is below 10 %, it stops the heaviest job
#   (a BPF/APF window: any window whose name starts with A_pf). It stops at most one job every 10 min. resume.zsh continues a stopped job later.
# Log: data.nosync/logs/T100_watchdog.log. Stop the watchdog: tmux kill-session -t trials100_watch.
cd "${0:A:h}/../.."
L=data.nosync/logs/T100_watchdog.log
last=0
desc() { local p; for p in $(pgrep -P $1); do echo $p; desc $p; done }
while tmux has-session -t trials100 2>/dev/null; do
    for pp in $(tmux list-panes -s -t trials100 -F '#{pane_pid}'); do
        for p in $pp $(desc $pp); do renice -n 10 -p $p >/dev/null 2>&1; done
    done
    disk=$(df -g . | awk 'NR==2 {print $4}')
    swap=$(sysctl -n vm.swapusage | awk '{gsub("M","",$6); print int($6)}')
    mem=$(memory_pressure 2>/dev/null | awk -F': ' '/free percentage/ {gsub("%","",$2); print int($2)}')
    now=$(date +%s)
    if (( disk < 8 || swap > 6000 || mem < 10 )) && (( now - last > 600 )); then
        for w in $(tmux list-windows -t trials100 -F '#{window_name}' | grep '^A_pf' | sort -r); do
            if tmux list-windows -t trials100 -F '#{window_name}' | grep -qx $w; then
                tmux kill-window -t trials100:$w
                echo "$(date '+%d-%b %T') STOPPED $w: disk ${disk} GB free, swap ${swap} MB, memory ${mem} % free" >> $L
                last=$now; break
            fi
        done
    fi
    (( $(date +%M) % 30 == 0 )) && echo "$(date '+%d-%b %T') ok: disk ${disk} GB free, swap ${swap} MB, memory ${mem} % free" >> $L
    sleep 60
done
echo "$(date '+%d-%b %T') trials100 ended, watchdog stops" >> $L

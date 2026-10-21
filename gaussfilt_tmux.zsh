#!/bin/zsh

SESSION=gaussfilt_exps
WORKDIR=/Users/kostastsampourakis/Desktop/code/Python/projects/gaussfiltax/
ENVNAME=jaxenv

tmux new-session -d -s $SESSION -c $WORKDIR "conda activate $ENVNAME; zsh"
tmux split-window -v -c $WORKDIR "conda activate $ENVNAME; zsh"
tmux split-window -h -c $WORKDIR "conda activate $ENVNAME; zsh"
tmux select-pane -t 0
tmux split-window -h -c $WORKDIR "conda activate $ENVNAME; zsh"
tmux select-layout tiled
tmux attach -t $SESSION


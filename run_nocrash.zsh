#!/bin/bash

START_TRIAL="$1"
END_TRIAL="$2"
FILES="$3"

while true; do
    python3 gaussfiltax/main.py trials 'r' $1 $2 $3
    status=$?
    if [ $status -eq 0 ]; then
        echo "run_trials completed successfully."
        break
    else
        echo "run_trials crashed with exit code $status. Retrying..."
        sleep 1  # optional: wait a bit before retrying
    fi
done

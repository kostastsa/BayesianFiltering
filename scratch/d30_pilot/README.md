# Pilot of decision D-30: wrapped bearing in Exp. A

This folder compares two versions of the GSF and AGSF rows of Tables I and III: the raw bearing (task T1) and the wrapped bearing. It is a check, not an experiment result (rule 5 of `CLAUDE.md`).

## Question

The bearing `arctan2(x2, x1)` of `mtt` has a jump of 2 pi on the negative x1-axis. The EKF and UKF update steps fail at this jump. Does a wrapped bearing change the rows of Tables I and III? The rule of the test is in decision D-30.

## Runs

- Descriptor files: `exps/mtt/state_dim_4/a_*/mtt_*_table_wrap_scan_num_prt.txt` (written by `create_exps.py`). They have the key `wrap: 1`.
- Command: `SEED_BASE=2000 ./run_tables_I_III.zsh FIRST LAST A lgsf_table_wrap,ugsf_table_wrap,lagsf_table_wrap,uagsf_table_wrap`.
- Logs: `data.nosync/logs/D30_pilot_*.log`.

## Paired comparison

The pilot uses the same seed base, the same filters and the same order of the descriptors as T1. Thus each trial gets the same key. The check on trial 1 (a = 0.5) gave this result for all 36 runs:

- The simulated states are equal to the states of T1.
- The observations are the wrapped observations of T1.

Thus each difference between the old row and the new row comes from the wrap only.

## Files

- `compare.py`: makes the table of the old and new rows (`compare.md`).

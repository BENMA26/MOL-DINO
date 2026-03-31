# QM9 AutoResearch

## What this does

`run_autoresearch_qm9.sh` runs Stage-2 + LDM candidate trials automatically, then:

- appends each trial to `logs/auto_research/qm9_autoresearch_results.csv`
- compresses and rewrites `EXPERIMENT_LOG.md` with current best results
- sends progress emails via local `sendmail`
- commits key changes with git when `git` is available

## UniGEM QM9 target used by the pipeline

- atom stability: 99.0% (`0.99`)
- molecule stability: 89.8% (`0.898`)
- validity: 95.0% (`0.95`)
- validity*uniqueness: 93.2% (`0.932`)

## Launch

```bash
cd /work/home/maben/project/blue_whale_lab/projects/mol_rep/MOL-DINO/3D-GSRD
nohup bash experiments/qm9_autoresearch/run_autoresearch_qm9.sh > logs/auto_research/launcher.log 2>&1 &
```

## Useful env vars

- `EMAIL_TO`: email recipient (default: local user)
- `STAGE2_GPU`: GPU id for Stage-2 (default: `2`)
- `LDM_GPU`: GPU id for LDM (default: `3`)
- `MAX_TRIALS`: limit number of candidates in one batch (default: `3`)
- `STOP_ON_TARGET`: stop when target reached (`1`/`0`, default: `1`)

# Experiment Log

Last Updated: 2026-03-31 11:56:02 
Workspace: `/work/home/maben/project/blue_whale_lab/projects/mol_rep/MOL-DINO/3D-GSRD`

## Current Goal

Surpass UniGEM QM9 generation metrics using MOL-DINO based LDM with UNILIP-inspired stage-2 finetuning strategy.

## UniGEM QM9 Target (ICLR 2025)

- Atom stability >= 0.990 (99.0%)
- Molecule stability >= 0.898 (89.8%)
- Validity >= 0.950 (95.0%)
- Validity*Uniqueness >= 0.932 (93.2%)

## Auto-Research Status

- Result CSV: `logs/auto_research/qm9_autoresearch_results.csv`
- Completed trials: 0
- Best run: N/A
- Target reached: `NO`

## Active Processes

- `532:maben    301081      1  0 00:20 ?        00:00:00 sh run_stage2_distill_then_ldm_every50.sh`
- `533:maben    301207 301081 99 00:20 ?        2-22:17:45 /work/home/maben/software/anaconda3/envs/rep/bin/python trainer_qm9_gen.py --disable_compile --seed 0 --filename flow_ae_qm9_stage2_distill1_every50 --root ./data/q`
- `587:maben    401322 394521  0 11:55 ?        00:00:00 bash experiments/qm9_autoresearch/run_autoresearch_qm9.sh`

## Top Results (By V*U)

No completed auto-research trials yet.

## Latest Note

- Auto-research pipeline initialized.

# Experiment Log

Last Updated: 2026-03-31 17:23:36 
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
- Completed trials: 1
- Best run: `20260331_120139_u_beta_lin`, atom=0.08989272266626358, mol=0.0, valid=0.8180000185966492, V*U=0.11699999868869781
- Target reached: `NO`

## Active Processes

- `544:maben    301081      1  0 00:20 ?        00:00:00 sh run_stage2_distill_then_ldm_every50.sh`
- `545:maben    301207 301081 99 00:20 ?        4-03:09:16 /work/home/maben/software/anaconda3/envs/rep/bin/python trainer_qm9_gen.py --disable_compile --seed 0 --filename flow_ae_qm9_stage2_distill1_every50 --root ./data/q`
- `572:maben    403370      1  0 12:01 ?        00:00:00 bash experiments/qm9_autoresearch/run_autoresearch_qm9.sh`
- `575:maben    410250 410248 99 12:47 ?        13:45:05 /work/home/maben/software/anaconda3/envs/rep/bin/python trainer_qm9_gen.py --disable_compile --seed 0 --filename safe_ablate_geom_stage2_20260331_124727_cfg1 --root .`

## Top Results (By V*U)

| rank | run_id | atom_stability | molecule_stability | validity | uniqueness | V*U |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 20260331_120139_u_beta_lin | 0.08989272266626358 | 0.0 | 0.8180000185966492 | 0.1436167061328888 | 0.11699999868869781 |

## Latest Note

- Trial 20260331_120139_u_beta_lin done: atom=0.08989272266626358, mol=0.0, valid=0.8180000185966492, V*U=0.11699999868869781, target=NO

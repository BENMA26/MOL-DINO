#!/usr/bin/env bash
# run_stage2_distill_then_ldm_every50.sh
# 1) 用 Stage1 checkpoint 进行 Stage2(distill) 训练
# 2) 从 Stage2 checkpoints 中选出每隔 INTERVAL_EPOCHS 的 checkpoint
# 3) 逐个启动 LDM 训练并汇总指标
#
# 用法（全部可选，推荐用环境变量覆盖）:
#   bash run_stage2_distill_then_ldm_every50.sh
#
# 常用环境变量:
#   STAGE1_CKPT, STAGE2_RUN_NAME, LDM_RUN_PREFIX
#   STAGE2_GPU, LDM_GPU
#   DISTILL_WEIGHT, ENCODER_LR_RATIO, LATENT_NOISE_STD
#   DISTILL_DECAY, DISTILL_MIN_RATIO
#   INTERVAL_EPOCHS
#   STAGE2_MAX_EPOCHS, STAGE2_MAX_STEPS
#   LDM_MAX_EPOCHS, LDM_MAX_STEPS

set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs

PYTHON_BIN="${PYTHON_BIN:-/work/home/maben/software/anaconda3/envs/rep/bin/python}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="$(command -v python)"
fi
if ! "${PYTHON_BIN}" -c 'import lightning' >/dev/null 2>&1; then
  echo "Python ${PYTHON_BIN} does not have lightning installed." >&2
  exit 1
fi

STAGE1_CKPT="${STAGE1_CKPT:-/work/home/maben/project/blue_whale_lab/projects/mol_rep/MOL-DINO/3D-GSRD/all_checkpoints/flow_ae_qm9_stage1/last.ckpt}"
if [[ ! -f "${STAGE1_CKPT}" ]]; then
  echo "Stage1 checkpoint not found: ${STAGE1_CKPT}" >&2
  exit 1
fi

STAGE2_RUN_NAME="${STAGE2_RUN_NAME:-flow_ae_qm9_stage2_distill1_every50}"
LDM_RUN_PREFIX="${LDM_RUN_PREFIX:-ldm_qm9_from_${STAGE2_RUN_NAME}}"

STAGE2_GPU="${STAGE2_GPU:-0}"
LDM_GPU="${LDM_GPU:-${STAGE2_GPU}}"

DISTILL_WEIGHT="${DISTILL_WEIGHT:-1.0}"
ENCODER_LR_RATIO="${ENCODER_LR_RATIO:-1.0}"
LATENT_NOISE_STD="${LATENT_NOISE_STD:-0.0}"
DISTILL_DECAY="${DISTILL_DECAY:-none}"
DISTILL_MIN_RATIO="${DISTILL_MIN_RATIO:-0.0}"

INTERVAL_EPOCHS="${INTERVAL_EPOCHS:-50}"

STAGE2_MAX_EPOCHS="${STAGE2_MAX_EPOCHS:-1000}"
STAGE2_MAX_STEPS="${STAGE2_MAX_STEPS:-2500000}"

LDM_MAX_EPOCHS="${LDM_MAX_EPOCHS:-20}"
LDM_MAX_STEPS="${LDM_MAX_STEPS:-40000}"
LDM_TRAIN_SMILES_PATH="${LDM_TRAIN_SMILES_PATH:-}"

STAGE2_LOG="logs/${STAGE2_RUN_NAME}.log"
SUMMARY_CSV="logs/${LDM_RUN_PREFIX}_summary.csv"

echo "===================================================="
echo "Stage2 run name      : ${STAGE2_RUN_NAME}"
echo "LDM run prefix       : ${LDM_RUN_PREFIX}"
echo "Stage1 checkpoint    : ${STAGE1_CKPT}"
echo "Stage2 GPU / LDM GPU : ${STAGE2_GPU} / ${LDM_GPU}"
echo "Distill settings     : distill=${DISTILL_WEIGHT}, decay=${DISTILL_DECAY}, min_ratio=${DISTILL_MIN_RATIO}, enc_lr_ratio=${ENCODER_LR_RATIO}, latent_noise=${LATENT_NOISE_STD}"
echo "Interval epochs      : ${INTERVAL_EPOCHS}"
echo "Stage2 epochs/steps  : ${STAGE2_MAX_EPOCHS} / ${STAGE2_MAX_STEPS}"
echo "LDM epochs/steps     : ${LDM_MAX_EPOCHS} / ${LDM_MAX_STEPS}"
echo "===================================================="

echo "[1/3] Stage2(distill) training..."
CUDA_VISIBLE_DEVICES="${STAGE2_GPU}" "${PYTHON_BIN}" trainer_qm9_gen.py \
  --disable_compile \
  --seed 0 \
  --filename "${STAGE2_RUN_NAME}" \
  --root ./data/qm9 \
  --dataset qm9 \
  --dataset_arg homo \
  --train_size 100000 \
  --val_size 17748 \
  --test_size 13083 \
  --batch_size 64 \
  --inference_batch_size 32 \
  --num_workers 4 \
  --aug_translation \
  --aug_translation_scale 0.1 \
  --accelerator gpu \
  --devices 1 \
  --precision 32-true \
  --max_epochs "${STAGE2_MAX_EPOCHS}" \
  --max_steps "${STAGE2_MAX_STEPS}" \
  --check_val_every_n_epoch 10 \
  --save_every_n_epochs "${INTERVAL_EPOCHS}" \
  --accumulate_grad_batches 1 \
  --gradient_clip_val 1.0 \
  --scheduler none \
  --init_lr 1e-5 \
  --min_lr 1e-7 \
  --warmup_lr 1e-7 \
  --warmup_steps 1000 \
  --weight_decay 1e-8 \
  --node_dim 14 \
  --edge_dim 4 \
  --hidden_dim 256 \
  --n_heads 8 \
  --encoder_blocks 8 \
  --trans_version v6 \
  --attn_activation silu \
  --decoder_blocks 8 \
  --atom_dim 5 \
  --max_num_atoms 29 \
  --stage 2 \
  --ckpt_path "${STAGE1_CKPT}" \
  --time_distribution uniform \
  --sample_schedule linear \
  --distill_weight "${DISTILL_WEIGHT}" \
  --distill_decay "${DISTILL_DECAY}" \
  --distill_min_ratio "${DISTILL_MIN_RATIO}" \
  --encoder_lr_ratio "${ENCODER_LR_RATIO}" \
  --latent_noise_std "${LATENT_NOISE_STD}" \
  > "${STAGE2_LOG}" 2>&1

STAGE2_CKPT_DIR="all_checkpoints/${STAGE2_RUN_NAME}"
if [[ ! -d "${STAGE2_CKPT_DIR}" ]]; then
  echo "Stage2 checkpoint dir not found: ${STAGE2_CKPT_DIR}" >&2
  exit 1
fi

echo "[2/3] Collect Stage2 checkpoints (every ${INTERVAL_EPOCHS} epochs + last)..."
mapfile -t CKPT_ROWS < <(
  "${PYTHON_BIN}" - "${STAGE2_CKPT_DIR}" "${INTERVAL_EPOCHS}" <<'PY'
import re
import sys
from pathlib import Path

ckpt_dir = Path(sys.argv[1])
interval = int(sys.argv[2])

epoch_to_file = {}
for p in ckpt_dir.glob("*.ckpt"):
    name = p.name
    m = re.match(r"^epoch=(\d+)\.ckpt$", name)
    if m is None:
        m2 = re.match(r"^(\d+)\.ckpt$", name)
        m = m2
    if m is None:
        continue
    epoch_zero_based = int(m.group(1))
    epoch_one_based = epoch_zero_based + 1
    if epoch_one_based % interval != 0:
        continue
    old = epoch_to_file.get(epoch_one_based)
    if old is None or p.stat().st_mtime > old.stat().st_mtime:
        epoch_to_file[epoch_one_based] = p

for e in sorted(epoch_to_file):
    print(f"epoch{e}\t{epoch_to_file[e]}")

last_ckpt = ckpt_dir / "last.ckpt"
if not last_ckpt.exists():
    cands = sorted(ckpt_dir.glob("last*.ckpt"), key=lambda x: x.stat().st_mtime, reverse=True)
    if cands:
        last_ckpt = cands[0]
    else:
        last_ckpt = None

if last_ckpt is not None:
    print(f"last\t{last_ckpt}")
PY
)

if [[ ${#CKPT_ROWS[@]} -eq 0 ]]; then
  echo "No Stage2 checkpoints selected. Check save cadence and max epochs." >&2
  exit 1
fi

echo "stage2_tag,stage2_ckpt,ldm_run,ldm_metrics_csv,test_ldm_loss,atom_stability,molecule_stability,validity_edm,uniqueness_edm,validity_uniqueness_edm" > "${SUMMARY_CSV}"

echo "[3/3] Launch LDM runs sequentially..."
for row in "${CKPT_ROWS[@]}"; do
  stage2_tag="${row%%$'\t'*}"
  stage2_ckpt="${row#*$'\t'}"

  if [[ ! -f "${stage2_ckpt}" ]]; then
    echo "Skip missing checkpoint: ${stage2_ckpt}" >&2
    continue
  fi

  ldm_run="${LDM_RUN_PREFIX}_${stage2_tag}"
  ldm_log="logs/${ldm_run}.log"

  echo "----------------------------------------------------"
  echo "Stage2 checkpoint tag : ${stage2_tag}"
  echo "Stage2 checkpoint     : ${stage2_ckpt}"
  echo "LDM run               : ${ldm_run}"
  echo "Log                   : ${ldm_log}"

  LDM_CMD=(
    "${PYTHON_BIN}" trainer_qm9_ldm.py
    --seed 0
    --filename "${ldm_run}"
    --root ./data/qm9
    --dataset qm9
    --dataset_arg homo
    --train_size 100000
    --val_size 17748
    --test_size 13083
    --batch_size 64
    --inference_batch_size 125
    --num_workers 4
    --aug_translation
    --aug_translation_scale 0.1
    --accelerator gpu
    --devices 1
    --precision 32-true
    --max_epochs "${LDM_MAX_EPOCHS}"
    --max_steps "${LDM_MAX_STEPS}"
    --check_val_every_n_epoch 5
    --save_every_n_epochs 10
    --accumulate_grad_batches 1
    --gradient_clip_val 1.0
    --scheduler linear_warmup_cosine_lr
    --init_lr 1e-4
    --min_lr 1e-6
    --warmup_lr 1e-6
    --warmup_steps 1000
    --weight_decay 1e-8
    --node_dim 14
    --edge_dim 4
    --hidden_dim 256
    --n_heads 8
    --encoder_blocks 8
    --decoder_blocks 8
    --trans_version v6
    --attn_activation silu
    --atom_dim 5
    --max_num_atoms 29
    --stage2_ckpt "${stage2_ckpt}"
    --ldm_num_heads 8
    --ldm_num_layers 6
    --ldm_ffn_mult 4
    --ldm_dropout 0.0
    --ldm_min_t 1e-3
    --ldm_sample_steps 100
    --ldm_eval_mol_metrics
    --ldm_eval_on_test
    --ldm_eval_every_n_epochs 1
    --ldm_eval_num_batches 8
    --ldm_eval_decode_steps 100
    --ldm_decode_sample_schedule linear
  )
  if [[ -n "${LDM_TRAIN_SMILES_PATH}" ]]; then
    LDM_CMD+=(--ldm_train_smiles_path "${LDM_TRAIN_SMILES_PATH}")
  fi

  CUDA_VISIBLE_DEVICES="${LDM_GPU}" "${LDM_CMD[@]}" > "${ldm_log}" 2>&1

  ldm_metrics_csv="$(ls -1dt all_checkpoints/${ldm_run}/lightning_logs/version_*/metrics.csv | head -n 1)"
  if [[ -z "${ldm_metrics_csv}" || ! -f "${ldm_metrics_csv}" ]]; then
    echo "Missing LDM metrics csv for ${ldm_run}" >&2
    exit 1
  fi

  "${PYTHON_BIN}" - "${SUMMARY_CSV}" "${stage2_tag}" "${stage2_ckpt}" "${ldm_run}" "${ldm_metrics_csv}" <<'PY'
import csv
import sys

summary_csv, stage2_tag, stage2_ckpt, ldm_run, metrics_csv = sys.argv[1:]
row = list(csv.DictReader(open(metrics_csv)))[-1]

with open(summary_csv, "a", newline="") as f:
    w = csv.writer(f)
    w.writerow([
        stage2_tag,
        stage2_ckpt,
        ldm_run,
        metrics_csv,
        row.get("test/ldm_loss", ""),
        row.get("test/mol/atom_stability", ""),
        row.get("test/mol/molecule_stability", ""),
        row.get("test/mol/validity_edm", ""),
        row.get("test/mol/uniqueness_edm", ""),
        row.get("test/mol/validity_uniqueness_edm", ""),
    ])
PY
done

echo "Done."
echo "Stage2 log   : ${STAGE2_LOG}"
echo "Summary CSV  : ${SUMMARY_CSV}"

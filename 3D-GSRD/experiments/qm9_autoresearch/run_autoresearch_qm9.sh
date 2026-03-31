#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."
ROOT_DIR="$(pwd)"
mkdir -p logs/auto_research

PYTHON_BIN="${PYTHON_BIN:-/work/home/maben/software/anaconda3/envs/rep/bin/python}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="$(command -v python)"
fi
if ! "${PYTHON_BIN}" -c 'import lightning' >/dev/null 2>&1; then
  echo "Python ${PYTHON_BIN} does not have lightning installed." >&2
  exit 1
fi

STAGE1_CKPT="${STAGE1_CKPT:-${ROOT_DIR}/all_checkpoints/flow_ae_qm9_stage1/last.ckpt}"
if [[ ! -f "${STAGE1_CKPT}" ]]; then
  echo "Stage1 checkpoint not found: ${STAGE1_CKPT}" >&2
  exit 1
fi

STAGE2_GPU="${STAGE2_GPU:-2}"
LDM_GPU="${LDM_GPU:-3}"
EMAIL_TO="${EMAIL_TO:-$(whoami)}"
EMAIL_FROM="${EMAIL_FROM:-autoresearch@localhost}"
STOP_ON_TARGET="${STOP_ON_TARGET:-1}"
MAX_TRIALS="${MAX_TRIALS:-3}"

TARGET_ATOM="${TARGET_ATOM:-0.99}"
TARGET_MOL="${TARGET_MOL:-0.898}"
TARGET_VALID="${TARGET_VALID:-0.95}"
TARGET_VU="${TARGET_VU:-0.932}"

RESULT_CSV="logs/auto_research/qm9_autoresearch_results.csv"
if [[ ! -f "${RESULT_CSV}" ]]; then
  echo "timestamp,run_id,stage2_run,stage2_ckpt,ldm_run,ldm_metrics_csv,distill_weight,distill_decay,distill_min_ratio,encoder_lr_ratio,latent_noise_std,time_distribution,time_alpha_factor,sample_schedule,stage2_init_lr,ldm_init_lr,ldm_layers,ldm_heads,ldm_sample_steps,ldm_decode_steps,test_ldm_loss,atom_stability,molecule_stability,validity_edm,uniqueness_edm,validity_uniqueness_edm,target_reached" > "${RESULT_CSV}"
fi

send_email() {
  local subject="$1"
  local body="$2"
  if [[ -z "${EMAIL_TO}" ]]; then
    return 0
  fi
  if ! command -v sendmail >/dev/null 2>&1; then
    echo "[WARN] sendmail not found, skip email: ${subject}"
    return 0
  fi
  {
    echo "To: ${EMAIL_TO}"
    echo "From: ${EMAIL_FROM}"
    echo "Subject: ${subject}"
    echo "Content-Type: text/plain; charset=UTF-8"
    echo
    echo "${body}"
  } | sendmail -t || true
}

update_log() {
  local note="$1"
  "${PYTHON_BIN}" experiments/qm9_autoresearch/update_experiment_log.py \
    --result_csv "${RESULT_CSV}" \
    --log_file "${ROOT_DIR}/EXPERIMENT_LOG.md" \
    --workspace "${ROOT_DIR}" \
    --latest_note "${note}" \
    --target_atom "${TARGET_ATOM}" \
    --target_mol "${TARGET_MOL}" \
    --target_valid "${TARGET_VALID}" \
    --target_vu "${TARGET_VU}"
}

maybe_git_commit() {
  local msg="$1"
  if ! command -v git >/dev/null 2>&1; then
    echo "[WARN] git not found, skip commit"
    return 0
  fi
  if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    echo "[WARN] not in git worktree, skip commit"
    return 0
  fi
  git add trainer_qm9_gen.py run_stage2_distill_then_ldm_every50.sh \
    experiments/qm9_autoresearch/run_autoresearch_qm9.sh \
    experiments/qm9_autoresearch/update_experiment_log.py \
    EXPERIMENT_LOG.md || true
  if git diff --cached --quiet; then
    return 0
  fi
  git commit -m "${msg}" || true
}

update_log "Auto-research pipeline initialized."
send_email "[QM9 AutoResearch] pipeline started" \
  "Workspace: ${ROOT_DIR}\nResult CSV: ${RESULT_CSV}\nTargets: atom=${TARGET_ATOM}, mol=${TARGET_MOL}, valid=${TARGET_VALID}, V*U=${TARGET_VU}"

CANDIDATES=(
  "u_beta_lin|1.0|linear|0.20|0.30|0.005|beta|0.50|linear|1e-5|80|180000|1|8|8|1e-4|40|80000|200|300|linear"
  "u_beta_cos|1.5|cosine|0.15|0.30|0.010|beta|0.70|power|8e-6|100|220000|1|8|8|8e-5|50|100000|300|400|power"
  "u_uniform_lin|1.0|linear|0.30|0.10|0.005|uniform|2.00|linear|8e-6|120|260000|1|10|8|1e-4|60|120000|300|500|linear"
)

echo "AutoResearch start: MAX_TRIALS=${MAX_TRIALS}, STOP_ON_TARGET=${STOP_ON_TARGET}, STAGE2_GPU=${STAGE2_GPU}, LDM_GPU=${LDM_GPU}"
echo "Candidate count: ${#CANDIDATES[@]}"

if (( MAX_TRIALS <= 0 )); then
  echo "MAX_TRIALS<=0, no trials executed."
  update_log "Auto-research skipped because MAX_TRIALS<=0."
  send_email "[QM9 AutoResearch] skipped" "MAX_TRIALS<=0, no trials executed."
  exit 0
fi

trial_count=0
for cand in "${CANDIDATES[@]}"; do
  if (( trial_count >= MAX_TRIALS )); then
    break
  fi
  trial_count=$((trial_count + 1))

  IFS='|' read -r tag distill_weight distill_decay distill_min_ratio encoder_lr_ratio latent_noise_std \
    time_distribution time_alpha_factor sample_schedule stage2_lr stage2_epochs stage2_steps \
    ldm_center_latents ldm_layers ldm_heads ldm_lr ldm_epochs ldm_steps ldm_sample_steps \
    ldm_decode_steps ldm_decode_schedule <<< "${cand}"

  ts="$(date +%Y%m%d_%H%M%S)"
  run_id="${ts}_${tag}"
  stage2_run="flow_ae_qm9_auto_${run_id}"
  ldm_run="ldm_qm9_auto_${run_id}"
  stage2_log="logs/auto_research/${stage2_run}.log"
  ldm_log="logs/auto_research/${ldm_run}.log"

  echo "============================================================"
  echo "Trial ${trial_count}/${MAX_TRIALS}: ${run_id}"
  echo "Stage2 run: ${stage2_run} (GPU ${STAGE2_GPU})"
  echo "LDM run:    ${ldm_run} (GPU ${LDM_GPU})"
  echo "============================================================"

  CUDA_VISIBLE_DEVICES="${STAGE2_GPU}" "${PYTHON_BIN}" trainer_qm9_gen.py \
    --disable_compile \
    --seed 0 \
    --filename "${stage2_run}" \
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
    --max_epochs "${stage2_epochs}" \
    --max_steps "${stage2_steps}" \
    --check_val_every_n_epoch 10 \
    --save_every_n_epochs 20 \
    --accumulate_grad_batches 1 \
    --gradient_clip_val 1.0 \
    --scheduler none \
    --init_lr "${stage2_lr}" \
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
    --time_distribution "${time_distribution}" \
    --time_alpha_factor "${time_alpha_factor}" \
    --sample_schedule "${sample_schedule}" \
    --distill_weight "${distill_weight}" \
    --distill_decay "${distill_decay}" \
    --distill_min_ratio "${distill_min_ratio}" \
    --encoder_lr_ratio "${encoder_lr_ratio}" \
    --latent_noise_std "${latent_noise_std}" \
    > "${stage2_log}" 2>&1

  stage2_ckpt="all_checkpoints/${stage2_run}/last.ckpt"
  if [[ ! -f "${stage2_ckpt}" ]]; then
    stage2_ckpt="$(ls -1t all_checkpoints/${stage2_run}/*.ckpt | head -n 1)"
  fi
  if [[ -z "${stage2_ckpt}" || ! -f "${stage2_ckpt}" ]]; then
    note="Trial ${run_id} failed: Stage2 checkpoint not found."
    update_log "${note}"
    send_email "[QM9 AutoResearch] ${run_id} failed" "${note}\nLog: ${stage2_log}"
    continue
  fi

  LDM_EXTRA=()
  if [[ "${ldm_center_latents}" == "1" ]]; then
    LDM_EXTRA+=(--ldm_center_latents)
  fi

  CUDA_VISIBLE_DEVICES="${LDM_GPU}" "${PYTHON_BIN}" trainer_qm9_ldm.py \
    --seed 0 \
    --filename "${ldm_run}" \
    --root ./data/qm9 \
    --dataset qm9 \
    --dataset_arg homo \
    --train_size 100000 \
    --val_size 17748 \
    --test_size 13083 \
    --batch_size 64 \
    --inference_batch_size 125 \
    --num_workers 4 \
    --aug_translation \
    --aug_translation_scale 0.1 \
    --accelerator gpu \
    --devices 1 \
    --precision 32-true \
    --max_epochs "${ldm_epochs}" \
    --max_steps "${ldm_steps}" \
    --check_val_every_n_epoch 5 \
    --save_every_n_epochs 10 \
    --accumulate_grad_batches 1 \
    --gradient_clip_val 1.0 \
    --scheduler linear_warmup_cosine_lr \
    --init_lr "${ldm_lr}" \
    --min_lr 1e-6 \
    --warmup_lr 1e-6 \
    --warmup_steps 1000 \
    --weight_decay 1e-8 \
    --node_dim 14 \
    --edge_dim 4 \
    --hidden_dim 256 \
    --n_heads 8 \
    --encoder_blocks 8 \
    --decoder_blocks 8 \
    --trans_version v6 \
    --attn_activation silu \
    --atom_dim 5 \
    --max_num_atoms 29 \
    --stage2_ckpt "${stage2_ckpt}" \
    --ldm_num_heads "${ldm_heads}" \
    --ldm_num_layers "${ldm_layers}" \
    --ldm_ffn_mult 4 \
    --ldm_dropout 0.0 \
    --ldm_min_t 1e-3 \
    --ldm_sample_steps "${ldm_sample_steps}" \
    --ldm_eval_mol_metrics \
    --ldm_eval_on_test \
    --ldm_eval_every_n_epochs 1 \
    --ldm_eval_num_batches 8 \
    --ldm_eval_decode_steps "${ldm_decode_steps}" \
    --ldm_decode_sample_schedule "${ldm_decode_schedule}" \
    "${LDM_EXTRA[@]}" \
    > "${ldm_log}" 2>&1

  if rg -q "\\[WARN\\] Missing keys|\\[WARN\\] Unexpected keys" "${ldm_log}"; then
    note="Trial ${run_id} failed: Stage2/LDM key mismatch."
    update_log "${note}"
    send_email "[QM9 AutoResearch] ${run_id} failed" "${note}\nLog: ${ldm_log}"
    continue
  fi

  ldm_metrics_csv="$(ls -1dt all_checkpoints/${ldm_run}/lightning_logs/version_*/metrics.csv | head -n 1)"
  if [[ -z "${ldm_metrics_csv}" || ! -f "${ldm_metrics_csv}" ]]; then
    note="Trial ${run_id} failed: missing LDM metrics csv."
    update_log "${note}"
    send_email "[QM9 AutoResearch] ${run_id} failed" "${note}\nLog: ${ldm_log}"
    continue
  fi

  metrics_line="$("${PYTHON_BIN}" - "${ldm_metrics_csv}" "${TARGET_ATOM}" "${TARGET_MOL}" "${TARGET_VALID}" "${TARGET_VU}" <<'PY'
import csv
import sys

metrics_csv = sys.argv[1]
ta, tm, tv, tvu = map(float, sys.argv[2:])
row = list(csv.DictReader(open(metrics_csv)))[-1]

def g(key):
    try:
        return float(row.get(key, "nan"))
    except Exception:
        return float("nan")

atom = g("test/mol/atom_stability")
mol = g("test/mol/molecule_stability")
valid = g("test/mol/validity_edm")
uniq = g("test/mol/uniqueness_edm")
vu = g("test/mol/validity_uniqueness_edm")
loss = row.get("test/ldm_loss", "")
hit = atom >= ta and mol >= tm and valid >= tv and vu >= tvu
print(f"{loss},{atom},{mol},{valid},{uniq},{vu},{'YES' if hit else 'NO'}")
PY
)"

  IFS=',' read -r test_ldm_loss atom_stability molecule_stability validity_edm uniqueness_edm validity_uniqueness_edm target_reached <<< "${metrics_line}"
  timestamp="$(date '+%Y-%m-%d %H:%M:%S')"
  echo "${timestamp},${run_id},${stage2_run},${stage2_ckpt},${ldm_run},${ldm_metrics_csv},${distill_weight},${distill_decay},${distill_min_ratio},${encoder_lr_ratio},${latent_noise_std},${time_distribution},${time_alpha_factor},${sample_schedule},${stage2_lr},${ldm_lr},${ldm_layers},${ldm_heads},${ldm_sample_steps},${ldm_decode_steps},${test_ldm_loss},${atom_stability},${molecule_stability},${validity_edm},${uniqueness_edm},${validity_uniqueness_edm},${target_reached}" >> "${RESULT_CSV}"

  note="Trial ${run_id} done: atom=${atom_stability}, mol=${molecule_stability}, valid=${validity_edm}, V*U=${validity_uniqueness_edm}, target=${target_reached}"
  update_log "${note}"
  send_email "[QM9 AutoResearch] ${run_id} completed (${target_reached})" \
    "${note}\nStage2 log: ${stage2_log}\nLDM log: ${ldm_log}\nMetrics: ${ldm_metrics_csv}"
  maybe_git_commit "auto-research: ${run_id} target=${target_reached}"

  if [[ "${target_reached}" == "YES" && "${STOP_ON_TARGET}" == "1" ]]; then
    echo "Target reached by ${run_id}, stop early."
    break
  fi
done

update_log "Auto-research pipeline finished current candidate list."
send_email "[QM9 AutoResearch] pipeline finished" \
  "Completed ${trial_count} trial(s). Result CSV: ${RESULT_CSV}"

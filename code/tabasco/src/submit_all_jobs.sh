#!/bin/bash
# 一键提交所有DINO生成实验作业 (QM9 dataset)

echo "=========================================="
echo "DINO Generation Experiment (QM9 + UniGEM)"
echo "=========================================="

# Stage 1: 冻结encoder训练decoder
echo ""
echo "[1/3] Submitting Stage 1: Freeze encoder, train decoder..."
JOB1=$(qsub train_dino_ae_stage1_pbs.sh)
echo "Job ID: $JOB1"

# Stage 2: 解冻encoder端到端微调（依赖Stage 1）
echo ""
echo "[2/3] Submitting Stage 2: Unfreeze encoder, end-to-end finetune..."
echo "Note: This job will wait for Stage 1 to complete"
JOB2=$(qsub -W depend=afterok:$JOB1 train_dino_ae_stage2_pbs.sh)
echo "Job ID: $JOB2"

# Stage 3: Flow matching训练（依赖Stage 2）
echo ""
echo "[3/3] Submitting Stage 3: Train flow matching in latent space..."
echo "Note: This job will wait for Stage 2 to complete"
JOB3=$(qsub -W depend=afterok:$JOB2 train_dino_flow_pbs.sh)
echo "Job ID: $JOB3"

echo ""
echo "=========================================="
echo "All jobs submitted successfully!"
echo "=========================================="
echo "Stage 1 (AE freeze):   $JOB1"
echo "Stage 2 (AE finetune): $JOB2 (depends on $JOB1)"
echo "Stage 3 (Flow):        $JOB3 (depends on $JOB2)"
echo ""
echo "Monitor jobs with: qstat -u \$USER"
echo "Check logs in current directory:"
echo "  - dino_ae_stage1.log"
echo "  - dino_ae_stage2.log"
echo "  - dino_flow.log"
echo ""
echo "After all jobs complete, run evaluation:"
echo "  python eval_dino_gen.py \\"
echo "    --ae_ckpt /scratch/yuxuan.ren/maben/outputs/dino_ae_qm9_stage2/best.ckpt \\"
echo "    --flow_ckpt /scratch/yuxuan.ren/maben/outputs/dino_flow_qm9/best.ckpt \\"
echo "    --ref_data_dir /scratch/yuxuan.ren/maben/data/qm9/qm9_train.pt \\"
echo "    --lmdb_dir /scratch/yuxuan.ren/maben/lmdb_cache/qm9 \\"
echo "    --n_samples 1000 \\"
echo "    --num_steps 100 \\"
echo "    --batch_size 64"
echo "=========================================="

#!/bin/bash
#PBS -N dino_ae_stage1
#PBS -P CFP03-CF-130
#PBS -l walltime=144:00:00
#PBS -l select=1:ncpus=10:mpiprocs=1:ompthreads=10:mem=64gb:ngpus=1
#PBS -j oe
#PBS -o dino_ae_stage1.log

# ============================================================
# 1. Environment
# ============================================================
source /scratch/yuxuan.ren/miniconda3/bin/activate
conda activate mol_dino
cd /scratch/yuxuan.ren/maben/code/tabasco/src

export CUDA_VISIBLE_DEVICES="0"
export TRITON_PTXAS_PATH=$CONDA_PREFIX/bin/ptxas
export TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS=1

echo "=========================================="
echo "Job started at $(date)"
echo "Running on node: $(hostname)"
echo "Working directory: $(pwd)"
echo "=========================================="

# ============================================================
# 2. Stage 1: freeze encoder, train decoder (QM9 dataset)
# ============================================================
python train_dino_ae.py \
    --gsrd_checkpoint /scratch/yuxuan.ren/maben/checkpoints/pretrain.ckpt \
    --data_dir /scratch/yuxuan.ren/maben/data/qm9/qm9_train.pt \
    --val_data_dir /scratch/yuxuan.ren/maben/data/qm9/qm9_val.pt \
    --lmdb_dir /scratch/yuxuan.ren/maben/lmdb_cache/qm9 \
    --output_dir /scratch/yuxuan.ren/maben/outputs/dino_ae_qm9_stage1 \
    --freeze_encoder \
    --max_epochs 100 \
    --batch_size 64 \
    --lr 1e-4 \
    --hidden_dim 256 \
    --gsrd_hidden_dim 256 \
    --kl_dim 6 \
    --num_heads 8 \
    --num_layers 8 \
    --num_workers 4

echo "Stage 1 done at $(date)"
echo "=========================================="

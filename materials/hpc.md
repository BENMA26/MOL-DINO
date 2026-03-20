---
layout: page
permalink: /hpc_user_guide/
title: HPC Guide
description: Instructions on how to register for, connect to, and efficiently use the NUS High Performance Computing (HPC) clusters.
nav: true
nav_order: 3
---
![hpc_guide](../pics/hpc-img/hpc_guide.png)
# NUS HPC User Guide (Atlas / Vanda / Hopper)

This guide provides instructions on how to register for, connect to, and efficiently use the NUS High Performance Computing (HPC) clusters.

## Table of Contents
1.  [Registration & Account Setup](#1-registration--account-setup)
2.  [Cluster Overview](#2-cluster-overview)
3.  [Connection Methods (SSH & VS Code)](#3-connection-methods)
4.  [Storage & Quotas](#4-storage--quotas)
5.  [Environment Management (Conda)](#5-environment-management-conda)
6.  [Job Submission (PBS)](#6-job-submission-pbs)
7.  [Common Commands Cheat Sheet](#7-common-commands-cheat-sheet)

---

## 1. Registration & Account Setup

To use the HPC resources, you must complete a two-step registration process.

1.  **SoC Cluster Acceptance**:
    *   Visit the [SoC MyAccount Services](https://mysoc.nus.edu.sg/~myacct/services.cgi).
    *   Login and agree to the terms for "Unix Cluster / Compute Cluster".
2.  **HPC Account Application**:
    *   Visit the [NUS IT HPC Request Page](https://nusit.nus.edu.sg/hpc/get-an-hpc-account/).
    *   Submit the application form.
    *   **Note for Hopper Cluster**: While Atlas and Vanda offer free tiers, the **Hopper** cluster (H100/H200 GPUs) is restricted. You must ask **Prof. Bian** to add your NUS ID to the lab's specific **Project ID** to gain access.

---

## 2. Cluster Overview

Choose the appropriate cluster based on your computational needs:

| Cluster Name | Hostname | Hardware Specs | Recommended Use Case |
| :--- | :--- | :--- | :--- |
| **Atlas** | `atlas9.nus.edu.sg` | AMD EPYC 7R13 (360GB RAM) | **CPU-only tasks**, data preprocessing, standard ML. |
| **Vanda** | `vanda.nus.edu.sg` | Intel 8452Y + **NVIDIA A40 (48GB)** | **Deep Learning**, Training standard models. High VRAM. |
| **Hopper** | `hopper.nus.edu.sg` | Intel 8480+ + **NVIDIA H100/H200** | **LLM Training/Fine-tuning**. High-performance needs only. |

---

## 3. Connection Methods

**Prerequisite:** If you are off-campus, you must connect to the **NUS VPN** (Pulse Secure / GlobalProtect) first.

### 3.1 Basic SSH (Terminal)
Use your terminal (Mac/Linux) or PowerShell/MobaXterm (Windows).
*   **Username**: Your NUS ID (e.g., `e0123456` or `svu...`)
*   **Password**: Your NUS Webmail password.

```bash
ssh <nus-id>@atlas9.nus.edu.sg
ssh <nus-id>@vanda.nus.edu.sg
ssh <nus-id>@hopper.nus.edu.sg
```

### 3.2 Web Portal
After your first login, wait approx. 1 hour for system sync. You can then access the graphical management interface:
*   [https://vanda.nus.edu.sg](https://vanda.nus.edu.sg)
*   [https://hopper.nus.edu.sg](https://hopper.nus.edu.sg)

### 3.3 VS Code Remote (Recommended)
For the best coding experience:
1.  Install the **"Remote - SSH"** extension in VS Code.
2.  Click the remote icon (bottom left) -> *Connect to Host* -> *Add New SSH Host*.
3.  Enter: `ssh <nus-id>@vanda.nus.edu.sg`.
4.  Enter your password when prompted. You can now edit files directly on the server.

---

## 4. Storage & Quotas

Storage management is critical. The system distinguishes between Personal quotas and Project quotas.

### 4.1 Check Quota (Compute & Storage)

*   **Check Compute Hours**:
    ```bash
    amgr login       # Logs into the resource manager
    hpc project      # Displays remaining CPU/GPU hours
    ```
    *Note: The first line shows your `personal` allowance. Lines below show the shared `Project` allowance.*

*   **Check Disk Space**:
    ```bash
    hpc space           # Checks your Home and Personal Scratch
    hpc space --project # Checks the Lab's shared Project storage
    ```

### 4.2 Storage Best Practices

| Directory | Path | Size Limit | Purpose |
| :--- | :--- | :--- | :--- |
| **Home** | `/home/svu/<nus-id>` | **Small (~40GB)** | Scripts, config files. **DO NOT store datasets here.** |
| **Scratch** | `/scratch/<nus-id>` | **Large (~500GB)** | **Conda Envs**, Datasets, Model Checkpoints. |
| **Project** | `/scratch/projects/...` | Very Large | Shared lab data. |

> **⚠️ Warning**: If your Home directory fills up (100%), you will be locked out of the cluster. Always use `/scratch` for large files.

---

## 5. Environment Management (Conda)

The system uses Environment Modules.

### 5.1 Setting up Conda
```bash
# 1. Load Miniconda
module load miniconda

# 2. Initialize (run once)
conda init bash
source ~/.bashrc

# 3. Prevent auto-activation of base (optional but recommended)
conda config --set auto_activate_base false
```

### 5.2 Creating Environments (Important)
By default, Conda installs environments in your Home directory, which fills up quickly. **It is highly recommended to install environments in Scratch.**

```bash
# Option A: Create env in a specific path (Recommended)
conda create --prefix /scratch/<nus-id>/envs/my_env python=3.9

# Activate the env using the path
conda activate /scratch/<nus-id>/envs/my_env
```

### 5.3 Loading Other Modules
```bash
module avail            # List all available software
module load cuda/12.1   # Load a specific version of CUDA
```

---

## 6. Job Submission (PBS)

**Strict Rule**: Do not run heavy training tasks on the Login Node. You must submit jobs to the Compute Nodes using PBS.

### 6.1 Interactive Job (Debug Mode)
Use this for debugging code or short tests. It gives you a shell on a compute node.

```bash
# Request 1 GPU, 1 CPU, for 2 hours
qsub -I -l select=1:ncpus=1:ngpus=1:mem=20gb -l walltime=02:00:00 -P <Project_ID>
```
*   `-I`: Interactive.
*   `-P`: Project ID (Required for Hopper; optional for Vanda personal quota).

### 6.2 Batch Job (Production Training)
Use this for long-running training. Create a script (e.g., `train.sh`).

```bash
#!/bin/bash
#PBS -N My_Job_Name
#PBS -P <Project_ID>
#PBS -q normal
#PBS -l walltime=12:00:00
#PBS -l select=1:ncpus=10:mpiprocs=1:ompthreads=10:mem=64gb:ngpus=1
#PBS -j oe
#PBS -o job_log.out

# 1. Load Environment
module load miniconda
conda activate /scratch/<nus-id>/envs/my_env

# 2. Move to working directory (PBS starts in home by default)
cd $PBS_O_WORKDIR

# 3. Run Code
echo "Starting job at $(date)"
python main.py --epochs 100 --batch_size 32
echo "Job finished at $(date)"
```

**Submit the script:**
```bash
qsub train.sh
```

---

## 7. Common Commands Cheat Sheet

| Action | Command | Note |
| :--- | :--- | :--- |
| **Submit Job** | `qsub script.sh` | Returns a Job ID (e.g., 12345.pbs) |
| **Check My Jobs** | `qstat` | Shows status (Q=Queued, R=Running, E=Exiting) |
| **Detailed Job Info** | `qstat -ans1` | Shows why a job is waiting or where it is running |
| **Delete Job** | `qdel <job_id>` | Stops a running or queued job |
| **Check GPU** | `nvidia-smi` | **Only works inside a compute node** |
| **Check Storage** | `hpc space` | Check if you are near the limit |

### Troubleshooting / Tips
1.  **"No Space Left on Device"**: Check your Home directory usage. Clean up `~/.cache` or move Conda environments to `/scratch`.
2.  **Job gets killed immediately**: Check if you requested enough memory (`mem=...`) or if your code has a syntax error (check the `.out` log file).
3.  **File Transfer**: Use `scp` or an SFTP client (like WinSCP/FileZilla) to upload data to `/scratch` before training.
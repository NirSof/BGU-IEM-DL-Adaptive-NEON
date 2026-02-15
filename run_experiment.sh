#!/bin/bash
#SBATCH --partition gpu
#SBATCH --time=7-00:00:00
#SBATCH --job-name neon_experiment
#SBATCH --output neon_experiment_output_%J.txt
#SBATCH --gpus=1
#SBATCH --nodes=1               
#SBATCH --ntasks-per-node=1
#SBATCH --mem=32G
#SBATCH --constraint="rtx_3090|rtx_4090|rtx_6000|rtx_pro_6000"

set -euo pipefail

# --- 1. Environment Setup ---
# Load Anaconda and activate the specific environment for the project
module load anaconda
source activate dna

echo "Starting Layer-Wise NEON Experiment..."

# --- 2. Execution ---
# Run the Grid Search logic distributed across the node.
# This command executes 'grid_search.py' with the defined parameter ranges.

python -m torch.distributed.run --standalone --nproc_per_node=1 \
  grid_search.py \
  --network_pkl_base "https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-cond-vp.pkl" \
  --network_pkl_aux_dir "training-runs/cifar10/ns6k" \
  --w_enc "0.75,1.0,1.25" \
  --w_dec "0.75,1.0,1.25" \
  --w_mid "0.75,1.0,1.25" \
  --seeds_test "0-1999" \
  --seeds "50000-99999" \
  --ref "https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/cifar10-32x32.npz" \
  --out_dir "results/cifar10/experiment_v1"

echo "Experiment Finished Successfully."

#!/bin/bash
#SBATCH --error=%j.err
#SBATCH --output=%j.log
#SBATCH --job-name=opti
#SBATCH --mail-type=ALL
#SBATCH --mail-user=YOUR_EMAIL
#SBATCH --time=03:00:00
#SBATCH --clusters=htls
#SBATCH --partition=htls_cm4
#SBATCH --reservation=htls_users
#SBATCH --get-user-env
#SBATCH --export=NONE
#SBATCH --ntasks=1
#SBATCH --ntasks-per-core=1

source ~/.conda_init

module load slurm_setup

conda activate myenv2

# passed argument is the gridlist (can be a subset of the simulated grid cells)
python run_optimization.py /dss/lxclscratch/0D/ge79vik2/constrained_opt_newthin/revision_runs_full_eu/ $1
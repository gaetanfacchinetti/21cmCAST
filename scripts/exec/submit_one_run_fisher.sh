#!/bin/bash
#
#SBATCH --job-name=FID0
#SBATCH --output=cluster_output_fid0.txt
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=00:12:00
#SBATCH --mem=40000
#
#SBATCH --mail-type=ALL
#SBATCH --mail-user=gaetan.facchinetti@ulb.be

module load releases/2021b
module load SciPy-bundle/2021.10-foss-2021b
module load GSL/2.7-GCC-11.2.0
module load Pillow/8.3.1-GCCcore-11.2.0
module load h5py/3.6.0-foss-2021b 
module load PyYAML/5.4.1-GCCcore-11.2.0

source ~/exo21cmFAST_release/bin/activate

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

python ./run_fisher.py ../runs/FID0/Config_L_X_0.001.config -nomp $SLURM_CPUS_PER_TASK -nruns 1 -rs 1993

#!/bin/bash
# ask for 16 cores on two nodes
#SBATCH --nodes=1 --ntasks-per-node=THREADS_DUMMY
#SBATCH --time=JOBRUNTIME_DUMMY
#SBATCH --job-name=NAME_DUMMY

module load intel/18

cd $SLURM_SUBMIT_DIR
/bin/hostname

export PYTHONPATH=/home/nsherck/SimPackages/simGit:$PYTHONPATH
export PYTHONPATH=/home/nsherck/SimPackages/simGit:$PYTHONPATH
export PATH=/home/nsherck/lammps-22Aug18/bin:$PATH

python CGModelScript_DUMMY
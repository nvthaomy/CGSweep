#!/bin/bash
#PBS -l nodes=1:ppn=THREADS_DUMMY
#PBS -l walltime=JOBRUNTIME_DUMMY
#PBS -V
#PBS -N NAME_DUMMY


cd $PBS_O_WORKDIR
#export PATH=/home/mnguyen/bin/lammps/lammps-12Dec18/bin/:$PATH
export PATH="/home/mnguyen/anaconda3/envs/py2/bin/:$PATH"
export PYTHONPATH=/home/mnguyen/bin/sim_git:$PYTHONPATH
python CGModelScript_DUMMY



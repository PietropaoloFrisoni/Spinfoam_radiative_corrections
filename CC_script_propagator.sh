#!/bin/bash
#SBATCH -A def-vidotto
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=6G
#SBATCH --time=0-6:00:00
#SBATCH --job-name=frogface_prop
#SBATCH --output=CLUSTER_COMPUTATIONS/frogface_prop.log
#SBATCH --error=CLUSTER_COMPUTATIONS/frogface_prop.err
#SBATCH --mail-type=BEGIN,FAIL,END
#SBATCH --mail-user=pfrisoni@uwo.ca


# folders

SL2CFOAM_DIR=/home/frisus95/projects/def-vidotto/frisus95/sl2cfoam_next_aggiornata
SL2CFOAM_DATA_DIR=${SL2CFOAM_DIR}/data_sl2cfoam
FASTWIG_TABLES_PATH=${SL2CFOAM_DIR}/data_sl2cfoam
JULIA_DIR=/home/frisus95/projects/def-vidotto/frisus95/julia-1.7.2
BASE_DIR=/home/frisus95/projects/def-vidotto/frisus95/Spinfoam_radiative_corrections
BOOSTER_DIR=${BASE_DIR}/booster_data

export LD_LIBRARY_PATH="${SL2CFOAM_DIR}/lib":$LD_LIBRARY_PATH
export JULIA_LOAD_PATH="${SL2CFOAM_DIR}/julia":$JULIA_LOAD_PATH

# parameters

CUTOFF=10
SHELL_MIN=0
SHELL_MAX=10
IMMIRZI=0.1
STORE_FOLDER=${BASE_DIR}/CLUSTER_COMPUTATIONS
BSPIN=0.5
CODE_TO_RUN=frogface_EPRL_propagator_matrix

echo "Running on: $SLURM_NODELIST"
echo

echo "Copying fastwig tables to: $SLURM_TMPDIR ..."
echo

cp ${FASTWIG_TABLES_PATH}/* $SLURM_TMPDIR/



echo "Extracting previous boosters to: $SLURM_TMPDIR ..."
echo

tar -xvf ${BOOSTER_DIR}/${CODE_TO_RUN}_SHELL_MIN_${SHELL_MIN}_SHELL_MAX_${SHELL_MAX}_IMMIRZI_${IMMIRZI}_CUTOFF_${CUTOFF}.tar.gz -C $SLURM_TMPDIR/



echo "Running: ${CODE_TO_RUN}"
echo

${JULIA_DIR}/bin/julia -p $SLURM_TASKS_PER_NODE --threads $SLURM_CPUS_PER_TASK ${BASE_DIR}/julia_codes/${CODE_TO_RUN}.jl $SLURM_TMPDIR ${CUTOFF} ${SHELL_MIN} ${SHELL_MAX} ${IMMIRZI} ${BSPIN} ${STORE_FOLDER}



echo "Compressing and copying computed boosters to ${BOOSTER_DIR}..."
echo

tar -czvf ${BOOSTER_DIR}/${CODE_TO_RUN}_SHELL_MIN_${SHELL_MIN}_SHELL_MAX_${SHELL_MAX}_IMMIRZI_${IMMIRZI}_CUTOFF_${CUTOFF}_BSPIN_${BSPIN}.tar.gz $SLURM_TMPDIR/vertex



echo "Completed"
echo

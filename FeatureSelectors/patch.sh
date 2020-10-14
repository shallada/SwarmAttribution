#!/bin/bash
source /home/smh0100/Environments/py3/bin/activate

RATIOS="0.0 0.1 0.3 0.5 0.7 0.9 1.0"
NRUNS=30
#ALGS="RandomMask ABCFeatureSelection GlowwormSwarmOptimization ParticleSwarmSelection AntSystemSelection EvolveFeatureSelection"
ALGS="ABCFeatureSelection"

DATA_SET=CASIS
WORK_DIR=/home/smh0100/SwarmAttribution/FeatureSelectors
WALL_TIME=40:00:00
MEMORY="128mb"

QSUB=qsub


ones_ratio=1.0
run=1
alg=ParticleSwarmSelection

echo "${WORK_DIR}/Algo.sh ABCFeatureSelection CASIS 0.5 16" | ${QSUB} -l "procs=1,mem=${MEMORY},walltime=${WALL_TIME}"
echo "${WORK_DIR}/Algo.sh ABCFeatureSelection CASIS 0.5 17" | ${QSUB} -l "procs=1,mem=${MEMORY},walltime=${WALL_TIME}"
echo "${WORK_DIR}/Algo.sh ABCFeatureSelection CASIS 0.9 5" | ${QSUB} -l "procs=1,mem=${MEMORY},walltime=${WALL_TIME}"
echo "${WORK_DIR}/Algo.sh AntSystemSelection CASIS 0.5 15" | ${QSUB} -l "procs=1,mem=${MEMORY},walltime=${WALL_TIME}"
echo "${WORK_DIR}/Algo.sh AntSystemSelection CASIS 0.5 16" | ${QSUB} -l "procs=1,mem=${MEMORY},walltime=${WALL_TIME}"
echo "${WORK_DIR}/Algo.sh ParticleSwarmSelection CASIS 0.5 16" | ${QSUB} -l "procs=1,mem=${MEMORY},walltime=${WALL_TIME}"
echo "${WORK_DIR}/Algo.sh ParticleSwarmSelection CASIS 0.5 17" | ${QSUB} -l "procs=1,mem=${MEMORY},walltime=${WALL_TIME}"
echo "${WORK_DIR}/Algo.sh ParticleSwarmSelection CASIS 0.5 18" | ${QSUB} -l "procs=1,mem=${MEMORY},walltime=${WALL_TIME}"
echo "${WORK_DIR}/Algo.sh GlowwormSwarmOptimization CASIS 0.5 16" | ${QSUB} -l "procs=1,mem=${MEMORY},walltime=${WALL_TIME}"
echo "${WORK_DIR}/Algo.sh GlowwormSwarmOptimization CASIS 0.5 17" | ${QSUB} -l "procs=1,mem=${MEMORY},walltime=${WALL_TIME}"
echo "${WORK_DIR}/Algo.sh GlowwormSwarmOptimization CASIS 0.5 18" | ${QSUB} -l "procs=1,mem=${MEMORY},walltime=${WALL_TIME}"
echo "${WORK_DIR}/Algo.sh GlowwormSwarmOptimization CASIS 1.0 1" | ${QSUB} -l "procs=1,mem=${MEMORY},walltime=${WALL_TIME}"

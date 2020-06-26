#!/bin/bash
source /home/smh0100/Environments/py3/bin/activate

RATIOS="0.0 0.1 0.3 0.5 0.7 0.9 1.0"
NRUNS=30
<<<<<<< HEAD
#ALGS="RandomMask ABCFeatureSelection GlowwormSwarmOptimization ParticleSwarmSelection AntSystemSelection EvolveFeatureSelection"
ALGS="RandomMask"

DATA_SET=CASIS
WORK_DIR=/home/smh0100/SwarmAttribution/FeatureSelectors
WALL_TIME=72:00:00
=======
ALGS="RandomMask ABCFeatureSelection GlowwormSwarmOptimization ParticleSwarmSelection AntSystemSelection EvolveFeatureSelection"

DATA_SET=CASIS
WORK_DIR=/home/smh0100/SwarmAttribution/FeatureSelectors
WALL_TIME=08:00:00
>>>>>>> bdce43335ca1d6dad7f604e12d7459b98f9e429d
MEMORY="128mb"

QSUB=qsub


for ones_ratio in $RATIOS
do
<<<<<<< HEAD
  for ((run=1; run<=$NRUNS; run++));
=======
  for ((run=1; run<=NRUNS; run++));
>>>>>>> bdce43335ca1d6dad7f604e12d7459b98f9e429d
  do
    for alg in $ALGS
    do
      echo "algorithm = ${alg}, dataset = ${DATA_SET}, ratio = $ones_ratio, run = $run"
      echo "${WORK_DIR}/Algo.sh ${alg} ${DATA_SET} ${ones_ratio} ${run}" | ${QSUB} -l "procs=1,mem=${MEMORY},walltime=${WALL_TIME}"
    done
  done
done

#/bin/bash
source /home/smh0100/Environments/py3/bin/activate
python --version
cd SwarmAttribution/FeatureSelectors
python ${1}.py $2 $3 $4

if [ "$#" -ne 1 ]; then
    echo "Missing argument: dataset subfolder name"
    exit 2
fi

rm -rf ../../Concatenator/$1/OpinionFinderWork
mkdir ../../Concatenator/$1/OpinionFinderWork


python CreateDocList.py $1

java -classpath ./opinionfinderv2.0/lib/weka.jar:./opinionfinderv2.0/lib/stanford-postagger.jar:./opinionfinderv2.0/opinionfinder.jar opin.main.RunOpinionFinder ../../Concatenator/$1/OpinionFinderWork/SentimentAnalysis.doclist -d -m opinionfinderv2.0/models/ -l opinionfinderv2.0/lexicons

rm -rf ../../Concatenator/$1/OpinionFinderWork/*.txt_auto_anns
mv ../../Datasets/$1/*.txt_auto_anns ../../Concatenator/$1/OpinionFinderWork

python DatasetCreator.py $1

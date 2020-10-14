if [ "$#" -ne 1 ]; then
    echo "Missing argument: dataset subfolder name"
    exit 2
fi

rm -rf ../../Concatenator/$1/MalletWork
mkdir ../../Concatenator/$1/MalletWork

mallet-2.0.8/bin/mallet import-dir --input ../../Datasets/$1 --output ../../Concatenator/$1/MalletWork/TopicModeling.mallet --keep-sequence --remove-stopwords

mallet-2.0.8/bin/mallet train-topics  --input ../../Concatenator/$1/MalletWork/TopicModeling.mallet --num-topics 20 --output-doc-topics ../../Concatenator/$1/MalletWork/Topics.mallet

python ExtractFeatures.py CASIS

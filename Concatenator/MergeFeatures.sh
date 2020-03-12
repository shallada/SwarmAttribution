if [ "$#" -ne 1 ]; then
    echo "Missing argument: dataset subfolder name"
    exit 2
fi

python MergeFeatures.py $1


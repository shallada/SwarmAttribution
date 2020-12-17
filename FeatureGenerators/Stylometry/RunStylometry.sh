if [ "$#" -ne 1 ]; then
    echo "Missing argument: dataset subfolder name"
    exit 2
fi

python main.py $1

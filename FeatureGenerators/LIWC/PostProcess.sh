if [ "$#" -ne 1 ]; then
    echo "Missing argument: dataset subfolder name"
    exit 2
fi

# Remove the header line and sort
tail -n +2 ../../Concatenator/$1/LIWCFeatures.txt | sort | tr "\\t" "," > tmp
rm ../../Concatenator/$1/LIWCFeatures.txt
mv tmp ../../Concatenator/$1/LIWCFeatures.txt


#!/bin/bash

for FILE in $(ls | grep "du5_to_s3.e")
do
    if [ -s $FILE ]
    then
        echo "$FILE is not empty !!!"
    else
        echo "$FILE is empty"
    fi
done

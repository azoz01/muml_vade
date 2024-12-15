#!/bin/bash
mkdir data
# HAR
cd data
mkdir har
wget -O har.zip https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip
mkdir tmp
unzip -d tmp/ har.zip
unzip -d "tmp/UCI HAR Dataset" "tmp/UCI HAR Dataset.zip"
mv "tmp/UCI HAR Dataset/UCI HAR Dataset/train/X_train.txt" "har/X_train.csv"
mv "tmp/UCI HAR Dataset/UCI HAR Dataset/test/X_test.txt" "har/X_test.csv"
mv "tmp/UCI HAR Dataset/UCI HAR Dataset/train/y_train.txt" "har/y_train.csv"
mv "tmp/UCI HAR Dataset/UCI HAR Dataset/test/y_test.txt" "har/y_test.csv"
rm -r tmp har.zip
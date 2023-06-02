#!/bin/bash

root=results
drive_link=https://drive.google.com/file/d/14xYRVCJbhxhkGR85JzizXn0Me-mMgEKa/view?usp=sharing

if [ ! -f $root/models/xnli/best/config.json ]
then
    echo "Downloading results file.."
    python src/drive_download.py $drive_link
    unzip results.zip
    rm results.zip
    echo "Done."
else
    echo "Results folder found, with content. Skipping download."
fi
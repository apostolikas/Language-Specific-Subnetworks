#!/bin/bash

root=results
drive_link=https://drive.google.com/file/d/1AUcMuQZkXixQoqN92ZopumoATcXF39pc/view?usp=drive_link

if [ ! -f $root/models/xnli/best/config.json ]
then
    echo "Downloading results file.."
    python src/drive_download.py $drive_link
    tar -xzvf results.tar.gz
    rm results.tar.gz
    echo "Done."
else
    echo "Results folder found, with content. Skipping download."
fi
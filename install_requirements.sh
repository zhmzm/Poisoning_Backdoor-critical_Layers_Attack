#!/bin/sh
echo "begin Install>>>>>>>>>>>>>>>>>>>>"


dqtime=$(date "+%Y-%m-%d %H:%M:%S")


echo "datetime: ${dqtime}"

echo "Install pytorch"
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge

echo "Install sklearn"
conda install scikit-learn

echo "Install opencv"
pip install opencv-python

echo "Install sk-image"
conda install Scikit-Image

echo "Install matplotlib"
conda install matplotlib

echo "Install hdbscan"
pip install hdbscan

echo "Install tqdm"
conda install tqdm


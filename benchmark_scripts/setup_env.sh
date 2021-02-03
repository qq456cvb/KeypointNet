#!/bin/bash

pwd=$(pwd)
conda env create -f env.yml

export CUDA_HOME=/usr/local/cuda

cd $pwd/models/pointnet2_utils/custom_ops
python setup.py install

cd $pwd/models/rscnn_utils/custom_ops
python setup.py install
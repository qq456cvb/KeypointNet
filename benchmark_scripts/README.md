## Run Benchmark
To train/test saliency/correspondence estimation with specific network, you can type e.g.

``
python train.py task=saliency network=pointnet
``

``
python test.py task=correspondence network=rscnn
``

Note that for PointNet++/PointNet2 and RSCNN networks, custom ops must be built by running ``python setup.py install`` under `[pointnet2|rscnn]_utils/custom_ops` folder.

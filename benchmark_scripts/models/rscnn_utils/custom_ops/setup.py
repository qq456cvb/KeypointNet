import glob
import os.path as osp

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

_ext_src_root = osp.join('src_root')
_ext_sources = glob.glob(osp.join(_ext_src_root, "csrc", "*.cpp")) + glob.glob(
    osp.join(_ext_src_root, "csrc", "*.cu")
)
_ext_headers = glob.glob(osp.join(_ext_src_root, "cinclude", "*"))

setup(
    name='rscnn_ops',
    ext_modules=[
        CUDAExtension('rscnn_ops._ext', _ext_sources, include_dirs=[osp.join(_ext_src_root, "cinclude")])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
import glob
import os
import os.path as osp

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# this_dir = osp.dirname(osp.abspath(__file__))
_ext_src_root = osp.join('src_root')
_ext_sources = glob.glob(osp.join(_ext_src_root, "csrc", "*.cpp")) + glob.glob(
    osp.join(_ext_src_root, "csrc", "*.cu")
)

# _ext_headers = glob.glob(osp.join(_ext_src_root, "cinclude", "*"))

setup(
    name='rscnn_ops',
    ext_modules=[
        CUDAExtension('rscnn_ops_ext', _ext_sources, include_dirs=[osp.join(_ext_src_root, "cinclude")])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
# requirements = ["torch>=1.4"]

# # exec(open(osp.join("pointnet2_ops", "_version.py")).read())

# os.environ["TORCH_CUDA_ARCH_LIST"] = "3.7+PTX;5.0;6.0;6.1;6.2;7.0;7.5"
# setup(
#     name="rscnn_ops",
#     version='0.0.1',
#     author="Erik Wijmans",
#     packages=find_packages(),
#     install_requires=requirements,
#     ext_modules=[
#         CUDAExtension(
#             name="rscnn_ops_ext",
#             sources=_ext_sources,
#             extra_compile_args={
#                 "cxx": ["-O3"],
#                 "nvcc": ["-O3", "-Xfatbin", "-compress-all"],
#             },
#             include_dirs=[osp.join(this_dir, _ext_src_root, "cinclude")],
#         )
#     ],
#     cmdclass={"build_ext": BuildExtension},
#     include_package_data=True,
# )
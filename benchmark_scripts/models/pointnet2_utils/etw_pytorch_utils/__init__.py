from __future__ import division, absolute_import, with_statement, print_function, unicode_literals
__version__ = '1.1.1'

try:
    __ETW_PT_UTILS_SETUP__
except:
    __ETW_PT_UTILS_SETUP__ = False

if not __ETW_PT_UTILS_SETUP__:
    from .pytorch_utils import *
    from .persistent_dataloader import DataLoader
    from .viz import *
    from .seq import Seq

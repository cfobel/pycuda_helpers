from __future__ import division

from jinja2 import FileSystemLoader, Environment
from path import path
import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule


dtype_map = { np.dtype('float32'): 'float', np.dtype('int32'): 'int32_t'}


def package_root():
    '''
    Return absolute path to scatter_gather package root directory.
    '''
    try:
        script = path(__file__)
    except NameError:
        import sys
        script = path(sys.argv[0])
    return script.parent.abspath()


def get_include_root():
    return package_root().joinpath('pycuda_include')


def get_template_root():
    return package_root().joinpath('pycuda_templates')


def get_template_loader():
    return FileSystemLoader(get_template_root())


def log2ceil(x):
    return int(np.ceil(np.log2(x)))



jinja_env = Environment(loader=get_template_loader())


def get_cuda_function(template_name, function_namebase, dtype=None,
        template_params=None, supported_types=None, keep=False):
    if supported_types is None:
        supported_types = dtype_map.keys()
    if template_params is None:
        template_params = {}

    code_template = jinja_env.get_template(template_name)
    mod = SourceModule(code_template.render(template_params), no_extern_c=True,
            keep=keep, include_dirs=[get_include_root(),
                    '/home/christian/Documents/dev/fpga_placement/VPR_v5/VPR_HET/pyvpr/pycuda_include',])

    if dtype is None:
        func_name = function_namebase
    else:
        assert(dtype in supported_types)
        func_name = '%s_%s' % (function_namebase, dtype_map[dtype])
    try:
        func = mod.get_function(func_name)
    except cuda.LogicError:
        print dtype, func_name
        raise
    return func

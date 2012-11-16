import sys

from path import path


def module_root():
    '''
    Return absolute path to pyvpr root directory.
    '''
    try:
        script = path(__file__)
    except NameError:
        script = path(sys.argv[0])
    return script.parent.abspath()


def get_include_root():
    return module_root().joinpath('pycuda_include')


def get_template_root():
    return module_root().joinpath('pycuda_templates')

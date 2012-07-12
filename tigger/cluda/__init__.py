from tigger.cluda import dtypes
import os.path
from mako.template import Template
from mako import exceptions

_PRELUDE = open(os.path.join(os.path.split(__file__)[0], 'prelude.cluda.mako')).read()
_MUL = open(os.path.join(os.path.split(__file__)[0], 'mul.cluda.mako')).read()

class MulCollector:

    def __init__(self):
        self.functions = {}

    def __call__(self, dtype1, dtype2, out_dtype=None):
        if out_dtype is None:
            out_dtype = numpy.result_type(dtype1, dtype2)
        ctypes = [dtypes.ctype(dt) for dt in (dtype1, dtype2)]
        out_ctype = dtypes.ctype(out_dtype)

        name = '_mul_' + '_'.join(ctypes) + '__' + out_ctype

        self.functions[name] = (dtype1, dtype2, out_dtype)
        return name


def render(template, env, **kwds):
    prelude = Template(_PRELUDE).render(env=env.params)
    mul_c = MulCollector()

    try:
        src = template.render(env=env.params, dtypes=dtypes, mul=mul_c, **kwds)
    except:
        # TODO: output to stderr?
        print exceptions.text_error_template().render()
        raise Exception("Template rendering failed")

    muls = Template(_MUL).render(dtypes=dtypes, mul_functions=mul_c.functions)
    return prelude + muls + src

def supportsCuda():
    try:
        import pycuda.driver
    except ImportError:
        return False

    return True

def supportsOcl():
    try:
        import pyopencl
    except ImportError:
        return False

    return True

def createCuda(*args, **kwds):
    from tigger.cluda.cuda import CudaEnvironment
    return CudaEnvironment(*args, **kwds)

def createOcl(*args, **kwds):
    from tigger.cluda.ocl import OclEnvironment
    return OclEnvironment(*args, **kwds)

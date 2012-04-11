from tigger.cluda import helpers
import os.path
from mako.template import Template

_PRELUDE = open(os.path.join(os.path.split(__file__)[0], 'prelude.cu.mako')).read()

def render(template_str, env, **kwds):
    return Template(_PRELUDE + template_str).render(env=env.params, helpers=helpers, **kwds)

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

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

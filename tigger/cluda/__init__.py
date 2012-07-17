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

def createContext(api, *args, **kwds):
    if api == 'cuda':
        try:
            from tigger.cluda.cuda import CudaContext
            return CudaContext(*args, **kwds)
        except:
            raise Exception("Cuda context is not available on this system")
    elif api == 'ocl':
        try:
            from tigger.cluda.ocl import OclContext
            return OclContext(*args, **kwds)
        except:
            raise Exception("OpenCL context is not available on this system")
    else:
        raise Exception("Unrecognized API: " + str(api))

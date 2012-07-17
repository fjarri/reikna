API_CUDA = 'cuda'
API_OCL = 'ocl'


def supports_api(api):
    """
    Checks if given GPU API is available.
    """
    if api == API_CUDA:
        try:
            import pycuda.driver
        except ImportError:
            return False
        return True
    elif api == API_OCL:
        try:
            import pyopencl
        except ImportError:
            return False
        return True
    else:
        raise Exception("Unrecognized API: " + str(api))

def create_context(api, *args, **kwds):
    """
    Creates CLUDA context for given API.
    """
    if api == API_CUDA:
        try:
            from tigger.cluda.cuda import CudaContext
            return CudaContext.create(*args, **kwds)
        except ImportError:
            raise Exception("Cuda context is not available on this system")
    elif api == API_OCL:
        try:
            from tigger.cluda.ocl import OclContext
            return OclContext.create(*args, **kwds)
        except ImportError:
            raise Exception("OpenCL context is not available on this system")
    else:
        raise Exception("Unrecognized API: " + str(api))

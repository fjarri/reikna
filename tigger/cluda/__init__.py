def supports_api(api):
    """
    Checks if given GPU API is available.
    """
    if api == 'cuda':
        try:
            import pycuda.driver
        except ImportError:
            return False
        return True
    elif api == 'ocl':
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
    if api == 'cuda':
        try:
            from tigger.cluda.cuda import CudaContext
            return CudaContext(*args, **kwds)
        except ImportError:
            raise Exception("Cuda context is not available on this system")
    elif api == 'ocl':
        try:
            from tigger.cluda.ocl import OclContext
            return OclContext(*args, **kwds)
        except ImportError:
            raise Exception("OpenCL context is not available on this system")
    else:
        raise Exception("Unrecognized API: " + str(api))

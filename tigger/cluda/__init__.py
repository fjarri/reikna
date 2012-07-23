#: Identifier for the PyCUDA-based API
API_CUDA = 'cuda'

#: Identifier for the PyOpenCL-based API
API_OCL = 'ocl'

#: List of identifiers for all known (not necessarily available for the current system) APIs
APIS = [API_CUDA, API_OCL]

def supports_api(api_id):
    """
    Returns ``True`` if given API is supported.
    """
    try:
        api(api_id)
    except ImportError:
        return False

    return True

def supported_apis():
    """
    Returns list of identifiers of supported APIs.
    """
    return [api_id for api_id in APIS if supports_api(api_id)]

def api(api_id):
    """
    Returns API module with the generalized interface :py:mod:`tigger.cluda.api`
    for the given identifier.
    """
    if api_id == API_CUDA:
        import tigger.cluda.cuda
        return tigger.cluda.cuda
    elif api_id == API_OCL:
        import tigger.cluda.ocl
        return tigger.cluda.ocl
    else:
        raise ValueError("Unrecognized API: " + str(api_id))

def cuda_api():
    """
    Returns API module for CUDA.
    """
    return api(API_CUDA)

def ocl_api():
    """
    Returns API module for OpenCL.
    """
    return api(API_OCL)

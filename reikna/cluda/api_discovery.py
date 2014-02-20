"""
This module contains functions for API discovery.
"""


def cuda_id():
    """Returns the identifier of the ``PyCUDA``-based API."""
    return 'cuda'

def ocl_id():
    """Returns the identifier of the ``PyOpenCL``-based API."""
    return 'ocl'

def api_ids():
    """
    Returns a list of identifiers for all known
    (not necessarily available for the current system) APIs.
    """
    return [ocl_id(), cuda_id()]


def supports_api(api_id):
    """
    Returns ``True`` if given API is supported.
    """
    try:
        get_api(api_id)
    except ImportError:
        return False

    return True


def supported_api_ids():
    """
    Returns a list of identifiers of supported APIs.
    """
    return [api_id for api_id in api_ids() if supports_api(api_id)]


def get_api(api_id):
    """
    Returns an API module with the generalized interface :py:mod:`reikna.cluda.api`
    for the given identifier.
    """
    if api_id == cuda_id():
        import reikna.cluda.cuda
        return reikna.cluda.cuda
    elif api_id == ocl_id():
        import reikna.cluda.ocl
        return reikna.cluda.ocl
    else:
        raise ValueError("Unrecognized API: " + str(api_id))


def cuda_api():
    """
    Returns the ``PyCUDA``-based API module.
    """
    return get_api(cuda_id())


def ocl_api():
    """
    Returns the ``PyOpenCL``-based API module.
    """
    return get_api(ocl_id())


def any_api():
    """
    Returns one of the API modules supported by the system or raises an ``Exception``
    if there are not any.
    """
    api_ids = supported_api_ids()
    if len(api_ids) == 0:
        raise Exception("No supported GPGPU APIs found")
    return get_api(api_ids[0])

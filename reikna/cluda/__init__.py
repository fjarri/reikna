from reikna.cluda.api_discovery import cuda_id, ocl_id, api_ids, supports_api, \
    supported_api_ids, get_api, cuda_api, ocl_api, any_api
from reikna.cluda.api_tools import find_devices
from reikna.cluda.kernel import Module, Snippet


class OutOfResourcesError(Exception):
    """
    Thrown by :py:meth:`~reikna.cluda.api.Thread.compile_static`
    if the provided ``local_size`` is too big, or one cannot be found.
    """
    pass

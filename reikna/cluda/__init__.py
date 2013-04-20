from reikna.cluda.api_discovery import *
from reikna.cluda.kernel import Module


class OutOfResourcesError(Exception):
    """
    Thrown by :py:meth:`~reikna.cluda.api.Context.compile_static`
    if the provided ``local_size`` is too big, or one cannot be found.
    """
    pass

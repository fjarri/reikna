from itertools import product
import gc, re

import numpy


def get_apis(config):
    """
    Get list of APIs to test, based on command line options and their availability.
    """
    # if we import it in the header, it messes up with coverage results
    import reikna.cluda as cluda

    conf_api_id = config.option.api

    if conf_api_id == "supported":
        api_ids = cluda.supported_api_ids()
    else:
        if not cluda.supports_api(conf_api_id):
            raise Exception("Requested API " + conf_api_id + " is not supported.")
        api_ids = [conf_api_id]
    apis = [cluda.get_api(api_id) for api_id in api_ids]
    return apis, api_ids


class ThreadParams:
    """
    Encapsulates a set of parameters necessary to create a test Thread.
    """

    def __init__(self, api, pnum, dnum):
        self.api_id = api.get_id()
        self.pnum = pnum
        self.dnum = dnum

        self._api = api

        platform = api.get_platforms()[pnum]
        self._device = platform.get_devices()[dnum]

        platform_name = platform.name
        device_name = self._device.name

        self.device_params = api.DeviceParameters(self._device)

        self.device_id = "{api},{pnum},{dnum}".format(api=api.get_id(), pnum=pnum, dnum=dnum)
        self.device_full_name = platform_name + ", " + device_name

        self.id = self.device_id

        # if we import it in the header, it messes up with coverage results
        import reikna.cluda as cluda

        self.cuda = (api.get_id() == cluda.cuda_id())

    def create_thread(self):
        return self._api.Thread(self._device)

    def __key(self):
        return (self.api_id, self.pnum, self.dnum)

    def __eq__(x, y):
        return x.__key() == y.__key()

    def __hash__(self):
        return hash(self.__key())

    def __str__(self):
        return self.id


class ThreadManager:
    """
    Caches Thread instances and handles push/pop mechanics of CUDA Threads properly.
    """

    def __init__(self):
        self._threads = {}

    def _create(self, thread_params):

        thread = thread_params.create_thread()

        # The context is in the stack right after creation, need to pop it.
        if thread_params.cuda:
            thread._cuda_pop()

        self._threads[thread_params] = thread

    def get(self, thread_params):

        if thread_params not in self._threads:
            self._create(thread_params)

        thread = self._threads[thread_params]
        if thread_params.cuda:
            thread._cuda_push()

        return thread

    def release(self, thread_params):
        thread = self._threads[thread_params]
        if thread_params.cuda:
            thread._cuda_pop()

_thread_manager = ThreadManager()


def get_threads(config):
    """
    Create a list of thread creators, based on command line options and their availability.
    """
    # if we import it in the header, it messes up with coverage results
    from reikna.cluda import find_devices

    apis, _ = get_apis(config)

    tps = []
    for api in apis:
        devices = find_devices(
            api,
            include_devices=config.option.device_include_mask,
            exclude_devices=config.option.device_exclude_mask,
            include_platforms=config.option.platform_include_mask,
            exclude_platforms=config.option.platform_exclude_mask,
            include_duplicate_devices=config.option.include_duplicate_devices)

        for pnum in sorted(devices.keys()):
            dnums = sorted(devices[pnum])
            for dnum in dnums:
                tps.append(ThreadParams(api, pnum, dnum))

    return tps, [str(tp) for tp in tps]


def pair_thread_with_doubles(metafunc, tp):
    d = metafunc.config.option.double
    ds = lambda dv: 'dp' if dv else 'sp'

    vals = []
    ids = []
    if d == "supported":
        for dv in [False, True]:
            if not dv or tp.device_params.supports_dtype(numpy.float64):
                vals.append((dv,))
                ids.append(ds(dv))
    else:
        dv = d == 'yes'
        vals.append((dv,))
        ids.append(ds(dv))

    return [tp] * len(vals), vals, ids


def get_thread_tuples(metafunc, get_remainders):
    tps, tp_ids = get_threads(metafunc.config)
    tuples = []
    ids = []
    for tp, tp_id in zip(tps, tp_ids):
        new_tps, remainders, rem_ids = get_remainders(metafunc, tp)
        for new_tp, remainder, rem_id in zip(new_tps, remainders, rem_ids):
            tuples.append((new_tp,) + remainder if isinstance(remainder, tuple) else (remainder,))
            ids.append(tp_id + "," + rem_id)

    # For tuples of 1 element (i.e. the thread itself), just use this element as a parameter
    tuples = [t[0] if len(t) == 1 else t for t in tuples]

    return tuples, ids


def create_thread_in_tuple(request):
    """
    Instantiate thread from the tuple of test parameters and create a corresponding finalizer.
    """
    params = request.param
    if isinstance(params, tuple):
        tp = params[0]
        remainder = tuple(params[1:])
    else:
        tp = params
        remainder = tuple()

    thr = _thread_manager.get(tp)

    def finalizer():
        # Py.Test holds the reference to the created funcarg/fixture,
        # which interferes with ``__del__`` functionality.
        # This method forcefully frees critical resources explicitly
        # (rendering the object unusable).
        _thread_manager.release(tp)
        # just in case there is some stuff left
        gc.collect()

    request.addfinalizer(finalizer)

    if isinstance(params, tuple):
        return (thr,) + remainder
    else:
        return thr


def parametrize_thread_tuple(metafunc, name, get_remainders):
    tuples, tuple_ids = get_thread_tuples(metafunc, get_remainders)
    metafunc.parametrize(name, tuples, ids=tuple_ids, indirect=True)

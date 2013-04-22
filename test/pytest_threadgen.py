from itertools import product
import gc, re

import numpy

from reikna.cluda import find_devices


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


def get_threads(config, vary_fast_math=False):
    """
    Create a list of thread creators, based on command line options and their availability.
    """

    class ThrCreator:
        def __init__(self, api, pnum, dnum, fast_math=None):
            platform = api.get_platforms()[pnum]
            device = platform.get_devices()[dnum]

            fm_suffix = {True:",fm", False:",nofm", None:""}[fast_math]
            self.device_id = api.get_id() + "," + str(pnum) + "," + str(dnum)
            self.platform_name = platform.name
            self.device_name = device.name
            self.id = self.device_id + fm_suffix

            kwds = dict(device=device)
            if fast_math is not None:
                kwds['fast_math'] = fast_math

            self.create = lambda: api.Thread.create(**kwds)

            thr = self.create()
            self.supports_double = thr.supports_dtype(numpy.float64)

        def __call__(self):
            return self.create()

        def __str__(self):
            return self.id

    apis, _ = get_apis(config)

    if vary_fast_math:
        fm = config.option.fast_math
        fms = dict(both=[False, True], no=[False], yes=[True])[fm]
    else:
        fms = [None]

    tcs = []
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
                for fm in fms:
                    tcs.append(ThrCreator(api, pnum, dnum, fast_math=fm))

    return tcs, [str(tc) for tc in tcs]


def pair_thread_with_doubles(metafunc, tc):
    d = metafunc.config.option.double
    ds = lambda dv: 'dp' if dv else 'sp'

    vals = []
    ids = []
    if d == "supported":
        for dv in [False, True]:
            if not dv or tc().supports_dtype(numpy.float64):
                vals.append((dv,))
                ids.append(ds(dv))
    else:
        dv = d == 'yes'
        vals.append((dv,))
        ids.append(ds(dv))

    return [tc] * len(vals), vals, ids


def get_thread_tuples(metafunc, get_remainders):
    tcs, tc_ids = get_threads(metafunc.config, vary_fast_math=True)
    tuples = []
    ids = []
    for tc, tc_id in zip(tcs, tc_ids):
        new_tcs, remainders, rem_ids = get_remainders(metafunc, tc)
        for new_tc, remainder, rem_id in zip(new_tcs, remainders, rem_ids):
            tuples.append((new_tc,) + remainder if isinstance(remainder, tuple) else (remainder,))
            ids.append(tc_id + "," + rem_id)

    # For tuples of 1 element (i.e. the thread itself), just use this element as a parameter
    tuples = [t[0] if len(t) == 1 else t for t in tuples]

    return tuples, ids


def create_thread_in_tuple(request):
    """
    Instantiate thread from the tuple of test parameters and create a corresponding finalizer.
    """
    params = request.param
    if isinstance(params, tuple):
        tc = params[0]
        remainder = tuple(params[1:])
    else:
        tc = params
        remainder = tuple()

    thr = tc()

    def finalizer():
        # Py.Test won't release the reference to thr, so we need to finalize it explicitly.
        thr._pytest_finalize()
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

from itertools import product
import gc, re

import numpy


def get_apis(config):
    """
    Get list of APIs to test, based on command line options and their availability.
    """
    # if we import it in the header, it messes up with coverage results
    import tigger.cluda as cluda

    conf_api_id = config.option.api

    if conf_api_id == "supported":
        api_ids = cluda.supported_apis()
    else:
        if not cluda.supports_api(conf_api_id):
            raise Exception("Requested API " + conf_api_id + " is not supported.")
        api_ids = [conf_api_id]
    apis = [cluda.api(api_id) for api_id in api_ids]
    return apis, api_ids


def get_contexts(config, vary_fast_math=False):
    """
    Create a list of context creators, based on command line options and their availability.
    """

    class CtxCreator:
        def __init__(self, api, pnum, dnum, fast_math=None):
            platform = api.get_platforms()[pnum]
            device = platform.get_devices()[dnum]

            fm_suffix = {True:",fm", False:",nofm", None:""}[fast_math]
            self.device_id = api.API_ID + "," + str(pnum) + "," + str(dnum)
            self.platform_name = platform.name
            self.device_name = device.name
            self.id = self.device_id + fm_suffix

            kwds = dict(device=device)
            if fast_math is not None:
                kwds['fast_math'] = fast_math

            self.create = lambda: api.Context.create(**kwds)

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

    include_devices = config.option.device_include_mask
    exclude_devices = config.option.device_exclude_mask
    include_platforms = config.option.platform_include_mask
    exclude_platforms = config.option.platform_exclude_mask

    def name_matches_masks(name, includes, excludes):
        if len(includes) > 0:
            for include in includes:
                if re.search(include, name):
                    break
            else:
                return False

        if len(excludes) > 0:
            for exclude in excludes:
                if re.search(exclude, name):
                    return False

        return True

    ccs = []
    seen_devices = set()
    for api in apis:
        for pnum, platform in enumerate(api.get_platforms()):

            seen_devices.clear()

            if not name_matches_masks(platform.name, include_platforms, exclude_platforms):
                continue

            for dnum, device in enumerate(platform.get_devices()):
                if not name_matches_masks(device.name, include_devices, exclude_devices):
                    continue

                if (not config.option.include_duplicate_devices and
                        device.name in seen_devices):
                    continue

                seen_devices.add(device.name)

                for fm in fms:
                    ccs.append(CtxCreator(api, pnum, dnum, fast_math=fm))

    return ccs, [str(cc) for cc in ccs]


def pair_context_with_doubles(metafunc, cc):
    d = metafunc.config.option.double
    ds = lambda dv: 'dp' if dv else 'sp'

    vals = []
    ids = []
    if d == "supported":
        for dv in [False, True]:
            if not dv or cc().supports_dtype(numpy.float64):
                vals.append((dv,))
                ids.append(ds(dv))
    else:
        dv = d == 'yes'
        vals.append((dv,))
        ids.append(ds(dv))

    return [cc] * len(vals), vals, ids


def get_context_tuples(metafunc, get_remainders):
    ccs, cc_ids = get_contexts(metafunc.config, vary_fast_math=True)
    tuples = []
    ids = []
    for cc, cc_id in zip(ccs, cc_ids):
        new_ccs, remainders, rem_ids = get_remainders(metafunc, cc)
        for new_cc, remainder, rem_id in zip(new_ccs, remainders, rem_ids):
            tuples.append((new_cc,) + remainder if isinstance(remainder, tuple) else (remainder,))
            ids.append(cc_id + "," + rem_id)

    # For tuples of 1 element (i.e. the context itself), just use this element as a parameter
    tuples = [t[0] if len(t) == 1 else t for t in tuples]

    return tuples, ids


def create_context_in_tuple(request):
    """
    Instantiate context from the tuple of test parameters and create a corresponding finalizer.
    """
    params = request.param
    if isinstance(params, tuple):
        cc = params[0]
        remainder = tuple(params[1:])
    else:
        cc = params
        remainder = tuple()

    ctx = cc()
    def finalizer():
        ctx.release()
        gc.collect()
    request.addfinalizer(finalizer)

    if isinstance(params, tuple):
        return (ctx,) + remainder
    else:
        return ctx


def parametrize_context_tuple(metafunc, name, get_remainders):
    tuples, tuple_ids = get_context_tuples(metafunc, get_remainders)
    metafunc.parametrize(name, tuples, ids=tuple_ids, indirect=True)

from itertools import product
import numpy


def get_apis(metafunc):
    """
    Get list of APIs to test, based on command line options and their availability.
    """
    # if we import it in the header, it messes up with coverage results
    import tigger.cluda as cluda

    conf_api_id = metafunc.config.option.api

    if conf_api_id == "supported":
        api_ids = cluda.supported_apis()
    else:
        if not cluda.supports_api(conf_api_id):
            raise Exception("Requested API " + conf_api_id + " is not supported.")
        api_ids = [conf_api_id]
    apis = [cluda.api(api_id) for api_id in api_ids]
    return apis, api_ids


def get_contexts(metafunc, vary_fast_math=False):
    """
    Create a list of context creators, based on command line options and their availability.
    """

    class CtxCreator:
        def __init__(self, api, fast_math=None):
            self.fast_math = fast_math
            self.api_id = api.API_ID
            kwds = {} if fast_math is None else {'fast_math':fast_math}
            self.create = lambda: api.Context.create(**kwds)

        def __call__(self):
            return self.create()

        def __str__(self):
            fm_suffix = {True:",fm", False:",nofm", None:""}[self.fast_math]
            return self.api_id + fm_suffix

    apis, _ = get_apis(metafunc)

    if vary_fast_math:
        fm = metafunc.config.option.fast_math
        fms = dict(both=[False, True], no=[False], yes=[True])[fm]
        ccs = [CtxCreator(api, fast_math=fm) for api, fm in product(apis, fms)]
    else:
        ccs = [CtxCreator(api) for api in apis]

    return ccs, [str(cc) for cc in ccs]


def pair_context_with_doubles(metafunc, cc):
    d = metafunc.config.option.double
    ds = lambda dv: 'dp' if dv else 'sp'

    vals = []
    ids = []
    if d == "supported":
        for dv in [False, True]:
            if not dv or cc().supports_dtype(numpy.float64):
                vals.append(dv)
                ids.append(ds(dv))
    else:
        dv = d == 'yes'
        vals.append(dv)
        ids.append(ds(dv))

    return [cc] * len(vals), vals, ids


def get_context_tuples(metafunc, get_remainders):
    ccs, cc_ids = get_contexts(metafunc, vary_fast_math=True)
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
    request.addfinalizer(lambda: ctx.release())

    if isinstance(params, tuple):
        return (ctx,) + remainder
    else:
        return ctx


def parametrize_context_tuple(metafunc, name, get_remainders):
    tuples, tuple_ids = get_context_tuples(metafunc, get_remainders)
    metafunc.parametrize(name, tuples, ids=tuple_ids, indirect=True)

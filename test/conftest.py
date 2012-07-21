from itertools import product

import numpy

pytest_plugins = ['pytest_returnvalues']


def pytest_addoption(parser):
    parser.addoption("--api", action="store",
        help="API: cuda/ocl/supported",
        # can't get API list from CLUDA, because if we import it here,
        # it messes up with coverage results
        # (modules get imported before coverage collector starts)
        default="supported", choices=["cuda", "ocl", "supported"])
    parser.addoption("--double", action="store",
        help="Use doubles: no/yes/supported",
        default="supported", choices=["no", "yes", "supported"])
    parser.addoption("--fast-math", dest="fast_math", action="store",
        help="Use fast math: no/yes/both",
        default="yes", choices=["no", "yes", "both"])


def pytest_funcarg__ctx_and_double(request):
    """
    Create context before call to test and release it when the test ends
    """
    cc, dv = request.param
    ctx = cc()
    request.addfinalizer(lambda: ctx.release())
    return ctx, dv

def pytest_funcarg__ctx(request):
    """
    Create context before call to test and release it when the test ends.
    """
    cc = request.param
    ctx = cc()
    request.addfinalizer(lambda: ctx.release())
    return ctx

def pytest_funcarg__some_ctx(request):
    """
    Create context before call to test and release it when the test ends.
    """
    cc = request.param
    ctx = cc()
    request.addfinalizer(lambda: ctx.release())
    return ctx


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

def get_contexts_and_doubles(metafunc):
    """
    Create a list of (context creator, use-double) pairs
    and corresponding ids to parameterize tests.
    """
    ccs, cc_ids = get_contexts(metafunc, vary_fast_math=True)

    d = metafunc.config.option.double
    ds = lambda dv: 'dp' if dv else 'sp'

    vals = []
    ids = []
    if d == "supported":
        for cc_and_id, dv in product(zip(ccs, cc_ids), [False, True]):
            cc, cc_id = cc_and_id
            if not dv or cc().supports_dtype(numpy.float64):
                vals.append((cc, dv))
                ids.append(cc_id + "," + ds(dv))
    else:
        dv = d == 'yes'
        for cc, cc_id in zip(ccs, cc_ids):
            vals.append((cc, dv))
            ids.append(cc_id + "," + ds(dv))

    return vals, ids


def pytest_generate_tests(metafunc):
    # if we import it in the header, it messes up with coverage results
    import tigger.cluda as cluda

    if 'ctx_and_double' in metafunc.funcargnames:
        pairs, pair_ids = get_contexts_and_doubles(metafunc)
        metafunc.parametrize('ctx_and_double', pairs, ids=pair_ids, indirect=True)

    if 'ctx' in metafunc.funcargnames:
        ccs, cc_ids = get_contexts(metafunc)
        metafunc.parametrize('ctx', ccs, ids=cc_ids, indirect=True)

    if 'some_ctx' in metafunc.funcargnames:
        # Just some context for tests that only check context-independent stuff.
        ccs, cc_ids = get_contexts(metafunc)
        metafunc.parametrize('some_ctx', [ccs[0]], ids=[cc_ids[0]], indirect=True)

    if 'cluda_api' in metafunc.funcargnames:
        apis, api_ids = get_apis(metafunc)
        metafunc.parametrize('cluda_api', apis, ids=api_ids)

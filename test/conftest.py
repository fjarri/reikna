from pytest_contextgen import create_context_in_tuple, \
    parametrize_context_tuple, pair_context_with_doubles, get_contexts, get_apis

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


pytest_funcarg__ctx_and_double = create_context_in_tuple
pytest_funcarg__ctx = create_context_in_tuple
pytest_funcarg__some_ctx = create_context_in_tuple


def pytest_generate_tests(metafunc):
    if 'ctx_and_double' in metafunc.funcargnames:
        parametrize_context_tuple(metafunc, 'ctx_and_double', pair_context_with_doubles)

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

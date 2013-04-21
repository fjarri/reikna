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
    parser.addoption("--device-include-mask", action="append",
        help="Run tests on matching devices only",
        default=[])
    parser.addoption("--device-exclude-mask", action="append",
        help="Run tests on matching devices only",
        default=[])
    parser.addoption("--platform-include-mask", action="append",
        help="Run tests on matching platforms only",
        default=[])
    parser.addoption("--platform-exclude-mask", action="append",
        help="Run tests on matching platforms only",
        default=[])
    parser.addoption("--include-duplicate-devices", action="store_true",
        help="Run tests on all available devices and not only on uniquely named ones",
        default=False)


pytest_funcarg__thr_and_double = create_context_in_tuple
pytest_funcarg__thr = create_context_in_tuple
pytest_funcarg__some_thr = create_context_in_tuple


def pytest_report_header(config):
    ccs, cc_ids = get_contexts(config)
    devices = {cc.device_id:(cc.platform_name + ", " + cc.device_name) for cc in ccs}
    if len(devices) == 0:
        raise ValueError("No devices match the criteria")

    print("Running tests on:")
    for device_id in sorted(devices):
        print("  " + device_id +  ": " + devices[device_id])


def pytest_generate_tests(metafunc):
    if 'thr_and_double' in metafunc.funcargnames:
        parametrize_context_tuple(metafunc, 'thr_and_double', pair_context_with_doubles)

    if 'thr' in metafunc.funcargnames:
        ccs, cc_ids = get_contexts(metafunc.config)
        metafunc.parametrize('thr', ccs, ids=cc_ids, indirect=True)

    if 'some_thr' in metafunc.funcargnames:
        # Just some context for tests that only check context-independent stuff.
        ccs, cc_ids = get_contexts(metafunc.config)
        metafunc.parametrize('some_thr', [ccs[0]], ids=[cc_ids[0]], indirect=True)

    if 'cluda_api' in metafunc.funcargnames:
        apis, api_ids = get_apis(metafunc.config)
        metafunc.parametrize('cluda_api', apis, ids=api_ids)

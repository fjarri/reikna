import pytest

from pytest_threadgen import create_thread_in_tuple, \
    parametrize_thread_tuple, pair_thread_with_doubles, get_threads, get_apis

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
        help="Use fast math (where applicable): no/yes/both",
        default="no", choices=["no", "yes", "both"])
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


@pytest.fixture
def thr_and_double(request):
    return create_thread_in_tuple(request)

@pytest.fixture
def thr(request):
    return create_thread_in_tuple(request)

@pytest.fixture
def some_thr(request):
    return create_thread_in_tuple(request)


def pytest_report_header(config):
    tps, tp_ids = get_threads(config)
    devices = dict((tp.device_id, tp.device_full_name) for tp in tps)
    if len(devices) == 0:
        raise ValueError("No devices match the criteria")

    print("Running tests on:")
    for device_id in sorted(devices):
        print("  " + device_id +  ": " + devices[device_id])


def pytest_generate_tests(metafunc):

    if 'fast_math' in metafunc.funcargnames:
        fm = metafunc.config.option.fast_math
        fms = dict(both=[False, True], no=[False], yes=[True])[fm]
        fm_ids = [{False:'nofm', True:'fm'}[fm] for fm in fms]
        metafunc.parametrize('fast_math', fms, ids=fm_ids)

    if 'thr_and_double' in metafunc.funcargnames:
        parametrize_thread_tuple(metafunc, 'thr_and_double', pair_thread_with_doubles)

    if 'thr' in metafunc.funcargnames:
        tps, tp_ids = get_threads(metafunc.config)
        metafunc.parametrize('thr', tps, ids=tp_ids, indirect=True)

    if 'some_thr' in metafunc.funcargnames:
        # Just some thread for tests that only check thread-independent stuff.
        tps, tp_ids = get_threads(metafunc.config)
        metafunc.parametrize('some_thr', [tps[0]], ids=[tp_ids[0]], indirect=True)

    if 'cluda_api' in metafunc.funcargnames:
        apis, api_ids = get_apis(metafunc.config)
        metafunc.parametrize('cluda_api', apis, ids=api_ids)

import pytest

from grunnur import Queue


pytest_plugins = ['pytest_returnvalues']


def pytest_addoption(parser):
    parser.addoption("--fast-math", dest="fast_math", action="store",
        help="Use fast math (where applicable): no/yes/both",
        default="no", choices=["no", "yes", "both"])


def pytest_generate_tests(metafunc):
    if 'fast_math' in metafunc.fixturenames:
        fm = metafunc.config.option.fast_math
        fms = dict(both=[False, True], no=[False], yes=[True])[fm]
        fm_ids = [{False:'nofm', True:'fm'}[fm] for fm in fms]
        metafunc.parametrize('fast_math', fms, ids=fm_ids)


@pytest.fixture
def queue(context):
    return Queue(context.device)


@pytest.fixture
def some_queue(some_context):
    return Queue(some_context.device)

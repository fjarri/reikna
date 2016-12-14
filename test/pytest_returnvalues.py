"""
This Py.Test plugin allows return value collection from testcases
(for cases when pass/fail is not enough).
"""

import pytest

# renderers
renderers = {
    'GFLOPS': lambda x: "{f:.2f} GFLOPS".format(f=float(x[1]) / x[0] / 1e9),
    'GB/s': lambda x: "{f:.2f} GB/s".format(f=float(x[1]) / x[0] / 2 ** 30)
}

def pytest_configure(config):
    config.pluginmanager.register(ReturnValuesPlugin(config), "returnvalues")
    config.addinivalue_line("markers", "returns(value_type): collect this testcase's return value")


class ReturnValuesPlugin(object):

    def __init__(self, config):
        pass

    @pytest.hookimpl(hookwrapper=True)
    def pytest_report_teststatus(self, report):
        outcome = yield
        out, letter, msg = outcome.get_result()

        # if we have some result attached to the testcase, print it instead of 'PASSED'
        if hasattr(report, 'retval'):
            msg = report.retval

        outcome.force_result((out, letter, msg))

    @pytest.hookimpl(tryfirst=True)
    def pytest_pyfunc_call(self, pyfuncitem):
        testfunction = pyfuncitem.obj

        # Taken from _pytest/python.py/pytest_pyfunc_call()
        # This bit uses some internal functions which seem to change without notice.
        # Need to replace it with proper mechanism when such functionality is available in pytest.
        # Could be done with raising a special exception and catching it here,
        # but it would be hard to import it from a test somewhere deep in the hierarchy.
        # Add it as a parameter value maybe?
        if pyfuncitem._isyieldedfunction():
            res = testfunction(*pyfuncitem._args)
        else:
            funcargs = pyfuncitem.funcargs
            testargs = {}
            for arg in pyfuncitem._fixtureinfo.argnames:
                testargs[arg] = funcargs[arg]
            res = testfunction(**testargs)

        pyfuncitem.retval = res
        return True # finished processing the callback

    @pytest.hookimpl(hookwrapper=True)
    def pytest_runtest_makereport(self, item, call):
        outcome = yield
        report = outcome.get_result()

        # if the testcase has passed, and has 'perf' marker, process its results
        if call.when == 'call' and report.passed and hasattr(item.function, 'returns'):
            mark = item.function.returns
            if len(mark.args) > 0:
                if mark.args[0] in renderers:
                    renderer = renderers[mark.args[0]]
                else:
                    renderer = lambda x: repr(x) + " " + mark.args[0]
            else:
                renderer = lambda x: repr(x)

            report.retval = renderer(item.retval)

        outcome.force_result(report)

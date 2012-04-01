"""
This Py.Test plugin allows return value collection from testcases
(for cases when pass/fail is not enough).
"""

import pytest

# renderers
renderers = {
    'GFLOPS': lambda x: "{f:.2f} GFLOPS".format(f=float(x[1]) / x[0] / 1e9)
}

def pytest_configure(config):
    config.pluginmanager.register(ReturnValuesPlugin(config), "returnvalues")
    config.addinivalue_line("markers", "returns(value_type): collect this testcase's return value")


class ReturnValuesPlugin(object):

    def __init__(self, config):
        pass

    def pytest_report_teststatus(self, __multicall__, report):
        outcome, letter, msg = __multicall__.execute()

        # if we have some result attached to the testcase, print it instead of 'PASSED'
        if hasattr(report, 'retval'):
            msg = report.retval

        return outcome, letter, msg

    def pytest_pyfunc_call(self, __multicall__, pyfuncitem):
        # collect testcase return result
        testfunction = pyfuncitem.obj
        if pyfuncitem._isyieldedfunction():
            res = testfunction(*pyfuncitem._args)
        else:
            funcargs = pyfuncitem.funcargs
            res = testfunction(**funcargs)
        pyfuncitem.retval = res

    def pytest_runtest_makereport(self, __multicall__, item, call):
        report = __multicall__.execute()

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

        return report

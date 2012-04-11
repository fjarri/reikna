from itertools import product
import tigger.cluda as cluda
import numpy

pytest_plugins = ['pytest_returnvalues']


def pytest_addoption(parser):
    parser.addoption("--platform", action="store",
        help="Platform: cuda/ocl/supported",
        default="supported", choices=["cuda", "ocl", "supported"])
    parser.addoption("--double", action="store",
        help="Use doubles: no/yes/supported",
        default="supported", choices=["no", "yes", "supported"])
    parser.addoption("--fast-math", dest="fast_math", action="store",
        help="Use fast math: no/yes/both",
        default="yes", choices=["no", "yes", "both"])

def pytest_funcarg__env(request):
    return request.param()

def pytest_funcarg__double(request):
    return request.param

def pytest_generate_tests(metafunc):

    if 'env' in metafunc.funcargnames:
        p = metafunc.config.option.platform
        fm = metafunc.config.option.fast_math

        def check_platform(name):
            return dict(cuda=cluda.supportsCuda, ocl=cluda.supportsOcl)[name]()

        class EnvCreator:
            def __init__(self, name, fast_math):
                self.fast_math = fast_math
                self.name = name
                ctr = dict(cuda=cluda.createCuda, ocl=cluda.createOcl)[name]
                self.create = lambda: ctr(fast_math=fm)

            def __call__(self):
                return self.create()

            def __str__(self):
                return self.name + (",fm" if self.fast_math else "")

        if p == "supported":
            ps = []
            for name in ('cuda', 'ocl'):
                if check_platform(name): ps.append(name)
        else:
            if not check_platform(p):
                raise Exception("Requested platform " + p + " is not supported.")
            ps = [p]

        if fm == "both":
            fms = [False, True]
        elif fm == "no":
            fms = [False]
        else:
            fms = [True]

        envs = [EnvCreator(p, fm) for p, fm in product(ps, fms)]

    if 'double' in metafunc.funcargnames:
        d = metafunc.config.option.double
        ds = lambda dv: 'dp' if dv else 'sp'

        if d == "supported":
            vals = [(e, dv) for e, dv in product(envs, [False, True]) if not dv or e().supportsDtype(numpy.float64)]
        else:
            dv = d == 'yes'
            vals = [(e, dv) for e in envs]

        ids = [str(e) + "," + ds(dv) for e, dv in vals]
        metafunc.parametrize(["env", "double"], vals, ids=ids, indirect=True)
    else:
        metafunc.parametrize("env", envs, [str(e) for e in envs], indirect=True)

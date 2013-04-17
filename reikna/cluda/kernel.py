import os.path
from logging import error
from warnings import warn

import numpy
from mako.template import Template
from mako import exceptions

import reikna.helpers as helpers
from reikna.helpers import AttrDict, template_for, template_from
from reikna.cluda import dtypes

TEMPLATE = template_for(__file__)


class FuncCollector:
    """
    .. py:method:: mul(*dtypes, out=None)

        Returns the name of the function that multiplies values of types ``dtypes``.
        If ``out`` is given, it will be set as a return type for this function.
        If the truncation of the imaginary part of the result has to be performed,
        :py:class:`numpy.ComplexWarning` is thrown.

    .. py:method:: div(dtype1, dtype2, out=None)

        Returns the name of the function that divides values of ``dtype1`` and ``dtype2``.
        If ``out`` is given, it will be set as a return type for this function.

    .. py:method:: cast(out_dtype, in_dtype)

        Returns the name of the function that casts values of ``in_dtype`` to ``out_dtype``.

    .. py:method:: conj(dtype)

        Returns the name of the function that conjugates the value of type ``dtype``
        (must be a complex data type).

    .. py:method:: norm(dtype)

        Returns the name of the function that returns the norm of the value of type ``dtype``
        (product by the complex conjugate if the value is complex, square otherwise).

    .. py:method:: exp(dtype)

        Returns the name of the function that exponentiates the value of type ``dtype``
        (must be a real or complex data type).

    .. py:method:: polar(dtype)

        Returns the name of the function that calculates ``rho * exp(i * theta)``
        for values ``rho, theta`` of type ``dtype`` (must be a real data type).
    """

    def __init__(self, prefix=""):
        self.prefix = prefix

        self._register_function('cast', 1, out_param='positional')
        self._register_function('mul', None, out_param='keyword')
        self._register_function('div', 2, out_param='keyword')
        self._register_function('conj', 1)
        self._register_function('norm', 1)
        self._register_function('exp', 1)
        self._register_function('polar', 1)

        self.functions = {}

    def _register_function(self, name, arguments, out_param=None):

        def func(*dts, **kwds):

            if arguments is not None:
                expected_args = arguments + (1 if out_param == 'positional' else 0)
                if len(dts) != expected_args:
                    raise TypeError(name + "() takes exactly " + str(expected_args) +
                        " arguments (" + str(len(dts)) + " given)")

            if out_param == 'keyword':
                result_dt = dtypes.result_type(*dts)
                out_dt = kwds.get('out', result_dt)

                if dtypes.is_complex(result_dt) and not dtypes.is_complex(out_dt):
                    warn("Imaginary part ignored during the downcast from " +
                        " * ".join([str(d) for d in dts]) +
                        " to " + str(out_dt),
                        numpy.ComplexWarning)

                in_dts = dts
            elif out_param == 'positional':
                out_dt = dts[0]
                in_dts = dts[1:]
            else:
                out_dt = None
                in_dts = dts

            typeid = lambda dtype: dtypes.ctype(dtype).replace(' ', '_')

            # Separating output and input names to avoid clashes
            full_name = "_" + self.prefix + "_" + name + \
                "__" + (typeid(out_dt) if out_dt is not None else "") + \
                "__" + "_".join([typeid(dtype) for dtype in in_dts])

            self.functions[full_name] = (name,
                ((out_dt,) if out_dt is not None else tuple()) + in_dts)

            return full_name

        setattr(self, name, func)

    def render(self):
        src = []
        for func_name, params in self.functions.items():
            tmpl_name, args = params
            src.append(TEMPLATE.get_def(tmpl_name).render(
                func_name, *args, dtypes=dtypes, numpy=numpy))
        return "\n".join(src)


def render_prelude(ctx):
    return TEMPLATE.get_def('prelude').render(api=ctx.api.API_ID, ctx_fast_math=ctx._fast_math)

def render_without_funcs(template, func_c, *args, **kwds):
    # add some "built-ins" to kernel
    render_kwds = dict(dtypes=dtypes, numpy=numpy, func=func_c, helpers=helpers)
    assert set(render_kwds).isdisjoint(set(kwds))
    render_kwds.update(kwds)

    try:
        src = template.render(*args, **render_kwds)
    except:
        error("Failed to render template:\n" + exceptions.text_error_template().render())
        raise Exception("Template rendering failed")
    return src


class Module:

    def __init__(self, template, render_kwds=None, snippet=False):
        self.template = template_from(template)
        self.render_kwds = dict(render_kwds)
        self.snippet = snippet


def get_prefix(n):
    return "_module" + str(n) + "_"


class ProcessedModule(AttrDict): pass


def flatten_module(module_list, module):

    processed_module = ProcessedModule(
        template=module.template,
        render_kwds=dict(module.render_kwds))

    for kwd, val in processed_module.render_kwds.items():
        if isinstance(val, Module):
            processed_module.render_kwds[kwd] = flatten_module(module_list, val)

    if not module.snippet:
        prefix = get_prefix(len(module_list))
        module_list.append(ProcessedModule(
            template=template_from("""\n${module(prefix)}\n"""),
            render_kwds=dict(module=processed_module, prefix=prefix)))
        return prefix
    else:
        return processed_module


def flatten_module_tree(src, render_kwds):
    main_module = Module(src, render_kwds=render_kwds, snippet=True)
    module_list = []
    main_module = flatten_module(module_list, main_module)
    module_list.append(main_module)
    return module_list


def render_snippet_tree(pm, func_c):
    kwds = pm.render_kwds
    for kwd, val in kwds.items():
        if isinstance(val, ProcessedModule):
            kwds[kwd] = render_snippet_tree(val, func_c)

    return lambda *args: render_without_funcs(
        pm.template, func_c, *args, **pm.render_kwds)


def render_template_source_with_modules(src, *args, **render_kwds):

    module_list = flatten_module_tree(src, render_kwds)
    func_c = FuncCollector()
    renderers = [render_snippet_tree(pm, func_c) for pm in module_list]
    src_list = [render() for render in renderers[:-1]]
    src_list.append(renderers[-1](*args))

    return func_c.render() + "\n".join(src_list)


def render_template(template, *args, **kwds):
    func_c = FuncCollector()
    src = render_without_funcs(template_from(template), func_c, *args, **kwds)
    return func_c.render() + src

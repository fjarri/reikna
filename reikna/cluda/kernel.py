from logging import error

import numpy
from mako import exceptions

import reikna.helpers as helpers
from reikna.helpers import AttrDict, template_for, template_from
from reikna.cluda import dtypes


TEMPLATE = template_for(__file__)


def render_prelude(ctx):
    return TEMPLATE.get_def('prelude').render(api=ctx.api.API_ID, ctx_fast_math=ctx._fast_math)

def render_without_funcs(template, *args, **kwds):
    # add some "built-ins" to kernel
    render_kwds = dict(dtypes=dtypes, numpy=numpy, helpers=helpers)
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
        self.render_kwds = {} if render_kwds is None else dict(render_kwds)
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


def render_snippet_tree(pm):
    kwds = pm.render_kwds
    for kwd, val in kwds.items():
        if isinstance(val, ProcessedModule):
            kwds[kwd] = render_snippet_tree(val)

    return lambda *args: render_without_funcs(
        pm.template, *args, **pm.render_kwds)


def render_template_source_with_modules(src, *args, **render_kwds):

    module_list = flatten_module_tree(src, render_kwds)
    renderers = [render_snippet_tree(pm) for pm in module_list]
    src_list = [render() for render in renderers[:-1]]
    src_list.append(renderers[-1](*args))

    return "\n\n".join(src_list)


def render_template(template, *args, **kwds):
    func_c = FuncCollector()
    src = render_without_funcs(template_from(template), func_c, *args, **kwds)
    return func_c.render() + src

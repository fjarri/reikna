from logging import error

import numpy
from mako import exceptions

import reikna.helpers as helpers
from reikna.helpers import AttrDict, template_for, template_from, \
    template_def, extract_argspec_and_value
from reikna.cluda import dtypes


TEMPLATE = template_for(__file__)


def render_prelude(thr):
    return TEMPLATE.get_def('prelude').render(
        api=thr.api.get_id(), thread_fast_math=thr._fast_math)


def render_template(template, *args, **kwds):
    # add some "built-ins" to the kernel
    render_kwds = dict(dtypes=dtypes, numpy=numpy, helpers=helpers)
    assert set(render_kwds).isdisjoint(set(kwds))
    render_kwds.update(kwds)

    try:
        src = template.render(*args, **render_kwds)
    except:
        error(
            "Failed to render template with"
            "\nargs: {args}\nkwds: {kwds}\nsource:\n{source}\n"
            "{exception}".format(
                args=args, kwds=kwds, source=template.source,
                exception=exceptions.text_error_template().render()))
        raise Exception("Template rendering failed")
    return src


class BaseModule:
    def __init__(self, template, render_kwds=None, snippet=False):
        self.template = template_from(template)
        self.render_kwds = {} if render_kwds is None else dict(render_kwds)
        self.snippet = snippet


class Snippet(BaseModule):
    """
    Contains a CLUDA snippet.
    See :ref:`tutorial-modules` for details.

    :param template_src: a ``Mako`` template with the module code,
        or a string with the template source.
    :type template_src: ``str`` or ``Mako`` template.
    :param render_kwds: a dictionary which will be used to render the template.
        Can contain other modules and snippets.
    """

    def __init__(self, template_src, render_kwds=None):
        BaseModule.__init__(self, template_src, render_kwds=render_kwds, snippet=True)

    @classmethod
    def create(cls, argspec_func, render_kwds=None):
        """
        Creates a snippet from the ``Mako`` def with the same signature as ``argspec_func``
        and the body equal to the string it returns.
        """
        argspec, code = extract_argspec_and_value(argspec_func)
        return cls(template_def(argspec, code), render_kwds=render_kwds)


class Module(BaseModule):
    """
    Contains a CLUDA module.
    See :ref:`tutorial-modules` for details.

    :param template_src: a ``Mako`` template with the module code,
        or a string with the template source.
    :type template_src: ``str`` or ``Mako`` template.
    :param render_kwds: a dictionary which will be used to render the template.
        Can contain other modules and snippets.
    """

    def __init__(self, template_src, render_kwds=None):
        BaseModule.__init__(self, template_src, render_kwds=render_kwds, snippet=False)

    @classmethod
    def create(cls, code, render_kwds=None):
        """
        Creates a module from the ``Mako`` def with a single positional argument ``prefix``
        and the body ``code``.
        """
        return cls(template_def(['prefix'], code), render_kwds=render_kwds)


class ProcessedModule(AttrDict): pass


def traverse_data(target_cls, target_func, accum, val):
    traverse = lambda v: traverse_data(target_cls, target_func, accum, v)

    if isinstance(val, target_cls):
        return target_func(accum, traverse, val)
    elif isinstance(val, AttrDict):
        return AttrDict({k:traverse(v) for k, v in val.items()})
    elif isinstance(val, dict):
        return {k:traverse(v) for k, v in val.items()}
    elif isinstance(val, tuple):
        return tuple(traverse(v) for v in val)
    elif isinstance(val, list):
        return [traverse(v) for v in val]
    else:
        return val


def flatten_module(module_list, traverse, module):

    processed_module = ProcessedModule(
        template=module.template,
        render_kwds=traverse(module.render_kwds))

    if not module.snippet:
        prefix = "_module" + str(len(module_list)) + "_"
        module_list.append(ProcessedModule(
            template=template_from("""\n${module(prefix)}\n"""),
            render_kwds=dict(module=processed_module, prefix=prefix)))
        return prefix
    else:
        return processed_module


def flatten_module_tree(src, args, render_kwds):
    main_module = Snippet(src, render_kwds=render_kwds)
    module_list = []
    traverse = lambda v: traverse_data(BaseModule, flatten_module, module_list, v)
    args = traverse(args)
    main_module = traverse(main_module)
    module_list.append(main_module)
    return args, module_list


def create_renderer(_, traverse, pm):
    pm.render_kwds = traverse(pm.render_kwds)
    return lambda *args: render_template(pm.template, *args, **pm.render_kwds)


def create_renderer_tree(pm):
    return traverse_data(ProcessedModule, create_renderer, None, pm)


def render_template_source(src, render_args=None, render_kwds=None):

    if render_args is None:
        render_args = []
    if render_kwds is None:
        render_kwds = {}

    args, module_list = flatten_module_tree(src, render_args, render_kwds)
    renderers = [create_renderer_tree(pm) for pm in module_list]
    src_list = [render() for render in renderers[:-1]]
    src_list.append(renderers[-1](*args))

    return "\n\n".join(src_list)

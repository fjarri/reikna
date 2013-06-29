from logging import error

import numpy
from mako import exceptions

import reikna.helpers as helpers
from reikna.helpers import AttrDict, template_for, template_from, \
    template_def, template_argspec, extract_argspec_and_value
from reikna.cluda import dtypes


TEMPLATE = template_for(__file__)


def render_prelude(thr):
    return TEMPLATE.get_def('prelude').render(
        api=thr.api.get_id(), thread_fast_math=thr._fast_math)


def render_template(template, *args, **kwds):
    # add some "built-ins" to the kernel
    render_kwds = dict(dtypes=dtypes, helpers=helpers)
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


class Snippet:
    """
    Contains a CLUDA snippet.
    See :ref:`tutorial-modules` for details.

    :param template_src: a ``Mako`` template with the module code,
        or a string with the template source.
    :type template_src: ``str`` or ``Mako`` template.
    :param render_kwds: a dictionary which will be used to render the template.
        Can contain other modules and snippets.

    .. py:attribute:: argspec

        An ``ArgSpec`` named tuple with the template's signature
        (see ``inspect`` module for details).
    """

    def __init__(self, template_src, render_kwds=None):
        self.template = template_from(template_src)
        self.argspec = template_argspec(self.template)
        self.render_kwds = {} if render_kwds is None else dict(render_kwds)

    @classmethod
    def create(cls, argspec_func, render_kwds=None):
        """
        Creates a snippet from the ``Mako`` def with the same signature as ``argspec_func``
        and the body equal to the string it returns.
        """
        argspec, code = extract_argspec_and_value(argspec_func)
        return cls(template_def(argspec, code), render_kwds=render_kwds)


class Module:
    """
    Contains a CLUDA module.
    See :ref:`tutorial-modules` for details.

    :param template_src: a ``Mako`` template with the module code,
        or a string with the template source.
    :type template_src: ``str`` or ``Mako`` template.
    :param render_kwds: a dictionary which will be used to render the template.
        Can contain other modules and snippets.

    .. py:attribute:: argspec

        An ``ArgSpec`` named tuple with the template's signature
        (see ``inspect`` module for details).
    """

    def __init__(self, template_src, render_kwds=None):
        self.template = template_from(template_src)
        self.argspec = template_argspec(self.template)
        self.render_kwds = {} if render_kwds is None else dict(render_kwds)

    @classmethod
    def create(cls, code, render_kwds=None):
        """
        Creates a module from the ``Mako`` def with a single positional argument ``prefix``
        and the body ``code``.
        """
        return cls(template_def(['prefix'], code), render_kwds=render_kwds)


class Renderable:
    def __init__(self, template_def, render_kwds):
        self.template_def = template_def
        self.render_kwds = render_kwds

    def __call__(self, *args):
        return render_template(self.template_def, *args, **self.render_kwds)


def process(obj, renderables):

    if isinstance(obj, Snippet):
        render_kwds = process(obj.render_kwds, renderables)
        return Renderable(obj.template, render_kwds)

    elif isinstance(obj, Module):
        render_kwds = process(obj.render_kwds, renderables)
        prefix = "_module" + str(len(renderables)) + "_"

        module_renderable = Renderable(obj.template, render_kwds)
        wrapper_renderable = Renderable(
            template_from("""\n${module(prefix)}\n"""),
            render_kwds=dict(module=module_renderable, prefix=prefix))

        renderables.append(wrapper_renderable)
        return prefix

    elif hasattr(obj, '__process_modules__'):
        return obj.__process_modules__(lambda x: process(x, renderables))
    elif isinstance(obj, dict):
        return {k:process(v, renderables) for k, v in obj.items()}
    elif isinstance(obj, tuple):
        return tuple(process(v, renderables) for v in obj)
    elif isinstance(obj, list):
        return [process(v, renderables) for v in obj]
    else:
        return obj


def render_template_source(src, render_args=None, render_kwds=None):

    if render_args is None:
        render_args = []
    if render_kwds is None:
        render_kwds = {}

    renderables = []
    render_args = process(render_args, renderables)
    main_renderable = process(Snippet(src, render_kwds=render_kwds), renderables)

    src_list = [renderable() for renderable in renderables]
    src_list.append(main_renderable(*render_args))

    return "\n\n".join(src_list)

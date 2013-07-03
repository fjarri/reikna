from logging import error

import numpy
from mako import exceptions

import reikna.helpers as helpers
from reikna.helpers import AttrDict, template_for, template_from, \
    template_def, extract_signature_and_value
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
    """

    def __init__(self, template_src, render_kwds=None):
        self.template = template_from(template_src)
        self.render_kwds = {} if render_kwds is None else dict(render_kwds)

    @classmethod
    def create(cls, signature_func, render_kwds=None):
        """
        Creates a snippet from the ``Mako`` def with the same signature as ``signature_func``
        and the body equal to the string it returns.
        """
        signature, code = extract_signature_and_value(signature_func)
        return cls(template_def(signature, code), render_kwds=render_kwds)


class Module:
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
        self.template = template_from(template_src)
        self.render_kwds = {} if render_kwds is None else dict(render_kwds)

    @classmethod
    def create(cls, code, render_kwds=None):
        """
        Creates a module from the ``Mako`` def with a single positional argument ``prefix``
        and the body ``code``.
        """
        return cls(template_def(['prefix'], code), render_kwds=render_kwds)


class SourceCollector:

    def __init__(self):
        self.sources = []
        self.prefix_counter = 0

    def add_module(self, template_def, args, render_kwds):
        prefix = "_module" + str(self.prefix_counter) + "_"
        self.prefix_counter += 1

        src = render_template(template_def, prefix, *args, **render_kwds)
        self.sources.append(src)

        return prefix

    def get_source(self):
        return "\n\n".join(self.sources)


class RenderableSnippet:

    def __init__(self, template_def, render_kwds):
        self.template_def = template_def
        self.render_kwds = render_kwds

    def __call__(self, *args):
        return render_template(self.template_def, *args, **self.render_kwds)


class RenderableModule:

    def __init__(self, collector, template_def, render_kwds):
        self.collector = collector
        self.template_def = template_def
        self.render_kwds = render_kwds
        self.no_arg_prefix = None

    def __call__(self, *args):
        prefix = self.collector.add_module(self.template_def, args, self.render_kwds)
        return prefix

    def __str__(self):
        # To avoid a lot of repeating module renders when it's called without arguments
        # (which will be the majority of calls),
        # we are caching the corresponding prefix.
        if self.no_arg_prefix is None:
            self.no_arg_prefix = self()
        return self.no_arg_prefix


def process(obj, collector):
    if isinstance(obj, Snippet):
        render_kwds = process(obj.render_kwds, collector)
        return RenderableSnippet(obj.template, render_kwds)
    elif isinstance(obj, Module):
        render_kwds = process(obj.render_kwds, collector)
        return RenderableModule(collector, obj.template, render_kwds)
    elif hasattr(obj, '__process_modules__'):
        return obj.__process_modules__(lambda x: process(x, collector))
    elif isinstance(obj, dict):
        return {k:process(v, collector) for k, v in obj.items()}
    elif isinstance(obj, tuple):
        return tuple(process(v, collector) for v in obj)
    elif isinstance(obj, list):
        return [process(v, collector) for v in obj]
    else:
        return obj


def render_template_source(src, render_args=None, render_kwds=None):

    if render_args is None:
        render_args = []
    if render_kwds is None:
        render_kwds = {}

    collector = SourceCollector()
    render_args = process(render_args, collector)
    main_renderable = process(Snippet(src, render_kwds=render_kwds), collector)

    main_src = main_renderable(*render_args)

    return collector.get_source() + "\n\n" + main_src

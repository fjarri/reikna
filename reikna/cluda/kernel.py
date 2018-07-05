from logging import error

from mako import exceptions

import reikna.helpers as helpers
from reikna.helpers import template_for, template_from, template_def, extract_signature_and_value
from reikna.cluda import dtypes


TEMPLATE = template_for(__file__)


def render_prelude(thr, fast_math=False, constant_arrays=None):
    return TEMPLATE.get_def('prelude').render(
        api=thr.api.get_id(), compile_fast_math=fast_math,
        dtypes=dtypes, constant_arrays=constant_arrays)


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
    def create(cls, func_or_str, render_kwds=None):
        """
        Creates a snippet from the ``Mako`` def:

        * if ``func_or_str`` is a function, then the def has the same signature as ``func_or_str``,
          and the body equal to the string it returns;
        * if ``func_or_str`` is a string, then the def has empty signature.
        """
        signature, code = extract_signature_and_value(func_or_str)
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
    def create(cls, func_or_str, render_kwds=None):
        """
        Creates a module from the ``Mako`` def:

        * if ``func_or_str`` is a function, then the def has the same signature as ``func_or_str``
          (prefix will be passed as the first positional parameter),
          and the body equal to the string it returns;
        * if ``func_or_str`` is a string, then the def has a single positional argument ``prefix``.
          and the body ``code``.
        """
        signature, code = extract_signature_and_value(func_or_str, default_parameters=['prefix'])
        return cls(template_def(signature, code), render_kwds=render_kwds)


class SourceCollector:

    def __init__(self):
        self.constant_modules = {}
        self.sources = []
        self.prefix_counter = 0

    def add_module(self, module_id, tmpl_def, args, render_kwds):

        # This caching serves two purposes.
        # First, it reduces the amount of generated code by not generating
        # the same module several times.
        # Second, if the same module object is used (without arguments) in other modules,
        # the data structures defined in this module will be suitable
        # for functions in these modules.
        if len(args) == 0:
            if module_id in self.constant_modules:
                return self.constant_modules[module_id]

        prefix = "_module" + str(self.prefix_counter) + "_"
        self.prefix_counter += 1

        src = render_template(tmpl_def, prefix, *args, **render_kwds)
        self.sources.append(src)

        if len(args) == 0:
            self.constant_modules[module_id] = prefix

        return prefix

    def get_source(self):
        return "\n\n".join(self.sources)


class RenderableSnippet:

    def __init__(self, tmpl_def, render_kwds):
        self.template_def = tmpl_def
        self.render_kwds = render_kwds

    def __call__(self, *args):
        return render_template(self.template_def, *args, **self.render_kwds)

    def __str__(self):
        return self()


class RenderableModule:

    def __init__(self, collector, module_id, tmpl_def, render_kwds):
        self.module_id = module_id
        self.collector = collector
        self.template_def = tmpl_def
        self.render_kwds = render_kwds

    def __call__(self, *args):
        return self.collector.add_module(
            self.module_id, self.template_def, args, self.render_kwds)

    def __str__(self):
        return self()


def process(obj, collector):
    if isinstance(obj, Snippet):
        render_kwds = process(obj.render_kwds, collector)
        return RenderableSnippet(obj.template, render_kwds)
    elif isinstance(obj, Module):
        render_kwds = process(obj.render_kwds, collector)
        return RenderableModule(collector, id(obj), obj.template, render_kwds)
    elif hasattr(obj, '__process_modules__'):
        return obj.__process_modules__(lambda x: process(x, collector))
    elif isinstance(obj, dict):
        return dict(((k,process(v, collector)) for k, v in obj.items()))
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

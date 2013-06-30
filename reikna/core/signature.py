import weakref

import funcsigs # backport of inspect.signature() and related objects for Py2
import numpy

import reikna.cluda.dtypes as dtypes
from reikna.helpers import wrap_in_tuple, product


class ArgType:

    def __init__(self, dtype, shape=None, strides=None):
        self.shape = tuple() if shape is None else wrap_in_tuple(shape)
        self.size = product(self.shape)
        self.dtype = dtypes.normalize_type(dtype)
        self.ctype = dtypes.ctype(self.dtype)
        if strides is None:
            self.strides = tuple([
                self.dtype.itemsize * product(self.shape[i+1:]) for i in range(len(self.shape))])
        else:
            self.strides = strides
        self._cast = dtypes.cast(self.dtype)

    def __eq__(self, other):
        return (self.shape == other.shape and self.dtype == other.dtype
            and self.strides == other.strides)

    @classmethod
    def from_value(cls, val):
        if isinstance(val, ArgType):
            return val
        elif numpy.issctype(val):
            return cls(val)
        elif hasattr(val, 'dtype') and hasattr(val, 'shape'):
            strides = val.strides if hasattr(val, 'strides') else None
            return cls(val.dtype, shape=val.shape, strides=strides)
        else:
            return cls(dtypes.detect_type(val))

    def __call__(self, val):
        return self._cast(val)


class Annotation:

    def __init__(self, type_, role=None):
        self.type = ArgType.from_value(type_)

        if role is None:
            role = 's' if len(self.type.shape) == 0 else 'io'

        assert role in ('i', 'o', 'io', 's')
        self.role = role
        if role == 's':
            self.array = False
            self.input = False
            self.output = False
        else:
            self.array = True
            self.input = 'i' in role
            self.output = 'o' in role

    def __eq__(self, other):
        return self.type == other.type and self.role == other.role


class Parameter(funcsigs.Parameter):
    # - names are mandatory
    # - array parameters do not have defaults
    # - technically, all objects have kind funcsigs.Parameter.POSITIONAL_OR_KEYWORD

    def __init__(self, name, annotation, default=funcsigs.Parameter.empty):

        if default is not funcsigs.Parameter.empty:
            assert not annotation.array
            default = annotation.type(default)

        # HACK: Parameter constructor is not documented.
        # But I need to create these objects somehow.
        funcsigs.Parameter.__init__(
            self, name, annotation=annotation,
            kind=funcsigs.Parameter.POSITIONAL_OR_KEYWORD,
            default=default)

    def rename(self, new_name):
        return Parameter(new_name, self.annotation, default=self.default)

    def __eq__(self, other):
        return (self.name == other.name and self.annotation == other.annotation
            and self.default == other.default)


class Signature(funcsigs.Signature):

    def __init__(self, parameters):
        # HACK: Signature constructor is not documented.
        # But I need to create these objects somehow.
        funcsigs.Signature.__init__(self, parameters)

    def bind_with_defaults(self, args, kwds, cast=False):
        ba = self.bind(*args, **kwds)
        for param in self.parameters.values():
            if param.name not in ba.arguments:
                ba.arguments[param.name] = param.default
            elif cast:
                if not param.annotation.array:
                    ba.arguments[param.name] = param.annotation.type(ba.arguments[param.name])
        return ba


class ComputationParameter(ArgType):

    def __init__(self, computation, name, type_):
        ArgType.__init__(self, type_.dtype, shape=type_.shape, strides=type_.strides)
        self._computation = weakref.ref(computation)
        self.name = name

    def belongs_to(self, comp):
        return self._computation() is comp

    def connect(self, tr, tr_connector, **connections):
        return self._computation().connect(self.name, tr, tr_connector, **connections)


class TransformationParameter(ArgType):

    def __init__(self, tr, name, type_):
        ArgType.__init__(self, type_.dtype, shape=type_.shape, strides=type_.strides)
        self._tr = weakref.ref(tr)
        self.name = name

    def belongs_to(self, tr):
        return self._tr() is tr


def extract_parameter_name(parent, obj):
    if isinstance(obj, ComputationParameter):
        if not obj.belongs_to(parent):
            raise ValueError(
                "Parameter " + obj.name + " belongs to a different computation object")
        return obj.name
    elif isinstance(obj, TransformationParameter):
        if not obj.belongs_to(parent):
            raise ValueError(
                "Parameter " + obj.name + " belongs to a different transformation object")
        return obj.name
    elif isinstance(obj, str):
        return obj
    raise ValueError("Unknown type of computation parameter: " + str(type(obj)))

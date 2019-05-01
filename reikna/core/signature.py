import funcsigs # backport of inspect.signature() and related objects for Py2
import numpy

import reikna.helpers as helpers
import reikna.cluda.dtypes as dtypes
from reikna.helpers import wrap_in_tuple, product


class Type:
    """
    Represents an array or, as a degenerate case, scalar type of a computation parameter.

    .. py:attribute:: shape

        A tuple of integers.
        Scalars are represented by an empty tuple.

    .. py:attribute:: dtype

        A ``numpy.dtype`` instance.

    .. py:attribute:: ctype

        A string with the name of C type corresponding to :py:attr:`dtype`,
        or a module if it is a struct type.

    .. py:attribute:: strides

        Tuple of bytes to step in each dimension when traversing an array.

    .. py:attribute:: offset

        The initial offset (in bytes).

    .. py:attribute:: nbytes

        The total size of the memory buffer (in bytes)
    """

    def __init__(self, dtype, shape=None, strides=None, offset=0, nbytes=None):
        self.shape = tuple() if shape is None else wrap_in_tuple(shape)
        self.size = product(self.shape)
        self.dtype = dtypes.normalize_type(dtype)
        self.ctype = dtypes.ctype_module(self.dtype)

        default_strides = helpers.default_strides(self.shape, self.dtype.itemsize)
        if strides is None:
            strides = default_strides
        else:
            strides = tuple(strides)
        self._default_strides = strides == default_strides
        self.strides = strides

        default_nbytes = helpers.min_buffer_size(self.shape, self.dtype.itemsize, self.strides)
        if nbytes is None:
            nbytes = default_nbytes
        self._default_nbytes = nbytes == default_nbytes
        self.nbytes = nbytes

        self.offset = offset
        self._cast = dtypes.cast(self.dtype)

    def __eq__(self, other):
        return (
            self.__class__ == other.__class__
            and self.shape == other.shape
            and self.dtype == other.dtype
            and self.strides == other.strides
            and self.offset == other.offset
            and self.nbytes == self.nbytes)

    def __hash__(self):
        return hash((
            self.__class__,
            self.shape,
            self.dtype,
            self.strides,
            self.offset,
            self.nbytes,
            ))

    def __ne__(self, other):
        return not (self == other)

    def compatible_with(self, other):
        if self.dtype != other.dtype:
            return False

        common_shape_len = min(len(self.shape), len(other.shape))
        if self.shape[-common_shape_len:] != other.shape[-common_shape_len:]:
            return False
        if self.strides[-common_shape_len:] != other.strides[-common_shape_len:]:
            return False
        if helpers.product(self.shape[:-common_shape_len]) != 1:
            return False
        if helpers.product(other.shape[:-common_shape_len]) != 1:
            return False
        if self.offset != other.offset:
            return False

        return True

    def broadcastable_to(self, other):
        """
        Returns ``True`` if the shape of this ``Type`` is broadcastable to ``other``,
        that is its dimensions either coincide with the innermost dimensions of ``other.shape``,
        or are equal to 1.
        """
        if len(self.shape) > len(other.shape):
            return False

        for i in range(1, len(self.shape)+1):
            if not (self.shape[-i] == 1 or self.shape[-i] == other.shape[-i]):
                return False

        return True

    def with_dtype(self, dtype):
        """
        Creates a :py:class:`Type` object with its ``dtype`` attribute replaced by the given dtype.
        """
        return Type(
            dtype, shape=self.shape, strides=self.strides, offset=self.offset, nbytes=self.nbytes)

    @classmethod
    def from_value(cls, val):
        """
        Creates a :py:class:`Type` object corresponding to the given value.
        """
        if isinstance(val, Type):
            # Creating a new object, because ``val`` may be some derivative of Type,
            # used as a syntactic sugar, and we do not want it to confuse us later.
            return cls(
                val.dtype, shape=val.shape, strides=val.strides,
                offset=val.offset, nbytes=val.nbytes)
        elif numpy.issctype(val):
            return cls(val)
        elif hasattr(val, 'dtype') and hasattr(val, 'shape'):
            strides = val.strides if hasattr(val, 'strides') else None
            offset = val.offset if hasattr(val, 'offset') else 0
            nbytes = val.nbytes if hasattr(val, 'nbytes') else None
            return cls(
                val.dtype, shape=val.shape, strides=strides, offset=offset, nbytes=nbytes)
        else:
            return cls(dtypes.detect_type(val))

    @classmethod
    def padded(cls, dtype, shape, pad=0):
        """
        Creates a :py:class:`Type` object corresponding to an array padded from all dimensions
        by `pad` elements.
        """
        dtype = dtypes.normalize_type(dtype)
        strides, offset, nbytes = helpers.padded_buffer_parameters(shape, dtype.itemsize, pad=pad)
        return cls(dtype, shape, strides=strides, offset=offset, nbytes=nbytes)

    def __call__(self, val):
        """
        Casts the given value to this type.
        """
        return self._cast(val)

    def __repr__(self):
        if len(self.shape) > 0 or self.offset != 0:
            res = "Type({dtype}, shape={shape}".format(dtype=self.dtype, shape=self.shape)
            if not self._default_strides:
                res += ", strides=" + str(self.strides)
            if self.offset != 0:
                res += ", offset=" + str(self.offset)
            if not self._default_nbytes:
                res += ", nbytes=" + str(self.nbytes)
            res += ")"
            return res
        else:
            return "Type({dtype})".format(dtype=self.dtype)

    def __process_modules__(self, process):
        tp = Type(
            self.dtype, shape=self.shape, strides=self.strides,
            offset=self.offset, nbytes=self.nbytes)
        tp.ctype = process(tp.ctype)
        return tp


class Annotation:
    """
    Computation parameter annotation,
    in the same sense as it is used for functions in the standard library.

    :param type_: a :py:class:`~reikna.core.Type` object.
    :param role: any of ``'i'`` (input), ``'o'`` (output),
        ``'io'`` (input/output), ``'s'`` (scalar).
        Defaults to ``'s'`` for scalars, ``'io'`` for regular arrays
        and ``'i'`` for constant arrays.
    :param constant: if ``True``, corresponds to a constant (cached) array.
    """

    def __init__(self, type_, role=None, constant=False):
        self.type = Type.from_value(type_)

        if role is None:
            if len(self.type.shape) == 0:
                role = 's'
            elif constant:
                role = 'i'
            else:
                role = 'io'

        assert role in ('i', 'o', 'io', 's')
        self.role = role
        self.constant = constant
        if role == 's':
            self.array = False
            self.input = False
            self.output = False
        else:
            self.array = True
            self.input = 'i' in role
            self.output = 'o' in role

    def __eq__(self, other):
        return (self.type == other.type
            and self.role == other.role and self.constant == other.constant)

    def can_be_argument_for(self, annotation):
        if not self.type.compatible_with(annotation.type):
            return False

        if self.role == annotation.role:
            return True

        if self.role == 'io' and annotation.array:
            return True

        return False

    def __repr__(self):
        if self.array:
            return "Annotation({type_}, role={role}{constant})".format(
                type_=self.type, role=repr(self.role),
                constant=", constant" if self.constant else "")
        else:
            return "Annotation({dtype})".format(dtype=self.type.dtype)

    def __process_modules__(self, process):
        ann = Annotation(self.type, role=self.role, constant=self.constant)
        ann.type = process(ann.type)
        return ann


class Parameter(funcsigs.Parameter):
    """
    Computation parameter,
    in the same sense as it is used for functions in the standard library.
    In its terms, all computation parameters have kind ``POSITIONAL_OR_KEYWORD``.

    :param name: parameter name.
    :param annotation: an :py:class:`~reikna.core.Annotation` object.
    :param default: default value for the parameter, can only be specified for scalars.
    """

    def __init__(self, name, annotation, default=funcsigs.Parameter.empty):

        if default is not funcsigs.Parameter.empty:
            if annotation.array:
                raise ValueError("Array parameters cannot have default values")
            default = annotation.type(default)

        # HACK: Parameter constructor is not documented.
        # But I need to create these objects somehow.
        funcsigs.Parameter.__init__(
            self, name, annotation=annotation,
            kind=funcsigs.Parameter.POSITIONAL_OR_KEYWORD,
            default=default)

    def rename(self, new_name):
        """
        Creates a new :py:class:`Parameter` object with the new name
        and the same annotation and default value.
        """
        return Parameter(new_name, self.annotation, default=self.default)

    def __eq__(self, other):
        return (self.name == other.name and self.annotation == other.annotation
            and self.default == other.default)

    def __process_modules__(self, process):
        return Parameter(self.name, process(self.annotation), default=self.default)


class Signature(funcsigs.Signature):
    """
    Computation signature,
    in the same sense as it is used for functions in the standard library.

    :param parameters: a list of :py:class:`~reikna.core.Parameter` objects.

    .. py:attribute:: parameters

        An ``OrderedDict`` with :py:class:`~reikna.core.Parameter` objects indexed by their names.
    """

    def __init__(self, parameters):
        # HACK: Signature constructor is not documented.
        # But I need to create these objects somehow.
        funcsigs.Signature.__init__(self, parameters)

    def bind_with_defaults(self, args, kwds, cast=False):
        """
        Binds passed positional and keyword arguments to parameters in the signature and
        returns the resulting ``BoundArguments`` object.
        """
        bound_args = self.bind(*args, **kwds)
        for param in self.parameters.values():
            if param.name not in bound_args.arguments:
                bound_args.arguments[param.name] = param.default
            elif cast:
                if not param.annotation.array:
                    bound_args.arguments[param.name] = \
                        param.annotation.type(bound_args.arguments[param.name])
        return bound_args

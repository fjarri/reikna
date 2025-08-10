import inspect
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Callable, Iterable, Mapping, Sequence, cast

import numpy
from grunnur import Array, ArrayMetadata, AsArrayMetadata, dtypes
from numpy.typing import DTypeLike

from .. import helpers
from ..helpers import product, wrap_in_tuple

if TYPE_CHECKING:
    from grunnur import Module


def is_compatible_with(lhs: ArrayMetadata, rhs: ArrayMetadata) -> bool:
    """
    Returns ``True`` if this and the other metadata represent essentially the same array,
    with the only difference being some outermost dimensions of size 1.
    """

    if lhs.dtype != rhs.dtype:
        return False

    common_shape_len = min(len(lhs.shape), len(rhs.shape))
    if lhs.shape[-common_shape_len:] != rhs.shape[-common_shape_len:]:
        return False
    if lhs.strides[-common_shape_len:] != rhs.strides[-common_shape_len:]:
        return False
    if helpers.product(lhs.shape[:-common_shape_len]) != 1:
        return False
    if helpers.product(rhs.shape[:-common_shape_len]) != 1:
        return False
    if lhs.first_element_offset != rhs.first_element_offset:
        return False

    return True


class Type(AsArrayMetadata):
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

    # TODO: this is temporary to use in `cast()` transformation.
    # Currently it can take both grunnur arrays and numpy arrays,
    # and they have different names for the offset and the buffer size.
    # Think of something more type-safe.
    @classmethod
    def like(cls, array_like: AsArrayMetadata | numpy.ndarray[Any, numpy.dtype[Any]]) -> "Type":
        if isinstance(array_like, AsArrayMetadata):
            return cls._from_metadata(array_like.as_array_metadata())
        else:
            return cls.array(
                shape=array_like.shape, dtype=array_like.dtype, strides=array_like.strides
            )

    @classmethod
    def array(
        cls,
        dtype: numpy.dtype[Any],
        shape: Sequence[int] | int,
        strides: Sequence[int] | None = None,
        offset: int = 0,
        nbytes: int | None = None,
    ) -> "Type":
        metadata = ArrayMetadata(
            dtype=dtype,
            shape=shape,
            strides=strides,
            first_element_offset=offset,
            buffer_size=nbytes,
        )
        return cls._from_metadata(metadata)

    @classmethod
    def _from_metadata(cls, metadata: ArrayMetadata | DTypeLike) -> "Type":
        if not isinstance(metadata, ArrayMetadata):
            metadata = numpy.dtype(metadata)
        ctype = dtypes.ctype(metadata.dtype if isinstance(metadata, ArrayMetadata) else metadata)
        return cls(metadata, ctype)

    @classmethod
    def scalar(cls, dtype: DTypeLike) -> "Type":
        return cls._from_metadata(dtype)

    def __init__(self, metadata: ArrayMetadata | numpy.dtype[Any], ctype: "str | Module"):
        if not isinstance(metadata, ArrayMetadata):
            metadata = numpy.dtype(metadata)
        self._metadata = metadata
        self.ctype = ctype

    def as_array_metadata(self) -> ArrayMetadata:
        return self.array_metadata

    @property
    def array_metadata(self) -> ArrayMetadata:
        # TODO: can this branch be eliminated by typing?
        if isinstance(self._metadata, ArrayMetadata):
            return self._metadata
        raise ValueError("This is a scalar type")

    @property
    def dtype(self) -> numpy.dtype[Any]:
        if isinstance(self._metadata, ArrayMetadata):
            return self._metadata.dtype
        else:
            return self._metadata

    @property
    def shape(self) -> tuple[int, ...]:
        if isinstance(self._metadata, ArrayMetadata):
            return self._metadata.shape
        else:
            return ()

    @property
    def strides(self) -> tuple[int, ...]:
        if isinstance(self._metadata, ArrayMetadata):
            return self._metadata.strides
        else:
            return ()

    @property
    def offset(self) -> int:
        if isinstance(self._metadata, ArrayMetadata):
            return self._metadata.first_element_offset
        else:
            return 0

    def __eq__(self, other: Any) -> Any:
        return self._metadata == other._metadata

    def __hash__(self) -> int:
        return hash((type(self), self._metadata))

    def is_scalar(self) -> bool:
        return isinstance(self._metadata, numpy.dtype)

    # TODO: move to `Annotation`
    def compatible_with(self, other: "Type") -> bool:
        if self.is_scalar() and other.is_scalar():
            return self._metadata == other._metadata

        if self.is_scalar() ^ other.is_scalar():
            return False

        return is_compatible_with(
            cast(ArrayMetadata, self._metadata), cast(ArrayMetadata, other._metadata)
        )

    # TODO: move the logic to `transformations.copy_broadcasted()`
    def broadcastable_to(self, other: "Type") -> bool:
        """
        Returns ``True`` if the shape of this ``Type`` is broadcastable to ``other``,
        that is its dimensions either coincide with the innermost dimensions of ``other.shape``,
        or are equal to 1.
        """
        if self.is_scalar() or other.is_scalar():
            return False

        if len(self._metadata.shape) > len(other._metadata.shape):
            return False

        for i in range(1, len(self._metadata.shape) + 1):
            if not (
                self._metadata.shape[-i] == 1
                or self._metadata.shape[-i] == other._metadata.shape[-i]
            ):
                return False

        return True

    def with_dtype(self, dtype: numpy.dtype[Any]) -> "Type":
        """
        Creates a :py:class:`Type` object with its ``dtype`` attribute replaced by the given dtype.
        """
        if isinstance(self._metadata, ArrayMetadata):
            return Type.array(
                dtype=dtype,
                shape=self._metadata.shape,
                strides=self._metadata.strides,
                offset=self._metadata.first_element_offset,
                nbytes=self._metadata.buffer_size,
            )
        else:
            return Type._from_metadata(dtype)

    @classmethod
    def from_value(cls, val: Any) -> "Type":
        """
        Creates a :py:class:`Type` object corresponding to the given value.
        """

        if isinstance(val, Type):
            # Creating a new object, because ``val`` may be some derivative of Type,
            # used as a syntactic sugar, and we do not want it to confuse us later.
            return cls._from_metadata(val._metadata)
        elif isinstance(val, Array):
            return cls._from_metadata(val.metadata)
        elif isinstance(val, ArrayMetadata):
            return cls._from_metadata(val)
        elif isinstance(val, numpy.dtype):
            return cls._from_metadata(val)
        elif isinstance(val, type) and issubclass(val, numpy.generic):
            return cls._from_metadata(numpy.dtype(val))
        elif hasattr(val, "dtype") and hasattr(val, "shape"):
            strides = val.strides if hasattr(val, "strides") else None
            offset = val.offset if hasattr(val, "offset") else 0
            nbytes = val.nbytes if hasattr(val, "nbytes") else None
            return cls.array(
                dtype=val.dtype, shape=val.shape, strides=strides, offset=offset, nbytes=nbytes
            )
        else:
            return cls._from_metadata(dtypes.result_type(val))

    def __call__(self, val: Any) -> numpy.ndarray[Any, numpy.dtype[Any]]:
        """
        Casts the given value to this type.
        """
        return numpy.asarray(val, dtype=self.dtype)

    def __repr__(self) -> str:
        return f"Type({self._metadata})"


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

    def __init__(self, type_: Any, role: str | None = None, constant: bool = False):
        self.type = Type.from_value(type_)

        if role is None:
            if len(self.type.shape) == 0:
                role = "s"
            elif constant:
                role = "i"
            else:
                role = "io"

        assert role in ("i", "o", "io", "s")
        self.role = role
        self.constant = constant
        if role == "s":
            assert self.type.is_scalar()
            self.array = False
            self.input = False
            self.output = False
        else:
            assert not self.type.is_scalar()
            self.array = True
            self.input = "i" in role
            self.output = "o" in role

    def __eq__(self, other: Any) -> Any:
        return (
            self.type == other.type and self.role == other.role and self.constant == other.constant
        )

    def can_be_argument_for(self, annotation: "Annotation") -> bool:
        if not self.type.compatible_with(annotation.type):
            return False

        if self.role == annotation.role:
            return True

        if self.role == "io" and annotation.array:
            return True

        return False

    def __repr__(self) -> str:
        if self.array:
            return "Annotation({type_}, role={role}{constant})".format(
                type_=self.type,
                role=repr(self.role),
                constant=", constant" if self.constant else "",
            )
        else:
            return "Annotation({dtype})".format(dtype=self.type.dtype)


class Parameter(inspect.Parameter):
    """
    Computation parameter,
    in the same sense as it is used for functions in the standard library.
    In its terms, all computation parameters have kind ``POSITIONAL_OR_KEYWORD``.

    :param name: parameter name.
    :param annotation: an :py:class:`~reikna.core.Annotation` object.
    :param default: default value for the parameter, can only be specified for scalars.
    """

    def __init__(
        self,
        name: str,
        annotation: Annotation,
        default: Any = inspect.Parameter.empty,
    ):
        if default is not inspect.Parameter.empty:
            if annotation.array:
                raise ValueError("Array parameters cannot have default values")
            default = annotation.type(default)

        # HACK: Parameter constructor is not documented.
        # But I need to create these objects somehow.
        inspect.Parameter.__init__(
            self,
            name,
            annotation=annotation,
            kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
            default=default,
        )

    def rename(self, new_name: str) -> "Parameter":
        """
        Creates a new :py:class:`Parameter` object with the new name
        and the same annotation and default value.
        """
        return Parameter(new_name, self.annotation, default=self.default)

    def __eq__(self, other: Any) -> Any:
        return (
            self.name == other.name
            and self.annotation == other.annotation
            and self.default == other.default
        )


class Signature(inspect.Signature):
    """
    Computation signature,
    in the same sense as it is used for functions in the standard library.

    :param parameters: a list of :py:class:`~reikna.core.Parameter` objects.

    .. py:attribute:: parameters

        An ``OrderedDict`` with :py:class:`~reikna.core.Parameter` objects indexed by their names.
    """

    def __init__(self, parameters: Sequence[Parameter]):
        # HACK: Signature constructor is not documented.
        # But I need to create these objects somehow.
        inspect.Signature.__init__(self, parameters)

    @property
    def _reikna_parameters(self) -> Mapping[str, Parameter]:
        # TODO: a temporary solution to make typing work.
        # Ideally we should not override inspect.Signature/Parameter types.
        return cast(Mapping[str, Parameter], self.parameters)

    def bind_with_defaults(
        self, args: tuple[Any, ...], kwds: Mapping[str, Any], cast: bool = False
    ) -> inspect.BoundArguments:
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
                    bound_args.arguments[param.name] = param.annotation.type(
                        bound_args.arguments[param.name]
                    )
        return bound_args

import inspect
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, cast

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
    return lhs.first_element_offset == rhs.first_element_offset


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

    def __init__(self, metadata: "Type | AsArrayMetadata | DTypeLike"):
        normalized_metadata: ArrayMetadata | numpy.dtype[Any]
        if isinstance(metadata, Type):
            # Strip all information the types derived from `Type` may have
            normalized_metadata = metadata._raw_metadata()  # noqa: SLF001
        elif isinstance(metadata, AsArrayMetadata):
            normalized_metadata = metadata.as_array_metadata()
        else:
            normalized_metadata = numpy.dtype(metadata)
        self._metadata = normalized_metadata
        self.ctype = dtypes.ctype(
            self._metadata.dtype if isinstance(self._metadata, ArrayMetadata) else self._metadata
        )

    def _raw_metadata(self) -> ArrayMetadata | numpy.dtype[Any]:
        return self._metadata

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
        return self._metadata

    @property
    def shape(self) -> tuple[int, ...]:
        return self.array_metadata.shape

    @property
    def strides(self) -> tuple[int, ...]:
        return self.array_metadata.strides

    @property
    def offset(self) -> int:
        return self.array_metadata.first_element_offset

    def __eq__(self, other: object) -> Any:
        return isinstance(other, Type) and self._metadata == other._metadata

    def __hash__(self) -> int:
        return hash((type(self), self._metadata))

    def is_scalar(self) -> bool:
        return isinstance(self._metadata, numpy.dtype)

    def is_array(self) -> bool:
        return not self.is_scalar()

    def compatible_with(self, other: "Type") -> bool:
        if self.is_scalar() and other.is_scalar():
            return self._metadata == other._metadata  # noqa: SLF001

        if self.is_scalar() ^ other.is_scalar():
            return False

        return is_compatible_with(
            cast("ArrayMetadata", self._metadata),
            cast("ArrayMetadata", other._metadata),  # noqa: SLF001
        )

    def with_dtype(self, dtype: numpy.dtype[Any]) -> "Type":
        """
        Creates a :py:class:`Type` object with its ``dtype``
        attribute replaced by the given dtype.
        """
        if isinstance(self._metadata, ArrayMetadata):
            return Type(self._metadata.with_(dtype=dtype))
        return Type(dtype)

    @classmethod
    def from_value(cls, val: Any) -> "Type":
        """Creates a :py:class:`Type` object corresponding to the given value."""
        if isinstance(val, Type):
            # Creating a new object, because ``val`` may be some derivative of Type,
            # used as a syntactic sugar, and we do not want it to confuse us later.
            return cls(val)
        if isinstance(val, Array):
            return cls(val.metadata)
        if isinstance(val, ArrayMetadata | numpy.dtype):
            return cls(val)
        if isinstance(val, type) and issubclass(val, numpy.generic):
            return cls(numpy.dtype(val))
        if hasattr(val, "dtype") and hasattr(val, "shape"):
            strides = val.strides if hasattr(val, "strides") else None
            offset = val.offset if hasattr(val, "offset") else 0
            nbytes = val.nbytes if hasattr(val, "nbytes") else None
            return cls(
                ArrayMetadata(
                    dtype=val.dtype,
                    shape=val.shape,
                    strides=strides,
                    first_element_offset=offset,
                    buffer_size=nbytes,
                )
            )
        return cls(dtypes.result_type(val))

    def cast_scalar(self, val: numpy.generic | complex) -> numpy.generic:
        """Casts the given value to this type."""
        if not self.is_scalar():
            raise ValueError("Can only cast scalars to a scalar type, this is an array")
        return cast("numpy.generic", numpy.asarray(val, dtype=self.dtype).flat[0])

    def __repr__(self) -> str:
        return f"Type({self._metadata})"


class Annotation(Type):
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

    def __init__(
        self,
        type_: Type | AsArrayMetadata | DTypeLike,
        role: str | None = None,
        *,
        constant: bool = False,
    ):
        super().__init__(type_)

        if role is None:
            if self.is_scalar():
                role = "s"
            elif constant:
                role = "i"
            else:
                role = "io"

        if role not in ("i", "o", "io", "s"):
            raise ValueError(f"Invalid role: {role}")
        self.role = role
        self.constant = constant
        if role == "s":
            if not self.is_scalar():
                raise ValueError("Only scalars can have the scalar role")
            self.input = False
            self.output = False
        else:
            if self.is_scalar():
                raise ValueError("Scalars cannot have input or output role")
            self.input = "i" in role
            self.output = "o" in role

    def can_be_argument_for(self, annotation: "Annotation") -> bool:
        if not self.compatible_with(annotation):
            return False

        if self.role == annotation.role:
            return True

        return self.role == "io" and annotation.is_array()

    def _as_type(self) -> Type:
        return Type(self)

    def __repr__(self) -> str:
        return "Annotation({metadata}, role={role}{constant})".format(
            metadata=self._metadata,
            role=self.role,
            constant=", constant" if self.constant else "",
        )


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
            if annotation.is_array():
                raise ValueError("Array parameters cannot have default values")
            default = annotation.cast_scalar(default)

        # TODO: Parameter constructor is not documented.
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


class Signature(inspect.Signature):
    """
    Computation signature,
    in the same sense as it is used for functions in the standard library.

    :param parameters: a list of :py:class:`~reikna.core.Parameter` objects.

    .. py:attribute:: parameters

        An ``OrderedDict`` with :py:class:`~reikna.core.Parameter` objects indexed by their names.
    """

    def __init__(self, parameters: Sequence[Parameter]):
        # TODO: Signature constructor is not documented.
        # But I need to create these objects somehow.
        inspect.Signature.__init__(self, parameters)

    @property
    def _reikna_parameters(self) -> Mapping[str, Parameter]:
        # TODO: a temporary solution to make typing work.
        # Ideally we should not override inspect.Signature/Parameter types.
        return cast("Mapping[str, Parameter]", self.parameters)

    def bind_with_defaults(
        self, args: tuple[Any, ...], kwds: Mapping[str, Any], *, cast: bool = False
    ) -> inspect.BoundArguments:
        """
        Binds passed positional and keyword arguments to parameters in the signature and
        returns the resulting ``BoundArguments`` object.
        """
        bound_args = self.bind(*args, **kwds)
        for param in self.parameters.values():
            if param.name not in bound_args.arguments:
                bound_args.arguments[param.name] = param.default
            elif cast and param.annotation.is_scalar():
                bound_args.arguments[param.name] = param.annotation.cast_scalar(
                    bound_args.arguments[param.name]
                )
        return bound_args

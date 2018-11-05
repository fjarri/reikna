import itertools
from fractions import gcd

import numpy
from reikna.helpers import bounding_power_of_2, log2, min_blocks
from reikna.cluda.api_discovery import cuda_id, ocl_id


_DTYPE_TO_BUILTIN_CTYPE = {}
_DTYPE_TO_CTYPE_MODULE = {}


def is_complex(dtype):
    """
    Returns ``True`` if ``dtype`` is complex.
    """
    dtype = normalize_type(dtype)
    return numpy.issubdtype(dtype, numpy.complexfloating)

def is_double(dtype):
    """
    Returns ``True`` if ``dtype`` is double precision floating point.
    """
    dtype = normalize_type(dtype)
    return numpy.issubdtype(dtype, numpy.float_) or numpy.issubdtype(dtype, numpy.complex_)

def is_integer(dtype):
    """
    Returns ``True`` if ``dtype`` is an integer.
    """
    dtype = normalize_type(dtype)
    return numpy.issubdtype(dtype, numpy.integer)

def is_real(dtype):
    """
    Returns ``True`` if ``dtype`` is a real.
    """
    dtype = normalize_type(dtype)
    return numpy.issubdtype(dtype, numpy.floating)

def _promote_dtype(dtype):
    # not all numpy datatypes are supported by GPU, so we may need to promote
    dtype = normalize_type(dtype)
    if dtype.kind == 'i' and dtype.itemsize < 4:
        dtype = numpy.int32
    elif dtype.kind == 'f' and dtype.itemsize < 4:
        dtype = numpy.float32
    elif dtype.kind == 'c' and dtype.itemsize < 8:
        dtype = numpy.complex64
    return normalize_type(dtype)

def result_type(*dtypes):
    """
    Wrapper for ``numpy.result_type``
    which takes into account types supported by GPUs.
    """
    return _promote_dtype(numpy.result_type(*dtypes))

def min_scalar_type(val):
    """
    Wrapper for ``numpy.min_scalar_dtype``
    which takes into account types supported by GPUs.
    """
    return _promote_dtype(numpy.min_scalar_type(val))

def detect_type(val):
    """
    Find out the data type of ``val``.
    """
    if hasattr(val, 'dtype'):
        return _promote_dtype(val.dtype)
    else:
        return min_scalar_type(val)

def normalize_type(dtype):
    """
    Function for wrapping all dtypes coming from the user.
    ``numpy`` uses two different classes to represent dtypes,
    and one of them does not have some important attributes.
    """
    return numpy.dtype(dtype)

def normalize_types(dtypes):
    """
    Same as :py:func:`normalize_type`, but operates on a list of dtypes.
    """
    return [normalize_type(dtype) for dtype in dtypes]

def complex_for(dtype):
    """
    Returns complex dtype corresponding to given floating point ``dtype``.
    """
    dtype = normalize_type(dtype)
    return numpy.dtype(dict(float32='complex64', float64='complex128')[dtype.name])

def real_for(dtype):
    """
    Returns floating point dtype corresponding to given complex ``dtype``.
    """
    dtype = normalize_type(dtype)
    return numpy.dtype(dict(complex64='float32', complex128='float64')[dtype.name])

def complex_ctr(dtype):
    """
    Returns name of the constructor for the given ``dtype``.
    """
    return 'COMPLEX_CTR(' + ctype(dtype) + ')'

def zero_ctr(dtype):
    """
    Returns the string with constructed zero value for the given ``dtype``.
    """
    if is_complex(dtype):
        return complex_ctr(dtype) + '(0, 0)'
    else:
        return '0'

def cast(dtype):
    """
    Returns function that takes one argument and casts it to ``dtype``.
    """
    def _cast(val):
        # Numpy cannot handle casts to struct dtypes (#4148),
        # so we're avoiding unnecessary casts.
        if not hasattr(val, 'dtype'):
            # A non-numpy scalar
            return numpy.array([val], dtype)[0]
        elif val.dtype != dtype:
            return numpy.cast[dtype](val)
        else:
            return val
    return _cast

def _c_constant_arr(val, shape):
    if len(shape) == 0:
        return c_constant(val)
    else:
        return "{" + ", ".join(_c_constant_arr(val[i], shape[1:]) for i in range(shape[0])) + "}"

def c_constant(val, dtype=None):
    """
    Returns a C-style numerical constant.
    If ``val`` has a struct dtype, the generated constant will have the form ``{ ... }``
    and can be used as an initializer for a variable.
    """
    if dtype is None:
        dtype = detect_type(val)
    else:
        dtype = normalize_type(dtype)

    val = cast(dtype)(val)

    if len(val.shape) > 0:
        return _c_constant_arr(val, val.shape)
    elif dtype.names is not None:
        return "{" + ", ".join([c_constant(val[name]) for name in dtype.names]) + "}"

    if is_complex(dtype):
        return "COMPLEX_CTR(" + ctype(dtype) + ")(" + \
            c_constant(val.real) + ", " + c_constant(val.imag) + ")"
    elif is_integer(dtype):
        if dtype.itemsize > 4:
            postfix = "L" if numpy.issubdtype(dtype, numpy.signedinteger) else "UL"
        else:
            postfix = ""
        return str(val) + postfix
    else:
        return repr(float(val)) + ("f" if dtype.itemsize <= 4 else "")

def _register_dtype(dtype, ctype_str):
    dtype = normalize_type(dtype)
    _DTYPE_TO_BUILTIN_CTYPE[dtype] = ctype_str

# Taken from compyte.dtypes
def _fill_dtype_registry(respect_windows=True):

    import sys
    import platform

    _register_dtype(numpy.bool, "bool")
    _register_dtype(numpy.int8, "char")
    _register_dtype(numpy.uint8, "unsigned char")
    _register_dtype(numpy.int16, "short")
    _register_dtype(numpy.uint16, "unsigned short")
    _register_dtype(numpy.int32, "int")
    _register_dtype(numpy.uint32, "unsigned int")

    if platform.system() == 'Windows' and respect_windows:
        i64_name = "long long"
    else:
        i64_name = "long"

    _register_dtype(numpy.int64, i64_name)
    _register_dtype(numpy.uint64, "unsigned %s" % i64_name)

    _register_dtype(numpy.float32, "float")
    _register_dtype(numpy.float64, "double")
    _register_dtype(numpy.complex64, "float2")
    _register_dtype(numpy.complex128, "double2")

_fill_dtype_registry()


def ctype(dtype):
    """
    For a built-in C type, returns a string with the name of the type.
    """
    return _DTYPE_TO_BUILTIN_CTYPE[normalize_type(dtype)]


def _alignment_str(alignment):
    if alignment is not None:
        return "ALIGN(" + str(alignment) + ")"
    else:
        return ""


def _lcm(*nums):
    """
    Returns the least common multiple of ``nums``.
    """
    if len(nums) == 1:
        return nums[0]
    elif len(nums) == 2:
        return nums[0] * nums[1] // gcd(nums[0], nums[1])
    else:
        return _lcm(nums[0], _lcm(*nums[1:]))


def _struct_alignment(alignments):
    """
    Returns the minimum alignment for a structure given alignments for its fields.
    According to the C standard, it the lowest common multiple of the alignments
    of all of the members of the struct rounded up to the nearest power of two.
    """
    return bounding_power_of_2(_lcm(*alignments))


def _find_minimal_alignment(offset, base_alignment, prev_end):
    """
    Returns the minimal alignment that must be set for a field with
    ``base_alignment`` (the one inherent to the type),
    so that the compiler positioned it at ``offset`` given that the previous field
    ends at the position ``prev_end``.
    """
    # Essentially, we need to find the minimal k such that:
    # 1) offset = m * base_alignment * 2**k, where m > 0 and k >= 0;
    #    (by definition of alignment)
    # 2) offset - prev_offset < base_alignment * 2**k
    #    (otherwise the compiler can just as well take m' = m - 1).
    alignment = base_alignment
    while True:
        if offset % alignment != 0:
            raise ValueError(
                ("Field cannot be positioned at offset {offset}, "
                "since it is not multiple of the minimal alignment {alignment}").format(
                    offset=offset, alignment=alignment))
        if alignment > offset:
            raise ValueError(
                "Could not find a suitable alignment for the field at offset {offset}".format(
                    offset=offset))
        if offset - prev_end >= alignment:
            alignment *= 2
            continue

        return alignment


def _find_alignments(dtype):
    """
    Returns a tuple (base_alignment, field_alignments) for the given dtype, where:

    ``field_alignments`` is a dictionary ``{name:alignment}`` with base alignments
    for every field of ``dtype``.

    ``base_alignment`` is the base alignment for the whole C type corresponding to ``dtype``.
    """

    # FIXME: for vector types with 3 components the alignment is 4*component_size.
    # But we do not support these at the moment anyway.
    if dtype.names is None:
        return dtype.base.itemsize, None

    field_alignments = {}
    explicit_alignments = []
    for i, name in enumerate(dtype.names):
        elem_dtype, elem_offset = dtype.fields[name]
        base_elem_dtype = elem_dtype.base

        base_alignment, _ = _find_alignments(base_elem_dtype)

        if i == 0:
            # The alignment of the first element does not matter, its offset is 0
            explicit_alignments.append(base_alignment)
            field_alignments[name] = None
        else:
            prev_dtype, prev_offset = dtype.fields[dtype.names[i-1]]
            prev_end = prev_offset + prev_dtype.itemsize
            alignment = _find_minimal_alignment(elem_offset, base_alignment, prev_end)

            explicit_alignments.append(alignment)
            field_alignments[name] = alignment if alignment != base_alignment else None

    struct_alignment = _struct_alignment(explicit_alignments)
    last_dtype, last_offset = dtype.fields[dtype.names[-1]]
    last_end = last_offset + last_dtype.itemsize
    struct_alignment = _find_minimal_alignment(dtype.itemsize, struct_alignment, last_end)

    return struct_alignment, field_alignments


def _get_struct_module(dtype, ignore_alignment=False):
    """
    Builds and returns a module with the C type definition for a given ``dtype``,
    possibly using modules for nested structures.
    """
    if not ignore_alignment:
        struct_alignment, field_alignments = _find_alignments(dtype)

    # FIXME: the tag (${prefix}_) is not necessary, but it helps to avoid
    # CUDA bug #1409907 (nested struct initialization like
    # "mystruct x = {0, {0, 0}, 0};" fails to compile)
    lines = ["typedef struct ${prefix}_ {"]
    kwds = {}
    for name in dtype.names:
        elem_dtype, elem_offset = dtype.fields[name]

        if len(elem_dtype.shape) == 0:
            base_elem_dtype = elem_dtype
            elem_dtype_shape = tuple()
        else:
            base_elem_dtype = elem_dtype.base
            elem_dtype_shape = elem_dtype.shape

        if len(elem_dtype_shape) == 0:
            array_suffix = ""
        else:
            array_suffix = "".join("[" + str(d) + "]" for d in elem_dtype_shape)

        typename_var = "typename_" + name
        field_alignment = None if ignore_alignment else field_alignments[name]
        lines.append(
            "    ${" + typename_var + "} " +
            _alignment_str(field_alignment) + " " +
            name + array_suffix + ";")
        kwds[typename_var] = ctype_module(base_elem_dtype, ignore_alignment=ignore_alignment)

    struct_alignment = None if ignore_alignment else struct_alignment
    lines.append("} " + _alignment_str(struct_alignment) + " ${prefix};")

    # Root level import creates an import loop.
    from reikna.cluda.kernel import Module

    return Module.create("\n".join(lines), render_kwds=kwds)


def ctype_module(dtype, ignore_alignment=False):
    """
    For a struct type, returns a :py:class:`~reikna.cluda.Module` object
    with the ``typedef`` of a struct corresponding to the given ``dtype``
    (with its name set to the module prefix);
    falls back to :py:func:`~reikna.cluda.dtypes.ctype` otherwise.

    The structure definition includes the alignment required
    to produce field offsets specified in ``dtype``;
    therefore, ``dtype`` must be either a simple type, or have
    proper offsets and dtypes (the ones that can be reporoduced in C
    using explicit alignment attributes, but without additional padding)
    and the attribute ``isalignedstruct == True``.
    An aligned dtype can be produced either by standard means
    (``aligned`` flag in ``numpy.dtype`` constructor and explicit offsets and itemsizes),
    or created out of an arbitrary dtype with the help of :py:func:`~reikna.cluda.dtypes.align`.

    If ``ignore_alignment`` is True, all of the above is ignored.
    The C structures produced will not have any explicit alignment modifiers.
    As a result, the the field offsets of ``dtype`` may differ from the ones
    chosen by the compiler.

    Modules are cached and the function returns a single module instance for equal ``dtype``'s.
    Therefore inside a kernel it will be rendered with the same prefix everywhere it is used.
    This results in a behavior characteristic for a structural type system,
    same as for the basic dtype-ctype conversion.

    .. warning::

        As of ``numpy`` 1.8, the ``isalignedstruct`` attribute is not enough to ensure
        a mapping between a dtype and a C struct with only the fields that are present in the dtype.
        Therefore, ``ctype_module`` will make some additional checks and raise ``ValueError``
        if it is not the case.
    """
    dtype = normalize_type(dtype)

    if dtype.names is None:
        return ctype(dtype)
    else:
        # FIXME: if numpy's ``isalignedstruct`` actually meant that the struct is aligned,
        # that would be enough.
        # Unfortunately, it only recognizes base alignments,
        # and does not check for itemsize consistency.
        # Therefore, there will be more checking in _get_struct_module.
        if not ignore_alignment and not dtype.isalignedstruct:
            raise ValueError("The data type must be an aligned struct")

        if len(dtype.shape) > 0:
            raise ValueError("The data type cannot be an array")

        if dtype not in _DTYPE_TO_CTYPE_MODULE:
            module = _get_struct_module(dtype, ignore_alignment=ignore_alignment)
            _DTYPE_TO_CTYPE_MODULE[dtype] = module
        else:
            module = _DTYPE_TO_CTYPE_MODULE[dtype]

        return module


def align(dtype):
    """
    Returns a new struct dtype with the field offsets changed to the ones a compiler would use
    (without being given any explicit alignment qualifiers).
    Ignores all existing explicit itemsizes and offsets.
    """
    dtype = normalize_type(dtype)

    if len(dtype.shape) > 0:
        return numpy.dtype((align(dtype.base), dtype.shape))

    if dtype.names is None:
        return dtype

    # Align the nested fields
    adjusted_fields = [
        align(dtype.fields[name][0])
        for name in dtype.names]

    # Get base alignments for the nested fields
    alignments = [_find_alignments(field_dtype)[0] for field_dtype in adjusted_fields]

    # Build offsets for the structure using a procedure
    # similar to the one a compiler would use
    offsets = [0]
    for name, prev_field_dtype, alignment in zip(
            dtype.names[1:], adjusted_fields[:-1], alignments[1:]):
        prev_end = offsets[-1] + prev_field_dtype.itemsize
        offsets.append(min_blocks(prev_end, alignment) * alignment)

    # Find the total itemsize.
    # According to the standard, it must be a multiple of the minimal alignment.
    struct_alignment = _struct_alignment(alignments)
    min_itemsize = offsets[-1] + adjusted_fields[-1].itemsize
    itemsize = min_blocks(min_itemsize, struct_alignment) * struct_alignment

    return numpy.dtype(dict(
        names=dtype.names,
        formats=adjusted_fields,
        offsets=offsets,
        itemsize=itemsize,
        aligned=True))


def _flatten_dtype(dtype, prefix=[]):

    if dtype.names is None:
        return [(prefix, dtype)]
    else:
        result = []
        for name in dtype.names:
            elem_dtype, _ = dtype.fields[name]
            if len(elem_dtype.shape) == 0:
                base_elem_dtype = elem_dtype
                elem_dtype_shape = tuple()
            else:
                base_elem_dtype = elem_dtype.base
                elem_dtype_shape = elem_dtype.shape

            if len(elem_dtype_shape) == 0:
                result += _flatten_dtype(base_elem_dtype, prefix=prefix + [name])
            else:
                for idxs in itertools.product(*[range(dim) for dim in elem_dtype_shape]):
                    result += _flatten_dtype(base_elem_dtype, prefix=prefix + [name] + list(idxs))
        return result


def flatten_dtype(dtype):
    """
    Returns a list of tuples ``(path, dtype)`` for each of the basic dtypes in
    a (possibly nested) ``dtype``.
    ``path`` is a list of field names/array indices leading to the corresponding element.
    """
    dtype = normalize_type(dtype)
    return _flatten_dtype(dtype)


def c_path(path):
    """
    Returns a string corresponding to the ``path`` to a struct element in C.
    The ``path`` is the sequence of field names/array indices returned from
    :py:func:`~reikna.cluda.dtypes.flatten_dtype`.
    """
    return  "".join(
        (("." + elem) if isinstance(elem, str) else ("[" + str(elem) + "]"))
        for elem in path)


def _extract_field(arr, path, array_idxs):
    """
    A helper function for ``extract_field``.
    Need to collect array indices for dtype sub-array fields since they are attached to the end
    of the full array index.
    """
    if len(path) == 0:
        if len(array_idxs) == 0:
            return arr
        else:
            slices = tuple(
                [slice(None, None, None)] * (len(arr.shape) - len(array_idxs)) + array_idxs)
            return arr[slices]
    elif isinstance(path[0], str):
        return _extract_field(arr[path[0]], path[1:], array_idxs)
    else:
        return _extract_field(arr, path[1:], array_idxs + [path[0]])


def extract_field(arr, path):
    """
    Extracts an element from an array of struct dtype.
    The ``path`` is the sequence of field names/array indices returned from
    :py:func:`~reikna.cluda.dtypes.flatten_dtype`.
    """
    return _extract_field(arr, path, [])

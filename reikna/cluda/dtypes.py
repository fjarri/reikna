import numpy
from reikna.cluda.api_discovery import cuda_id, ocl_id


_DTYPE_TO_BUILTIN_CTYPE = {}
_DTYPE_TO_CTYPE_MODULE = {}


def is_complex(dtype):
    """
    Returns ``True`` if ``dtype`` is complex.
    """
    dtype = normalize_type(dtype)
    return dtype.kind == 'c'

def is_double(dtype):
    """
    Returns ``True`` if ``dtype`` is double precision floating point.
    """
    dtype = normalize_type(dtype)
    return dtype.name in ['float64', 'complex128']

def is_integer(dtype):
    """
    Returns ``True`` if ``dtype`` is an integer.
    """
    dtype = normalize_type(dtype)
    return dtype.kind in ('i', 'u')

def is_real(dtype):
    """
    Returns ``True`` if ``dtype`` is a real.
    """
    dtype = normalize_type(dtype)
    return dtype.kind == 'f'

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
    return numpy.cast[dtype]

def c_constant(val, dtype=None):
    """
    Returns a C-style numerical constant.
    """
    if dtype is None:
        dtype = detect_type(val)
    else:
        dtype = normalize_type(dtype)
    val = numpy.cast[dtype](val)

    if is_complex(dtype):
        return "COMPLEX_CTR(" + ctype(dtype) + ")(" + \
            c_constant(val.real) + ", " + c_constant(val.imag) + ")"
    elif is_integer(dtype):
        return str(val) + ("L" if dtype.itemsize > 4 else "")
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

    # recommended by Python docs
    is_64bits = sys.maxsize > 2 ** 32

    if is_64bits:
        if platform.system == 'Windows' and respect_windows:
            i64_name = "long long"
        else:
            i64_name = "long"

        _register_dtype(numpy.int64, i64_name)
        _register_dtype(numpy.uint64, "unsigned %s" % i64_name)

        # http://projects.scipy.org/numpy/ticket/2017
        _register_dtype(numpy.uintp, "unsigned %s" % i64_name)
    else:
        _register_dtype(numpy.uintp, "unsigned")

    _register_dtype(numpy.float32, "float")
    _register_dtype(numpy.float64, "double")
    _register_dtype(numpy.complex64, "float2")
    _register_dtype(numpy.complex128, "double2")

_fill_dtype_registry()


def _get_struct_ctype_rec(dtype, alignment=None):
    """
    A recursive helper function for ``_get_struct_ctype``.
    Returns a tuple consisting of a string with the C type definition
    for a given ``dtype``, and the corresponding array suffix
    (if ``dtype`` is not an array, it is an empty string).
    If ``alignment`` is not ``None``, an explicit alignment (in bytes)
    will be specified in the resulting declaration.
    """

    if alignment is not None:
        alignment_str = "ALIGN(" + str(alignment) + ")"
    else:
        alignment_str = ""

    if len(dtype.shape) == 0:
        base_dtype = dtype
        dtype_shape = tuple()
    else:
        base_dtype = dtype.base
        dtype_shape = dtype.shape

    if base_dtype.names == None:
        type_str = ctype(base_dtype)
    else:
        lines = ["struct {"]
        for i, name in enumerate(base_dtype.names):
            elem_dtype, elem_offset = base_dtype.fields[name]
            if i == len(base_dtype.names) - 1:
                # If it is the last field of the struct, its alignment does not matter ---
                # the enompassing struct's one will override it anyway.
                alignment = None
            else:
                alignment = base_dtype.fields[base_dtype.names[i+1]][1] - elem_offset
                if alignment <= elem_dtype.itemsize:
                    alignment = None

            decl, suffix = _get_struct_ctype_rec(elem_dtype, alignment)

            # Add indentation to make nested structures easier to read
            decl = "\n".join("    " + line for line in decl.split("\n"))

            lines.append(decl + " " + name + suffix + ";")

        lines.append("}")
        type_str = "\n".join(lines)

    if len(dtype_shape) == 0:
        array_suffix = ""
    else:
        array_suffix = "".join("[" + str(d) + "]" for d in dtype_shape)

    return type_str + " " + alignment_str, array_suffix


def _get_struct_ctype(dtype):
    """
    Returns a string with the C type definition for a given ``dtype``.
    """

    expected_itemsize = sum(dt.itemsize for dt, _ in dtype.fields.values())
    if dtype.itemsize > expected_itemsize:
        alignment = dtype.itemsize
    else:
        alignment = None

    if len(dtype.shape) > 0:
        raise ValueError("The root structure cannot be an array")

    decl, _ = _get_struct_ctype_rec(dtype, alignment=alignment)

    return decl


def adjust_alignment(thr, dtype):
    """
    Returns a new data type object with the same fields as in ``dtype``
    and the field offsets set by the compiler in ``thr``.
    All existing non-standard offsets in ``dtype`` are ignored.
    """

    if dtype.names is None:
        return dtype

    adjusted_dtypes = [
        adjust_alignment(thr, dtype.fields[name][0])
        for name in dtype.names]

    new_dtype = numpy.dtype(dict(
        names=dtype.names,
        formats=adjusted_dtypes))

    struct = ctype_module(new_dtype)

    program = thr.compile(
    """
    #define my_offsetof(type, field) ((size_t)(&((type *)0)->field))

    KERNEL void test(GLOBAL_MEM int *dest)
    {
      %for i, name in enumerate(dtype.names):
      dest[${i}] = my_offsetof(${struct}, ${name});
      %endfor
      dest[${len(dtype.names)}] = sizeof(${struct});
    }
    """, render_kwds=dict(
        struct=struct,
        dtype=new_dtype))

    offsets = thr.array(len(dtype.names) + 1, numpy.int32)
    test = program.test
    test(offsets, global_size=1)
    offsets = offsets.get()

    # Casting to Python ints, becase numpy ints as dtype offsets make it unhashable.
    offsets = [int(offset) for offset in offsets]

    return numpy.dtype(dict(
        names=dtype.names,
        formats=adjusted_dtypes,
        offsets=offsets[:-1],
        itemsize=offsets[-1]))


def ctype(dtype):
    """
    For a built-in C type, returns a string with the name of the type.
    """
    return _DTYPE_TO_BUILTIN_CTYPE[normalize_type(dtype)]


def ctype_module(dtype):
    """
    For a struct type, returns a :py:class:`~reikna.cluda.Module` object
    with the ``typedef`` of a struct corresponding to the given ``dtype``
    (with its name set to the module prefix);
    falls back to :py:func:`~reikna.cluda.dtypes.ctype` otherwise.
    The structure definition includes the alignment required
    to produce field offsets specified in ``dtype``.

    Modules are cached and the function returns a single module instance for equal ``dtype``'s.
    Therefore inside a kernel it will be rendered with the same prefix everywhere it is used.
    This results in a behavior characteristic for a structural type system,
    same as for the basic dtype-ctype conversion.
    """
    if dtype.names is None:
        return ctype(dtype)
    else:
        # Root level import creates an import loop.
        from reikna.cluda.kernel import Module

        if dtype not in _DTYPE_TO_CTYPE_MODULE:
            struct = _get_struct_ctype(dtype)
            module = Module.create("typedef " + struct + " ${prefix};")
            _DTYPE_TO_CTYPE_MODULE[dtype] = module
        else:
            module = _DTYPE_TO_CTYPE_MODULE[dtype]

        return module

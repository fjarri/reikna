import numpy

import reikna.helpers as helpers
import reikna.cluda.dtypes as dtypes
from reikna.cluda import Module

TEMPLATE = helpers.template_for(__file__)


def create_struct_types(word_dtype, key_words, counter_words):

    key_dtype = dtypes.align(numpy.dtype([('v', (word_dtype, (key_words,)))]))
    key_ctype = dtypes.ctype_module(key_dtype)

    counter_dtype = dtypes.align(numpy.dtype([('v', (word_dtype, (counter_words,)))]))
    counter_ctype = dtypes.ctype_module(counter_dtype)

    return key_dtype, key_ctype, counter_dtype, counter_ctype


class Bijection:
    """
    Contains a CBRNG bijection module and accompanying metadata.
    Supports ``__process_modules__`` protocol.

    .. py:attribute:: word_dtype

        The data type of the integer word used by the generator.

    .. py:attribute:: key_words

        The number of words used by the key.

    .. py:attribute:: counter_words

        The number of words used by the counter.

    .. py:attribute:: key_dtype

        The ``numpy.dtype`` object representing a bijection key.
        Contains a single array field ``v`` with ``key_words`` of ``word_dtype`` elements.

    .. py:attribute:: counter_dtype

        The ``numpy.dtype`` object representing a bijection counter.
        Contains a single array field ``v`` with ``key_words`` of ``word_dtype`` elements.

    .. py:attribute:: raw_functions

        A dictionary ``dtype:function_name`` of available functions ``function_name``
        in :py:attr:`module` that produce a random full-range integer ``dtype``
        from a :c:type:`State`, advancing it.
        Available functions: :c:func:`get_raw_uint32`, :c:func:`get_raw_uint64`.

    .. py:attribute:: module

        The module containing the CBRNG function.
        It provides the C functions below.

    .. c:macro:: COUNTER_WORDS

        Contains the value of :py:attr:`counter_words`.

    .. c:macro:: KEY_WORDS

        Contains the value of :py:attr:`key_words`.

    .. c:type:: Word

        Contains the type corresponding to :py:attr:`word_dtype`.

    .. c:type:: Key

        Describes the bijection key.
        Alias for the structure generated from :py:attr:`key_dtype`.

        .. c:member:: Word v[KEY_WORDS]

    .. c:type:: Counter

        Describes the bijection counter, or its output.
        Alias for the structure generated from :py:attr:`counter_dtype`.

        .. c:member:: Word v[COUNTER_WORDS]

    .. c:function:: Counter make_counter_from_int(int x)

        Creates a counter object from an integer.

    .. c:function:: Counter bijection(Key key, Counter counter)

        The main bijection function.

    .. c:type:: State

        A structure containing the CBRNG state which is used by :py:mod:`~reikna.cbrng.samplers`.

    .. c:function:: State make_state(Key key, Counter counter)

        Creates a new state object.

    .. c:function:: Counter get_next_unused_counter(State state)

        Extracts a counter which has not been used in random sampling.

    .. c:type:: uint32

        A type of unsigned 32-bit word, corresponds to ``numpy.uint32``.

    .. c:type:: uint64

        A type of unsigned 64-bit word, corresponds to ``numpy.uint64``.

    .. c:function:: uint32 get_raw_uint32(State *state)

        Returns uniformly distributed unsigned 32-bit word and updates the state.

    .. c:function:: uint64 get_raw_uint64(State *state)

        Returns uniformly distributed unsigned 64-bit word and updates the state.
    """
    def __init__(self, module, word_dtype, key_dtype, counter_dtype):
        """__init__()""" # hide the signature from Sphinx

        self.module = module
        self.word_dtype = word_dtype

        self.key_words = key_dtype.fields['v'][0].shape[0]
        self.counter_words = counter_dtype.fields['v'][0].shape[0]

        self.counter_dtype = counter_dtype
        self.key_dtype = key_dtype

        # Compensate for the mysterious distinction numpy makes between
        # a predefined dtype and a generic dtype.
        self.raw_functions = {
            numpy.uint32: 'get_raw_uint32',
            numpy.dtype('uint32'): 'get_raw_uint32',
            numpy.uint64: 'get_raw_uint64',
            numpy.dtype('uint64'): 'get_raw_uint64',
        }

    def __process_modules__(self, process):
        return Bijection(
            process(self.module), self.word_dtype, self.key_dtype, self.counter_dtype)


def threefry(bitness, counter_words, rounds=20):
    """
    A CBRNG based on a big number of fast rounds (bit rotations).

    :param bitness: ``32`` or ``64``, corresponds to the size of generated random integers.
    :param counter_words: ``2`` or ``4``, number of integers generated in one go.
    :param rounds: ``1`` to ``72``, the more rounds, the better randomness is achieved.
        Default values are big enough to qualify as PRNG.
    :returns: a :py:class:`Bijection` object.
    """

    ROTATION_CONSTANTS = {
        # These are the R_256 constants from the Threefish reference sources
        # with names changed to R_64x4...
        (64, 4): numpy.array([[14, 52, 23, 5, 25, 46, 58, 32], [16, 57, 40, 37, 33, 12, 22, 32]]).T,

        # Output from skein_rot_search: (srs64_B64-X1000)
        # Random seed = 1. BlockSize = 128 bits. sampleCnt =  1024. rounds =  8, minHW_or=57
        # Start: Tue Mar  1 10:07:48 2011
        # rMin = 0.136. #0325[*15] [CRC=455A682F. hw_OR=64. cnt=16384. blkSize= 128].format
        (64, 2): numpy.array([[16, 42, 12, 31, 16, 32, 24, 21]]).T,
        # 4 rounds: minHW =  4  [  4  4  4  4 ]
        # 5 rounds: minHW =  8  [  8  8  8  8 ]
        # 6 rounds: minHW = 16  [ 16 16 16 16 ]
        # 7 rounds: minHW = 32  [ 32 32 32 32 ]
        # 8 rounds: minHW = 64  [ 64 64 64 64 ]
        # 9 rounds: minHW = 64  [ 64 64 64 64 ]
        # 10 rounds: minHW = 64  [ 64 64 64 64 ]
        # 11 rounds: minHW = 64  [ 64 64 64 64 ]

        # Output from skein_rot_search: (srs-B128-X5000.out)
        # Random seed = 1. BlockSize = 64 bits. sampleCnt =  1024. rounds =  8, minHW_or=28
        # Start: Mon Aug 24 22:41:36 2009
        # ...
        # rMin = 0.472. #0A4B[*33] [CRC=DD1ECE0F. hw_OR=31. cnt=16384. blkSize= 128].format
        (32, 4): numpy.array([[10, 11, 13, 23, 6, 17, 25, 18], [26, 21, 27, 5, 20, 11, 10, 20]]).T,
        # 4 rounds: minHW =  3  [  3  3  3  3 ]
        # 5 rounds: minHW =  7  [  7  7  7  7 ]
        # 6 rounds: minHW = 12  [ 13 12 13 12 ]
        # 7 rounds: minHW = 22  [ 22 23 22 23 ]
        # 8 rounds: minHW = 31  [ 31 31 31 31 ]
        # 9 rounds: minHW = 32  [ 32 32 32 32 ]
        # 10 rounds: minHW = 32  [ 32 32 32 32 ]
        # 11 rounds: minHW = 32  [ 32 32 32 32 ]

        # Output from skein_rot_search (srs32x2-X5000.out)
        # Random seed = 1. BlockSize = 64 bits. sampleCnt =  1024. rounds =  8, minHW_or=28
        # Start: Tue Jul 12 11:11:33 2011
        # rMin = 0.334. #0206[*07] [CRC=1D9765C0. hw_OR=32. cnt=16384. blkSize=  64].format
        (32, 2): numpy.array([[13, 15, 26, 6, 17, 29, 16, 24]]).T
        # 4 rounds: minHW =  4  [  4  4  4  4 ]
        # 5 rounds: minHW =  6  [  6  8  6  8 ]
        # 6 rounds: minHW =  9  [  9 12  9 12 ]
        # 7 rounds: minHW = 16  [ 16 24 16 24 ]
        # 8 rounds: minHW = 32  [ 32 32 32 32 ]
        # 9 rounds: minHW = 32  [ 32 32 32 32 ]
        # 10 rounds: minHW = 32  [ 32 32 32 32 ]
        # 11 rounds: minHW = 32  [ 32 32 32 32 ]
    }

    # Taken from Skein
    PARITY_CONSTANTS = {
        64: numpy.uint64(0x1BD11BDAA9FC1A22),
        32: numpy.uint32(0x1BD11BDA)
    }

    assert 1 <= rounds <= 72

    word_dtype = dtypes.normalize_type(numpy.uint32 if bitness == 32 else numpy.uint64)
    key_words = counter_words
    key_dtype, key_ctype, counter_dtype, counter_ctype = create_struct_types(
        word_dtype, key_words, counter_words)

    module = Module(
        TEMPLATE.get_def("threefry"),
        render_kwds=dict(
            word_dtype=word_dtype, word_ctype=dtypes.ctype(word_dtype),
            key_words=key_words, counter_words=counter_words,
            key_ctype=key_ctype, counter_ctype=counter_ctype,
            rounds=rounds, rotation_constants=ROTATION_CONSTANTS[(bitness, counter_words)],
            parity_constant=PARITY_CONSTANTS[bitness]))

    return Bijection(module, word_dtype, key_dtype, counter_dtype)


def philox(bitness, counter_words, rounds=10):
    """
    A CBRNG based on a low number of slow rounds (multiplications).

    :param bitness: ``32`` or ``64``, corresponds to the size of generated random integers.
    :param counter_words: ``2`` or ``4``, number of integers generated in one go.
    :param rounds: ``1`` to ``12``, the more rounds, the better randomness is achieved.
        Default values are big enough to qualify as PRNG.
    :returns: a :py:class:`Bijection` object.
    """

    W_CONSTANTS = {
        64: [
            numpy.uint64(0x9E3779B97F4A7C15), # golden ratio
            numpy.uint64(0xBB67AE8584CAA73B) # sqrt(3)-1
        ],
        32: [
            numpy.uint32(0x9E3779B9), # golden ratio
            numpy.uint32(0xBB67AE85) # sqrt(3)-1
        ]
    }

    M_CONSTANTS = {
        (64,2): [numpy.uint64(0xD2B74407B1CE6E93)],
        (64,4): [numpy.uint64(0xD2E7470EE14C6C93), numpy.uint64(0xCA5A826395121157)],
        (32,2): [numpy.uint32(0xD256D193)],
        (32,4): [numpy.uint32(0xD2511F53), numpy.uint32(0xCD9E8D57)]
    }

    assert 1 <= rounds <= 12
    word_dtype = dtypes.normalize_type(numpy.uint32 if bitness == 32 else numpy.uint64)
    key_words = counter_words // 2
    key_dtype, key_ctype, counter_dtype, counter_ctype = create_struct_types(
        word_dtype, key_words, counter_words)

    module = Module(
        TEMPLATE.get_def("philox"),
        render_kwds=dict(
            word_dtype=word_dtype, word_ctype=dtypes.ctype(word_dtype),
            key_words=key_words, counter_words=counter_words,
            key_ctype=key_ctype, counter_ctype=counter_ctype,
            rounds=rounds, w_constants=W_CONSTANTS[bitness],
            m_constants=M_CONSTANTS[(bitness, counter_words)]))

    return Bijection(module, word_dtype, key_dtype, counter_dtype)

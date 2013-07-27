import numpy

import reikna.helpers as helpers
import reikna.cluda.dtypes as dtypes
from reikna.cluda import Module

TEMPLATE = helpers.template_for(__file__)


class CBRNGModule:
    """
    The base class for the kernel API of CBRNG.
    Supports ``__process_modules__`` protocol.

    .. py:attribute:: dtype

        The data type of the integer word used by the generator.

    .. py:attribute:: counter_words

        The number of words used by the counter.

    .. py:attribute:: key_words

        The number of words used by the key.

    .. py:attribute:: module

        The module containing the CBRNG function.
        It provides:

        * the ``${prefix}COUNTER`` structure, which contains a single field
          ``v[${counter_words}]`` of type ``dtype``;
        * the ``${prefix}KEY`` structure, which contains a single field
          ``v[${key_words}]`` of type ``dtype``;
        * the function with the signature
          ``${prefix}COUNTER ${prefix}(${prefix}KEY, ${prefix}COUNTER)``.
    """
    pass


class Threefry(CBRNGModule):
    """
    Bases: :py:class:`~reikna.cbrng.rngs.CBRNGModule`

    A CBRNG based on a big number of fast rounds (bit rotations).

    :param bitness: ``32`` or ``64``, corresponds to the size of generated random integers.
    :param counter_words: ``2`` or ``4``, number of integers generated in one go.
    :param rounds: ``1`` to ``72``, the more rounds, the better randomness is achieved.
        Default values are big enough to qualify as PRNG.
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

    def __init__(self, bitness, counter_words, rounds=20, processed_module=None):

        assert 1 <= rounds <= 72

        self.dtype = dtypes.normalize_type(numpy.uint32 if bitness == 32 else numpy.uint64)
        ctype = dtypes.ctype(self.dtype)

        self.counter_words = counter_words
        self.key_words = counter_words
        self._rounds = rounds

        if processed_module is None:
            self.module = Module(
                TEMPLATE.get_def("threefry"),
                render_kwds=dict(
                    ctype=ctype,
                    counter_words=counter_words, key_words=self.key_words,
                    bitness=bitness, rounds=rounds,
                    rotation_constants=self.ROTATION_CONSTANTS[(bitness, counter_words)],
                    parity_constant=self.PARITY_CONSTANTS[bitness]))
        else:
            self.module = processed_module

    def __process_modules__(self, process):
        return Threefry(
            self.dtype.itemsize * 8, self.counter_words,
            rounds=self._rounds, processed_module=process(self.module))


class Philox(CBRNGModule):
    """
    Bases: :py:class:`~reikna.cbrng.rngs.CBRNGModule`

    A CBRNG based on a low number of slow rounds (multiplications).

    :param bitness: ``32`` or ``64``, corresponds to the size of generated random integers.
    :param counter_words: ``2`` or ``4``, number of integers generated in one go.
    :param rounds: ``1`` to ``12``, the more rounds, the better randomness is achieved.
        Default values are big enough to qualify as PRNG.
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

    def __init__(self, bitness, counter_words, rounds=10, processed_module=None):

        assert 1 <= rounds <= 12
        self.dtype = dtypes.normalize_type(numpy.uint32 if bitness == 32 else numpy.uint64)
        ctype = dtypes.ctype(self.dtype)

        self.counter_words = counter_words
        self.key_words = counter_words // 2
        self._rounds = rounds

        if processed_module is None:
            self.module = Module(
                TEMPLATE.get_def("philox"),
                render_kwds=dict(
                    dtype=self.dtype, ctype=ctype,
                    bitness=bitness, counter_words=counter_words,
                    key_words=self.key_words, rounds=rounds,
                    w_constants=self.W_CONSTANTS[bitness],
                    m_constants=self.M_CONSTANTS[(bitness, counter_words)]))
        else:
            self.module = processed_module

    def __process_modules__(self, process):
        return Philox(
            self.dtype.itemsize * 8, self.counter_words,
            rounds=self._rounds, processed_module=process(self.module))

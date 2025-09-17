"""
Reference implementation of the counter-based RNGs Threefry and Philox from
Salmon et al., P. Int. C. High. Perform. 16 (2011), doi:10.1145/2063384.2063405.
Based on the source code of Random123 library (http://www.thesalmons.org/john/random123/).

This implementation favors simplicity over speed and therefore
is not for use in production.
It only guarantees to produce the same results as the original Random123.
"""

import sys

import numpy

from reikna.helpers import IgnoreIntegerOverflow

# Rotation constants:
THREEFRY_ROTATION = {
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
    (32, 2): numpy.array([[13, 15, 26, 6, 17, 29, 16, 24]]).T,
    # 4 rounds: minHW =  4  [  4  4  4  4 ]
    # 5 rounds: minHW =  6  [  6  8  6  8 ]
    # 6 rounds: minHW =  9  [  9 12  9 12 ]
    # 7 rounds: minHW = 16  [ 16 24 16 24 ]
    # 8 rounds: minHW = 32  [ 32 32 32 32 ]
    # 9 rounds: minHW = 32  [ 32 32 32 32 ]
    # 10 rounds: minHW = 32  [ 32 32 32 32 ]
    # 11 rounds: minHW = 32  [ 32 32 32 32 ]
}


THREEFRY_KS_PARITY = {64: 0x1BD11BDAA9FC1A22, 32: 0x1BD11BDA}


def threefry_rotate(bits, shift, x):
    # Cast to uint is required by numpy coercion rules.
    # "% bits" is technically redundant since shift < bits always.
    s1 = numpy.uint32(shift % bits)
    s2 = numpy.uint32((bits - shift) % bits)
    return (x << s1) | (x >> s2)


def threefry(bits, words, ctr, key, rounds=None):
    """
    bits: word length (32, 64)
    words: number of generated items, 2 or 4
    rounds: number of rounds (up to 72)
    ctr: counter, array(words, numpy.uint${bits})
    key: key, array(words, numpy.uint${bits})
    returns: array(words, numpy.uint${bits})

    Note: for bits=64, words=4 and rounds=72 it is the same as Threefish algorithm from Skein,
    only without the 128-bit "tweak" which is applied to the key.
    With this tweak it is possible to upgrade this algorithm from PRNG to QRNG.
    """
    assert bits in (32, 64)
    assert words in (2, 4)

    if rounds is None:
        rounds = 20

    dtype = numpy.uint32 if bits == 32 else numpy.uint64

    if words == 2:
        assert rounds <= 32
        ks = numpy.empty(3, dtype)
    else:
        assert rounds <= 72
        ks = numpy.empty(5, dtype)

    result = numpy.empty(words, dtype)
    assert ctr.size == key.size == words

    ks[words] = THREEFRY_KS_PARITY[bits]
    for i in range(words):
        ks[i] = key[i]
        result[i] = ctr[i]
        ks[words] ^= key[i]

    # Insert initial key before round 0
    for i in range(words):
        result[i] += ks[i]

    rotation = THREEFRY_ROTATION[(bits, words)]

    with IgnoreIntegerOverflow():
        for rnd in range(rounds):
            # TODO: In the current version of Random123 (1.06),
            # there is a bug in r_idx calculation for words == 2, where
            #    r_idx = rnd % 8 if rnd < 20 else (rnd - 4) % 8
            # instead of what goes below.
            # When this bug is fixed, Random123 and this implementation
            # will start to produce identical results again,
            # and this comment can be removed.
            r_idx = rnd % 8

            if words == 2:
                result[0] += result[1]
                result[1] = threefry_rotate(bits, rotation[r_idx, 0], result[1])
                result[1] ^= result[0]
            else:
                idx1 = 1 if rnd % 2 == 0 else 3
                idx2 = 3 if rnd % 2 == 0 else 1

                result[0] += result[idx1]
                result[idx1] = threefry_rotate(bits, rotation[r_idx, 0], result[idx1])
                result[idx1] ^= result[0]

                result[2] += result[idx2]
                result[idx2] = threefry_rotate(bits, rotation[r_idx, 1], result[idx2])
                result[idx2] ^= result[2]

            if rnd % 4 == 3:
                for i in range(words):
                    result[i] += ks[(rnd // 4 + i + 1) % (words + 1)]

                result[words - 1] += dtype(rnd // 4 + 1)

    return result


PHILOX_W = {
    64: [
        numpy.uint64(0x9E3779B97F4A7C15),  # golden ratio
        numpy.uint64(0xBB67AE8584CAA73B),  # sqrt(3)-1
    ],
    32: [
        numpy.uint32(0x9E3779B9),  # golden ratio
        numpy.uint32(0xBB67AE85),  # sqrt(3)-1
    ],
}

PHILOX_M = {
    (64, 2): [numpy.uint64(0xD2B74407B1CE6E93)],
    (64, 4): [numpy.uint64(0xD2E7470EE14C6C93), numpy.uint64(0xCA5A826395121157)],
    (32, 2): [numpy.uint32(0xD256D193)],
    (32, 4): [numpy.uint32(0xD2511F53), numpy.uint32(0xCD9E8D57)],
}


def philox_mulhilo(bits, x, y):
    res = int(x) * int(y)
    return numpy.asarray(res // (2**bits), x.dtype), numpy.asarray(res % (2**bits), x.dtype)


def philox_round(bits, words, rnd, ctr, key):
    ctr = ctr.copy()
    rnd = numpy.asarray(rnd, ctr.dtype)

    if words == 2:
        key0 = key[0] + PHILOX_W[bits][0] * rnd
        hi, lo = philox_mulhilo(bits, PHILOX_M[(bits, words)][0], ctr[0])
        ctr[0] = hi ^ key0 ^ ctr[1]
        ctr[1] = lo
    else:
        key0 = key[0] + PHILOX_W[bits][0] * rnd
        key1 = key[1] + PHILOX_W[bits][1] * rnd
        hi0, lo0 = philox_mulhilo(bits, PHILOX_M[(bits, words)][0], ctr[0])
        hi1, lo1 = philox_mulhilo(bits, PHILOX_M[(bits, words)][1], ctr[2])
        ctr[0] = hi1 ^ ctr[1] ^ key0
        ctr[1] = lo1
        ctr[2] = hi0 ^ ctr[3] ^ key1
        ctr[3] = lo0

    return ctr


def philox(bits, words, ctr, key, rounds=None):
    """
    bits: word length (32, 64)
    words: number of generated items, 2 or 4
    rounds: number of rounds (up to 16)
    inn: counter, array(words, numpy.uint${bits}) --- counter
    k: key, array(words/2, numpy.uint${bits}) --- key
    returns: array(words, numpy.uint${bits})
    """
    assert bits in (32, 64)
    assert words in (2, 4)

    if rounds is None:
        rounds = 10

    assert rounds <= 16

    with IgnoreIntegerOverflow():
        for rnd in range(rounds):
            ctr = philox_round(bits, words, rnd, ctr, key)

    return ctr


# Some simple testcases to compare with Random123 results.


def myhex(x):
    s = hex(x)[2:-1]
    return "0" * (2 * x.nbytes - len(s)) + s


def test(name):
    kk = numpy.array([123, 456, 789, 101112])

    for bits in (32, 64):
        for words in (2, 4):
            dtype = numpy.uint32 if bits == 32 else numpy.uint64
            if name == "threefry":
                key = kk[:words].astype(dtype)
                rounds = 72 if words == 4 else 32
                func = threefry
            elif name == "philox":
                key = kk[: words // 2].astype(dtype)
                rounds = 10
                func = philox

            ctr = numpy.zeros(words).astype(dtype)

            print(f"{name}: bits = {bits}, words = {words}")  # noqa: T201
            for i in range(5):
                ctr[0] = i
                res = func(bits, words, ctr, key, rounds=rounds)
                print(f"ctr {i}: " + " ".join([myhex(i) for i in res]))  # noqa: T201


if __name__ == "__main__":
    test("threefry")
    test("philox")

import numpy

from reikna.cluda import Module
import reikna.helpers as helpers


class KeyGenerator:

    def __init__(self, module, base_key):
        self.module = module
        self._base_key = base_key

    def __process_modules__(self, process):
        return KeyGenerator(process(self.module), self._base_key)

    @classmethod
    def create(cls, bijection, seed=None, reserve_id_space=True):

        if reserve_id_space:
            if bijection.key_words == 1 and bijection.dtype.itemsize == 4:
            # It's too hard to compress both global and thread-dependent part
            # in a single 32-bit word.
            # Let the user handle this himself.
                raise ValueError("Cannor reserve ID space in a 32-bit key")

            if bijection.dtype.itemsize == 4:
                key_words32 = bijection.key_words - 1
            else:
                if bijection.key_words > 1:
                    key_words32 = (bijection.key_words - 1) * 2
                else:
                    # Philox-2x64 case, the key is a single 64-bit integer.
                    # We use first 32 bit for the key, and the remaining 32 bit for a thread identifier.
                    key_words32 = 1
        else:
            key_words32 = bijection.key_words * (bijection.dtype.itemsize // 4)

        if isinstance(seed, numpy.ndarray):
            # explicit key was provided
            assert seed.size == key_words32 and seed.dtype == numpy.uint32
            key = seed.copy().flatten()
        else:
            # use numpy to generate the key from seed
            np_rng = numpy.random.RandomState(seed)

            # 32-bit Python can only generate random integer up to 2**31-1
            key16 = np_rng.randint(0, 2**16, key_words32 * 2)
            key = numpy.zeros(key_words32, numpy.uint32)
            for i in range(key_words32 * 2):
                key[i // 2] += key16[i] << (16 if i % 2 == 0 else 0)

        full_key = numpy.zeros(bijection.key_words, bijection.dtype)
        if bijection.dtype.itemsize == 4:
            full_key[:key_words32] = key
        else:
            for i in range(key_words32):
                full_key[i // 2] += key[i] << (32 if i % 2 == 0 else 0)

        module = Module.create("""
            WITHIN_KERNEL ${bijection.module}KEY ${prefix}(int id)
            {
                ${bijection.module}KEY result;

                %for i in range(bijection.key_words):
                result.v[${i}] = ${key[i]}
                    %if i == bijection.key_words - 1:
                    + id
                    %endif
                ;
                %endfor

                return result;
            }
            """,
            render_kwds=dict(
                bijection=bijection,
                key=full_key))

        return cls(module, full_key)

    def reference(self, idx):
        key = self._base_key.copy()
        key[-1] += idx
        return key

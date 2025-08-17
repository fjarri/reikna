from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy
from grunnur import Module, dtypes

from reikna import helpers

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from .bijections import Bijection


class KeyGenerator:
    """
    Contains a key generator module and accompanying metadata.

    .. py:attribute:: module

        A module with the key generator function:

    .. c:function:: Key key_from_int(int idx)

        Generates and returns a key, suitable for the bijection which was given to the constructor.
    """

    def __init__(self, module: Module, base_key: NDArray[Any]):
        """__init__()"""  # hide the signature from Sphinx
        self.module = module
        self._base_key = base_key

    @classmethod
    def create(  # noqa: PLR0912
        cls,
        bijection: Bijection,
        seed: int | NDArray[numpy.uint32] | None = None,
        *,
        reserve_id_space: bool = True,
    ) -> KeyGenerator:
        """
        Creates a generator.

        :param bijection: a :py:class:`~reikna.cbrng.bijections.Bijection` object.
        :param seed: an integer, or numpy array of 32-bit unsigned integers.
        :param reserve_id_space: if ``True``, the last 32 bit of the key will be reserved
            for the thread identifier.
            As a result, the total size of the key should be 64 bit or more.
            If ``False``, the thread identifier will be just added to the key,
            which will still result in different keys for different threads,
            with the danger that different seeds produce the same sequences.
        """
        if reserve_id_space:
            if bijection.key_words == 1 and bijection.word_dtype.itemsize == 4:  # noqa: PLR2004
                # It's too hard to compress both global and thread-dependent part
                # in a single 32-bit word.
                # Let the user handle this himself.
                raise ValueError("Cannor reserve ID space in a 32-bit key")

            if bijection.word_dtype.itemsize == 4:  # noqa: PLR2004
                key_words32 = bijection.key_words - 1
            elif bijection.key_words > 1:
                key_words32 = (bijection.key_words - 1) * 2
            else:
                # Philox-2x64 case, the key is a single 64-bit integer.
                # We use first 32 bit for the key, and the remaining 32 bit for a thread identifier.
                key_words32 = 1
        else:
            key_words32 = bijection.key_words * (bijection.word_dtype.itemsize // 4)

        if isinstance(seed, numpy.ndarray):
            # explicit key was provided
            if seed.size != key_words32 or seed.dtype != numpy.uint32:
                raise ValueError(f"Invalid seed: {seed}")
            key = seed.copy().flatten()
        else:
            # use numpy to generate the key from seed
            np_rng = numpy.random.RandomState(seed)

            # 32-bit Python can only generate random integer up to 2**31-1
            key16 = np_rng.randint(0, 2**16, key_words32 * 2)
            key = numpy.zeros(key_words32, numpy.uint32)
            for i in range(key_words32 * 2):
                key[i // 2] += key16[i] << (16 if i % 2 == 0 else 0)

        full_key = numpy.zeros(1, bijection.key_dtype)[0]
        if bijection.word_dtype.itemsize == 4:  # noqa: PLR2004
            full_key["v"][:key_words32] = key
        else:
            for i in range(key_words32):
                full_key["v"][i // 2] += key[i] << (32 if i % 2 == 0 else 0)

        module = Module.from_string(
            """
            FUNCTION ${bijection.module}Key ${prefix}key_from_int(int idx)
            {
                ${bijection.module}Key result;

                %for i in range(bijection.key_words):
                result.v[${i}] = ${key['v'][i]}
                    %if i == bijection.key_words - 1:
                    + idx
                    %endif
                ;
                %endfor

                return result;
            }
            """,
            render_globals=dict(bijection=bijection, key=full_key),
        )

        return cls(module, full_key)

    def reference(self, idx: int) -> NDArray[Any]:
        """
        Reference function that returns the key given the thread identifier.
        Uses the same algorithm as the module.
        """
        key = self._base_key.copy()
        # TODO: can we make this typeable?
        key["v"][-1] += numpy.asarray(idx, key.dtype.fields["v"][0].base)  # type: ignore[index]
        return key

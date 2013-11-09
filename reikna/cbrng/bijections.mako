<%def name="common_declarations(prefix, word_ctype, key_words, counter_words, key_ctype, counter_ctype)">
#define ${prefix}KEY_WORDS ${key_words}
#define ${prefix}COUNTER_WORDS ${counter_words}
#define ${prefix}Word ${word_ctype}
#define ${prefix}Key ${key_ctype}
#define ${prefix}Counter ${counter_ctype}

WITHIN_KERNEL ${counter_ctype} ${prefix}make_counter_from_int(int x)
{
    ${counter_ctype} result;
    %for i in range(counter_words - 1):
    result.v[${i}] = 0;
    %endfor
    result.v[${counter_words - 1}] = x;
    return result;
}
</%def>


<%def name="raw_samplers(prefix, word_dtype, word_ctype, counter_words, key_ctype, counter_ctype)">
<%
    counter_uints32 = counter_words * (word_dtype.itemsize // 4)
    uint32 = dtypes.ctype(numpy.uint32)
    uint64 = dtypes.ctype(numpy.uint64)
%>

typedef ${uint32} ${prefix}uint32;
typedef ${uint64} ${prefix}uint64;

typedef struct ${prefix}
{
    ${key_ctype} key;
    ${counter_ctype} counter;
    union {
        ${counter_ctype} buffer;
        ${uint32} buffer_uint32[${counter_uints32}];
    };
    int buffer_uint32_cursor;
} ${prefix}State;


WITHIN_KERNEL void ${prefix}bump_counter(${prefix}State *state)
{
    %for i in range(counter_words - 1, 0, -1):
    state->counter.v[${i}] += 1;
    if (state->counter.v[${i}] == 0)
    {
    %endfor
    state->counter.v[0] += 1;
    %for i in range(counter_words - 1, 0, -1):
    }
    %endfor
}

WITHIN_KERNEL ${counter_ctype} ${prefix}get_next_unused_counter(${prefix}State state)
{
    if (state.buffer_uint32_cursor > 0)
    {
        ${prefix}bump_counter(&state);
    }
    return state.counter;
}

WITHIN_KERNEL void ${prefix}refill_buffer(${prefix}State *state)
{
    state->buffer = ${prefix}bijection(state->key, state->counter);
}

WITHIN_KERNEL ${prefix}State ${prefix}make_state(${key_ctype} key, ${counter_ctype} counter)
{
    ${prefix}State state;
    state.key = key;
    state.counter = counter;
    state.buffer_uint32_cursor = 0;
    ${prefix}refill_buffer(&state);
    return state;
}

WITHIN_KERNEL ${uint32} ${prefix}get_raw_uint32(${prefix}State *state)
{
    if (state->buffer_uint32_cursor == ${counter_uints32})
    {
        ${prefix}bump_counter(state);
        state->buffer_uint32_cursor = 0;
        ${prefix}refill_buffer(state);
    }

    int cur = state->buffer_uint32_cursor;
    state->buffer_uint32_cursor += 1;
    return state->buffer_uint32[cur];
}

WITHIN_KERNEL ${uint64} ${prefix}get_raw_uint64(${prefix}State *state)
{
    if (state->buffer_uint32_cursor >= ${counter_uints32} - 1)
    {
        ${prefix}bump_counter(state);
        state->buffer_uint32_cursor = 0;
        ${prefix}refill_buffer(state);
    }

    int cur = state->buffer_uint32_cursor;
    state->buffer_uint32_cursor += 2;
    %if word_dtype.itemsize == 8:
    return state->buffer.v[cur / 2];
    %else:
    ${uint32} hi = state->buffer_uint32[cur];
    ${uint32} lo = state->buffer_uint32[cur+1];
    return ((${uint64})hi << 32) + (${uint64})lo;
    %endif
}
</%def>


<%def name="threefry(prefix)">
${common_declarations(prefix, word_ctype, key_words, counter_words, key_ctype, counter_ctype)}

WITHIN_KERNEL INLINE ${word_ctype} ${prefix}threefry_rotate(${word_ctype} x, ${word_ctype} lshift)
{
#ifdef CUDA
    return (x << lshift) | (x >> (${word_dtype.itemsize * 8} - lshift));
#else
    return rotate(x, lshift);
#endif
}

WITHIN_KERNEL ${counter_ctype} ${prefix}bijection(
    const ${key_ctype} key, const ${counter_ctype} counter)
{
    // Prepare the key
    %for i in range(counter_words):
    const ${word_ctype} key${i} = key.v[${i}];
    %endfor

    const ${word_ctype} key${counter_words} = ${parity_constant}
    %for i in range(counter_words):
        ^ key${i}
    %endfor
        ;

    ${counter_ctype} X = counter;

    // Insert initial key before round 0
    %for i in range(counter_words):
    X.v[${i}] += key${i};
    %endfor

    %for rnd in range(rounds):
    // round ${rnd}
    <%
        R_idx = rnd % 8
    %>
    %if counter_words == 2:
        X.v[0] += X.v[1];
        X.v[1] = ${prefix}threefry_rotate(X.v[1], ${rotation_constants[R_idx, 0]});
        X.v[1] ^= X.v[0];
    %else:
    <%
        idx1 = 1 if rnd % 2 == 0 else 3
        idx2 = 3 if rnd % 2 == 0 else 1
    %>
        X.v[0] += X.v[${idx1}];
        X.v[${idx1}] = ${prefix}threefry_rotate(X.v[${idx1}], ${rotation_constants[R_idx, 0]});
        X.v[${idx1}] ^= X.v[0];

        X.v[2] += X.v[${idx2}];
        X.v[${idx2}] = ${prefix}threefry_rotate(X.v[${idx2}], ${rotation_constants[R_idx, 1]});
        X.v[${idx2}] ^= X.v[2];
    %endif

    %if rnd % 4 == 3:
    %for i in range(counter_words):
        X.v[${i}] += key${(rnd // 4 + i + 1) % (counter_words + 1)};
    %endfor
        X.v[${counter_words - 1}] += ${rnd // 4 + 1};
    %endif

    %endfor

    return X;
}

${raw_samplers(prefix, word_dtype, word_ctype, counter_words, key_ctype, counter_ctype)}
</%def>


<%def name="philox(prefix)">
${common_declarations(prefix, word_ctype, key_words, counter_words, key_ctype, counter_ctype)}

WITHIN_KERNEL INLINE ${word_ctype} ${prefix}mulhilo(
    ${word_ctype} *hip, ${word_ctype} a, ${word_ctype} b)
{
%if word_dtype.itemsize == 4:
<%
    d_ctype = dtypes.ctype(numpy.uint64)
%>
    ${d_ctype} product = ((${d_ctype})a)*((${d_ctype})b);
    *hip = product >> ${word_dtype.itemsize * 8};
    return (${word_ctype})product;
%else:
#ifdef CUDA
    *hip = __umul64hi(a, b);
#else
    *hip = mul_hi(a, b);
#endif
    return a * b;
%endif
}

WITHIN_KERNEL ${counter_ctype} ${prefix}bijection(
    const ${key_ctype} key, const ${counter_ctype} counter)
{
    ${counter_ctype} X = counter;

    %for i in range(counter_words // 2):
    ${word_ctype} key${i} = key.v[${i}];
    %endfor

    %if counter_words == 2:
    ${word_ctype} hi, lo;
    %else:
    ${word_ctype} hi0, lo0, hi1, lo1;
    %endif

    %for rnd in range(rounds):
    // round ${rnd}

    %if counter_words == 2:
        lo = ${prefix}mulhilo(&hi, ${m_constants[0]}, X.v[0]);
        X.v[0] = hi ^ key0 ^ X.v[1];
        X.v[1] = lo;
    %else:
        lo0 = ${prefix}mulhilo(&hi0, ${m_constants[0]}, X.v[0]);
        lo1 = ${prefix}mulhilo(&hi1, ${m_constants[1]}, X.v[2]);
        X.v[0] = hi1 ^ X.v[1] ^ key0;
        X.v[1] = lo1;
        X.v[2] = hi0 ^ X.v[3] ^ key1;
        X.v[3] = lo0;
    %endif

    %if rnd < rounds - 1:
    // bump key
    key0 += ${w_constants[0]};
    %if counter_words == 4:
    key1 += ${w_constants[1]};
    %endif
    %endif

    %endfor

    return X;
}

${raw_samplers(prefix, word_dtype, word_ctype, counter_words, key_ctype, counter_ctype)}
</%def>

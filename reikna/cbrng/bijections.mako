<%def name="structure_declarations(prefix, dtype, counter_words, key_words)">
<%
    ctype = dtypes.ctype(dtype)
%>
typedef struct ${prefix}_COUNTER
{
    ${ctype} v[${counter_words}];
} ${prefix}COUNTER;

typedef struct ${prefix}_KEY
{
    ${ctype} v[${key_words}];
} ${prefix}KEY;

WITHIN_KERNEL ${prefix}COUNTER ${prefix}make_counter_from_int(int x)
{
    ${prefix}COUNTER result;
    %for i in range(counter_words - 1):
    result.v[${i}] = 0;
    %endfor
    result.v[${counter_words - 1}] = x;
    return result;
}
</%def>


<%def name="raw_samplers(prefix, dtype, counter_words)">
<%
    ctype = dtypes.ctype(dtype)
    counter_uints32 = counter_words * (dtype.itemsize // 4)
    uint32 = dtypes.ctype(numpy.uint32)
    uint64 = dtypes.ctype(numpy.uint64)
%>

typedef struct ${prefix}_STATE
{
    ${prefix}KEY key;
    ${prefix}COUNTER counter;
    union {
        ${prefix}COUNTER buffer;
        ${uint32} buffer_uint32[${counter_uints32}];
    };
    int buffer_uint32_cursor;
} ${prefix}STATE;


WITHIN_KERNEL void ${prefix}bump_counter(${prefix}STATE *state)
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

WITHIN_KERNEL ${prefix}COUNTER ${prefix}get_next_unused_counter(${prefix}STATE state)
{
    if (state.buffer_uint32_cursor > 0)
    {
        ${prefix}bump_counter(&state);
    }
    return state.counter;
}

WITHIN_KERNEL void ${prefix}refill_buffer(${prefix}STATE *state)
{
    state->buffer = ${prefix}(state->key, state->counter);
}

WITHIN_KERNEL ${prefix}STATE ${prefix}make_state(${prefix}KEY key, ${prefix}COUNTER counter)
{
    ${prefix}STATE state;
    state.key = key;
    state.counter = counter;
    state.buffer_uint32_cursor = 0;
    ${prefix}refill_buffer(&state);
    return state;
}

WITHIN_KERNEL ${uint32} ${prefix}get_raw_uint32(${prefix}STATE *state)
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

WITHIN_KERNEL ${uint64} ${prefix}get_raw_uint64(${prefix}STATE *state)
{
    if (state->buffer_uint32_cursor >= ${counter_uints32} - 1)
    {
        ${prefix}bump_counter(state);
        state->buffer_uint32_cursor = 0;
        ${prefix}refill_buffer(state);
    }

    int cur = state->buffer_uint32_cursor;
    state->buffer_uint32_cursor += 2;
    %if dtype.itemsize == 8:
    return state->buffer.v[cur / 2];
    %else:
    ${uint32} hi = state->buffer_uint32[cur];
    ${uint32} lo = state->buffer_uint32[cur+1];
    return ((${uint64})hi << 32) + (${uint64})lo;
    %endif
}
</%def>


<%def name="threefry(prefix)">
${structure_declarations(prefix, dtype, counter_words, key_words)}

WITHIN_KERNEL INLINE ${ctype} ${prefix}threefry_rotate(${ctype} x, ${ctype} lshift)
{
#ifdef CUDA
    return (x << lshift) | (x >> (${dtype.itemsize * 8} - lshift));
#else
    return rotate(x, lshift);
#endif
}

WITHIN_KERNEL ${prefix}COUNTER ${prefix}(const ${prefix}KEY key, const ${prefix}COUNTER counter)
{
    // Prepare the key
    %for i in range(counter_words):
    const ${ctype} key${i} = key.v[${i}];
    %endfor

    const ${ctype} key${counter_words} = ${parity_constant}
    %for i in range(counter_words):
        ^ key${i}
    %endfor
        ;

    ${prefix}COUNTER X;
    %for i in range(counter_words):
    X.v[${i}] = counter.v[${i}];
    %endfor

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

${raw_samplers(prefix, dtype, counter_words)}
</%def>


<%def name="philox(prefix)">
${structure_declarations(prefix, dtype, counter_words, key_words)}

WITHIN_KERNEL INLINE ${ctype} ${prefix}mulhilo(${ctype} *hip, ${ctype} a, ${ctype} b)
{
%if dtype.itemsize == 4:
<%
    d_ctype = dtypes.ctype(numpy.uint64)
%>
    ${d_ctype} product = ((${d_ctype})a)*((${d_ctype})b);
    *hip = product >> ${dtype.itemsize * 8};
    return (${ctype})product;
%else:
#ifdef CUDA
    *hip = __umul64hi(a, b);
#else
    *hip = mul_hi(a, b);
#endif
    return a*b;
%endif
}

WITHIN_KERNEL ${prefix}COUNTER ${prefix}(const ${prefix}KEY key, const ${prefix}COUNTER counter)
{
    ${prefix}COUNTER X;
    %for i in range(counter_words):
    X.v[${i}] = counter.v[${i}];
    %endfor

    %for i in range(counter_words // 2):
    ${ctype} key${i} = key.v[${i}];
    %endfor

    %if counter_words == 2:
    ${ctype} hi, lo;
    %else:
    ${ctype} hi0, lo0, hi1, lo1;
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

${raw_samplers(prefix, dtype, counter_words)}
</%def>

<%def name="structure_declarations(prefix, ctype, counter_words, key_words)">
typedef struct ${prefix}_COUNTER
{
    ${ctype} v[${counter_words}];
} ${prefix}COUNTER;

typedef struct ${prefix}_KEY
{
    ${ctype} v[${key_words}];
} ${prefix}KEY;
</%def>


<%def name="threefry(prefix)">
${structure_declarations(prefix, ctype, counter_words, key_words)}

WITHIN_KERNEL INLINE ${ctype} ${prefix}threefry_rotate(${ctype} x, ${ctype} lshift)
{
#ifdef CUDA
    return (x << lshift) | (x >> (${bitness} - lshift));
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
</%def>


<%def name="philox(prefix)">
${structure_declarations(prefix, ctype, counter_words, key_words)}

WITHIN_KERNEL INLINE ${ctype} ${prefix}mulhilo(${ctype} *hip, ${ctype} a, ${ctype} b)
{
%if bitness == 32:
<%
    d_ctype = dtypes.ctype(numpy.uint64)
%>
    ${d_ctype} product = ((${d_ctype})a)*((${d_ctype})b);
    *hip = product >> ${bitness};
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
</%def>

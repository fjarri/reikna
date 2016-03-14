<%def name="result_struct(prefix, ctype, randoms_per_call)">
#define ${prefix}Value ${ctype};
#define ${prefix}RANDOMS_PER_CALL ${randoms_per_call}

typedef struct
{
    ${ctype} v[${randoms_per_call}];
} ${prefix}Result;
</%def>


<%def name="uniform_integer(prefix)">
${result_struct(prefix, ctype, 1)}

WITHIN_KERNEL ${prefix}Result ${prefix}sample(${bijection.module}State *state)
{
    ${prefix}Result result;
    ${raw_ctype} non_offset = 0;

    %if max_num % num == 0:
    ${raw_ctype} t = ${bijection.module}${raw_func}(state);
    non_offset = t / ${max_num // num};
    %else:
    while(1)
    {
        ${raw_ctype} t = ${bijection.module}${raw_func}(state);
        if (t < ${max_num - max_num % num})
        {
            non_offset = t / ${max_num // num};
            break;
        }
    }
    %endif

    result.v[0] = (${ctype})non_offset + (${ctype})${low};
    return result;
}
</%def>


<%def name="uniform_float(prefix)">
${result_struct(prefix, ctype, 1)}

WITHIN_KERNEL ${prefix}Result ${prefix}sample(${bijection.module}State *state)
{
    ${prefix}Result result;
    ${ctype} normalized = (${ctype})${bijection.module}${raw_func}(state) / ${raw_max};
    result.v[0] = normalized * (${size}) + (${low});
    return result;
}
</%def>


<%def name="normal_bm(prefix)">
<%
    component_std = std / numpy.sqrt(2) if complex_res else std
    real = lambda x: dtypes.c_constant(x, r_dtype)
%>

%if complex_res:
${result_struct(prefix, c_ctype, 1)}
%else:
${result_struct(prefix, r_ctype, 2)}
%endif

WITHIN_KERNEL ${prefix}Result ${prefix}sample(${bijection.module}State *state)
{
    ${prefix}Result result;
    ${uf.module}Result r1 = ${uf.module}sample(state);
    ${uf.module}Result r2 = ${uf.module}sample(state);
    ${r_ctype} u1 = r1.v[0];
    ${r_ctype} u2 = r2.v[0];

    ${r_ctype} ang = ${real(2.0 * numpy.pi)} * u2;
    ${c_ctype} cos_sin = ${polar_unit}(ang);
    ${r_ctype} coeff = sqrt(${real(-2.0)} * log(u1)) * (${real(component_std)});
    ${c_ctype} c_res = COMPLEX_CTR(${c_ctype})(coeff * cos_sin.x, coeff * cos_sin.y);

    %if complex_res:
    ${c_ctype} mean = COMPLEX_CTR(${c_ctype})(${real(mean.real)}, ${real(mean.imag)});
    result.v[0] = c_res + mean;
    %else:
    result.v[0] = c_res.x + (${real(mean)});
    result.v[1] = c_res.y + (${real(mean)});
    %endif

    return result;
}
</%def>


<%def name="gamma(prefix)">
${result_struct(prefix, ctype, 1)}

WITHIN_KERNEL ${prefix}Result ${prefix}sample(${bijection.module}State *state)
{
    <%
        d = shape - 1. / 3
        c = 1 / numpy.sqrt(9 * d)
    %>

    ${prefix}Result result;
    ${uf.module}Result rand_float;
    ${nbm.module}Result rand_normal;
    bool normals_need_regen = true;

    const ${ctype} d = ${dtypes.c_constant(d, dtype)};
    const ${ctype} c = ${dtypes.c_constant(c, dtype)};

    for (;;)
    {
        ${ctype} X, V;

        do
        {
            if (normals_need_regen)
            {
                rand_normal = ${nbm.module}sample(state);
                X = rand_normal.v[0];
                normals_need_regen = false;
            }
            else
            {
                X = rand_normal.v[1];
                normals_need_regen = true;
            }

            V = 1 + c * X;
        } while (V <= 0);

        V = V * V * V;
        rand_float = ${uf.module}sample(state);
        if (rand_float.v[0] < 1 - 0.0331f * (X * X) * (X * X))
        {
            result.v[0] = (d * V) * (${scale});
            break;
        }
        if (log(rand_float.v[0]) < 0.5f * X * X + d * (1 - V + log(V)))
        {
            result.v[0] = (d * V) * (${scale});
            break;
        }
    }

    return result;
}
</%def>


<%def name="vonmises(prefix)">
## Uses the rejection algorithm by Best and Fisher.
## See Chapter 9 of "Non-Uniform Random Variate Generation" by Luc Devroye.
## http://www.nrbook.com/devroye/

${result_struct(prefix, ctype, 1)}

WITHIN_KERNEL ${prefix}Result ${prefix}sample(${bijection.module}State *state)
{
    <%
        tau = 1 + (1 + 4 * kappa**2)**0.5
        rho = (tau - (2 * tau)**0.5) / (2 * kappa)
        r = (1 + rho**2) / (2 * rho)

        pi = dtypes.c_constant(numpy.pi, dtype)
    %>

    ${prefix}Result result;
    ${uf.module}Result rand_float;

    const ${ctype} r = ${dtypes.c_constant(r, dtype)};

    ${ctype} f;

    for (;;)
    {
        rand_float = ${uf.module}sample(state);
        const ${ctype} u1 = rand_float.v[0];

        const ${ctype} z = cos(${pi} * u1);

        f = (1 + r * z) / (r + z);
        const ${ctype} c = ${dtypes.c_constant(kappa, dtype)} * (r - f);

        rand_float = ${uf.module}sample(state);
        const ${ctype} u2 = rand_float.v[0];

        if (u2 < c * (2 - c) || c <= log(c / u2) + 1)
            break;
    }

    rand_float = ${uf.module}sample(state);
    const ${ctype} u3 = rand_float.v[0];

    ${ctype} x;
    if (u3 < ${dtypes.c_constant(0.5, dtype)})
        x = ${mu} - acos(f);
    else
        x = ${mu} + acos(f);

    result.v[0] = fmod(x + ${pi}, 2 * ${pi}) - ${pi};
    return result;
}
</%def>

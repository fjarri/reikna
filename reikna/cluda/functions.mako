<%def name="cast(prefix)">
WITHIN_KERNEL ${dtypes.ctype(out_dtype)} ${prefix}(${dtypes.ctype(in_dtype)} x)
{
<%
    if dtypes.is_complex(out_dtype) and not dtypes.is_complex(in_dtype):
        result = dtypes.complex_ctr(out_dtype) + "(x, 0)"
    elif dtypes.is_complex(out_dtype) == dtypes.is_complex(in_dtype):
        result = "(" + dtypes.ctype(out_dtype) + ")x"
    else:
        raise NotImplementedError("Cast from " + str(in_dtype) + " to " + str(out_dtype) +
            " is not supported")
%>
    return ${result};
}
</%def>


<%def name="mul(prefix)">
<%
    argnames = ["a" + str(i + 1) for i in xrange(len(in_dtypes))]
%>
WITHIN_KERNEL ${dtypes.ctype(out_dtype)} ${prefix}(
    ${", ".join(dtypes.ctype(dt) + " " + name for name, dt in zip(argnames, in_dtypes))})
{
<%
    last_result = argnames[-1]
    last_dtype = in_dtypes[-1]
%>
    %for i in xrange(len(in_dtypes) - 2, -1, -1):
    <%
        dt = in_dtypes[i]

        new_dtype = dtypes.result_type(last_dtype, dt)
        if dtypes.is_double(new_dtype) and not dtypes.is_double(out_dtype):
            new_dtype = numpy.complex64 if dtypes.is_complex(new_dtype) else numpy.float32

        ca = dtypes.is_complex(dt)
        cb = dtypes.is_complex(last_dtype)
        a = argnames[i]
        b = last_result

        temp_name = "temp" + str(i)
        result_ctr = dtypes.complex_ctr(new_dtype) if dtypes.is_complex(new_dtype) else ""
    %>
        ${dtypes.ctype(new_dtype)} ${temp_name }= ${result_ctr}(
        %if not ca and not cb:
            ${a} * ${b}
        %elif ca and not cb:
            ${a}.x * ${b}, ${a}.y * ${b}
        %elif not ca and cb:
            ${b}.x * ${a}, ${b}.y * ${a}
        %else:
            ${a}.x * ${b}.x - ${a}.y * ${b}.y, ${a}.x * ${b}.y + ${a}.y * ${b}.x
        %endif
            );
    <%
        last_dtype = new_dtype
        last_result = temp_name
    %>
    %endfor

    ## Cast output
    <%
        c_res = dtypes.is_complex(last_dtype)
        c_out = dtypes.is_complex(out_dtype)
    %>
    %if not c_res and not c_out:
    return ${last_result};
    %elif not c_res and c_out:
    return ${dtypes.complex_ctr(out_dtype)}(${last_result}, 0);
    %elif c_res and not c_out:
    return ${last_result}.x;
    %else:
    return ${dtypes.complex_ctr(out_dtype)}(${last_result}.x, ${last_result}.y);
    %endif
}
</%def>


<%def name="div(prefix)">
WITHIN_KERNEL ${dtypes.ctype(out_dtype)} ${prefix}(
    ${dtypes.ctype(in_dtype1)} a, ${dtypes.ctype(in_dtype2)} b)
{
<%
    c1 = dtypes.is_complex(in_dtype1)
    c2 = dtypes.is_complex(in_dtype2)
    if dtypes.is_complex(out_dtype):
        out_ctr = dtypes.complex_ctr(out_dtype)
    else:
        out_ctr = ""

    if not c1 and not c2:
        result = "a / b"
    elif c1 and not c2:
        result = out_ctr + "(a.x / b, a.y / b)"
    elif not c1 and c2:
        result = out_ctr + "(a * b.x / (b.x * b.x + b.y * b.y), -a * b.y / (b.x * b.x + b.y * b.y))"
    else:
        result = out_ctr + "((a.x * b.x + a.y * b.y) / (b.x * b.x + b.y * b.y), " + \
            "(-a.x * b.y + a.y * b.x) / (b.x * b.x + b.y * b.y))"
%>
    return ${result};
}
</%def>


<%def name="norm(prefix)">
<%
    if dtypes.is_complex(dtype):
        out_dtype = dtypes.real_for(dtype)
        result = "a.x * a.x + a.y * a.y"
    else:
        out_dtype = dtype
        result = "abs(a)"
%>
WITHIN_KERNEL ${dtypes.ctype(out_dtype)} ${prefix}(${dtypes.ctype(dtype)} a)
{
    return ${result};
}
</%def>


<%def name="conj(prefix)">
WITHIN_KERNEL ${dtypes.ctype(dtype)} ${prefix}(${dtypes.ctype(dtype)} a)
{
    return ${dtypes.complex_ctr(dtype) + "(a.x, -a.y)"};
}
</%def>


<%def name="sincos(dtype)">
    ${dtypes.ctype(dtypes.complex_for(dtype))} res;

    #ifdef CUDA
        ${"sincos" + ("" if dtypes.is_double(dtype) else "f")}(theta, &(res.y), &(res.x));
    #else
    ## It seems that native_cos/sin option is only available for single precision.
    %if not dtypes.is_double(dtype):
    #ifdef CTX_FAST_MATH
        res.x = native_cos(theta);
        res.y = native_sin(theta);
    #else
    %endif
        real_t tmp;
        res.y = sincos(theta, &tmp);
        res.x = tmp;
    %if not dtypes.is_double(dtype):
    #endif
    %endif
    #endif
</%def>


<%def name="exp(prefix)">
WITHIN_KERNEL ${dtypes.ctype(dtype)} ${prefix}(${dtypes.ctype(dtype)} a)
{
    %if dtypes.is_real(dtype):
    return exp(a);
    %else:
    ${sincos(dtypes.real_for(dtype))}
    ${dtypes.ctype(dtypes.real_for(dtype))} rho = exp(a.x);
    res.x *= rho;
    res.y *= rho;
    return res;
    %endif
}
</%def>


<%def name="polar(prefix)">
<%
    out_dtype = dtypes.complex_for(dtype)
%>
WITHIN_KERNEL ${dtypes.ctype(out_dtype)} ${prefix}(
    ${dtypes.ctype(dtype)} rho, ${dtypes.ctype(dtype)} theta)
{
    ${sincos(dtype)}
    res.x *= rho;
    res.y *= rho;
    return res;
}
</%def>

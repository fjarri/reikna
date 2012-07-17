<%def name="mul(name, out_dtype, dtype1, dtype2)">
WITHIN_KERNEL ${dtypes.ctype(out_dtype)} ${name}(
    ${dtypes.ctype(dtype1)} a, ${dtypes.ctype(dtype2)} b)
{
<%
    c1 = dtypes.is_complex(dtype1)
    c2 = dtypes.is_complex(dtype2)
    if dtypes.is_complex(out_dtype):
        out_ctr = dtypes.complex_ctr(out_dtype)

    if not c1 and not c2:
        result = "a * b"
    elif c1 and not c2:
        result = out_ctr + "(a.x * b, a.y * b)"
    elif not c1 and c2:
        result = out_ctr + "(b.x * a, b.y * a)"
    else:
        result = out_ctr + "(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x)"
%>
    return ${result};
}
</%def>

<%def name="cast(name, out_dtype, in_dtype)">
WITHIN_KERNEL ${dtypes.ctype(out_dtype)} ${name}(${dtypes.ctype(in_dtype)} x)
{
<%
    if dtypes.is_complex(out_dtype) and not dtypes.is_complex(in_dtype):
        result = dtypes.complex_ctr(out_dtype) + "(x, 0)"
    elif not dtypes.is_complex(out_dtype) and not dtypes.is_complex(in_dtype):
        result = "(" + dtypes.ctype(out_dtype) + ")x"
    else:
        raise NotImplementedError("Cast from " + str(in_dtype) + " to " + str(out_dtype) +
            "is not supported")
%>
    return ${result};
}
</%def>

%for name in functions:
<%
    funcs = dict(mul=mul, cast=cast)
    func_name, args = functions[name]
    func = funcs[func_name]
%>
${func(name, *args)}
%endfor

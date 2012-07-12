KERNEL void dummy(${signature})
{
    int idx = GLOBAL_INDEX;
    ${ctype.A} a = ${load.A}(idx);
    ${ctype.B} b = ${load.B}(idx);
    ${ctype.C} c = a + ${mul(dtype.coeff, dtype.B)}(${param.coeff}, b);
    ${store.C}(idx, c);
}

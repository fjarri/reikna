<%def name="transpose(output, input)">

<%
    ctype = dtypes.ctype(basis.dtype)
%>

${kernel_definition}
{
    // To prevent shared memory bank confilcts:
    // - Load each component into a different array. Since the array size is a
    //   multiple of the number of banks, each thread reads x and y from
    //   the same bank. If a single value_pair array is used, thread n would read
    //   x and y from banks n and n+1, and thread n+8 would read values from the
    //   same banks - causing a bank conflict.
    // - Use (half) warp size + 1 as the x size of the array. This way each row of the
    //   array starts in a different bank - so reading from shared memory
    //   doesn't cause bank conflicts when writing the transpose out to global
    //   memory.
    LOCAL_MEM ${ctype} block[(${block_width} + 1) * ${block_width}];

    unsigned int lid_x = LID_FLAT % ${block_width};
    unsigned int lid_y = LID_FLAT / ${block_width};

    unsigned int gid_x = GID_FLAT % ${grid_width};
    unsigned int gid_y = GID_FLAT / ${grid_width};

    unsigned int xBlock = ${block_width} * gid_x;
    unsigned int yBlock = ${block_width} * gid_y;
    unsigned int xIndex = xBlock + lid_x;
    unsigned int yIndex = yBlock + lid_y;
    unsigned int index_block = lid_y * (${block_width} + 1) + lid_x;
    unsigned int index_transpose = lid_x * (${block_width} + 1) + lid_y;
    unsigned int index_in = ${basis.input_width} * yIndex + xIndex;
    unsigned int index_out = ${basis.input_height} * (xBlock + lid_y) + yBlock + lid_x;

    for(int n = 0; n < ${basis.batch}; ++n)
    {
        if(xIndex < ${basis.input_width} && yIndex < ${basis.input_height})
            block[index_block] = ${input.load}(index_in);

        LOCAL_BARRIER;

        if(xBlock + lid_y < ${basis.input_width} && yBlock + lid_x < ${basis.input_height})
            ${output.store}(index_out, block[index_transpose]);

        index_in += ${basis.input_width * basis.input_height};
        index_out += ${basis.input_width * basis.input_height};
    }
}

</%def>

Core classes
============

Computations and transformations are based on two classes, which provide the user-side interface of computation objects.

.. py:module:: tigger.core

.. py:class:: Computation(ctx, debug=False)

    Creates a computation class and performs basic initialization for the :py:class:`~tigger.cluda.api.Context` object ``ctx``.
    Note that the computation is unusable until :py:func:`prepare` or :py:func:`prepare_for` is called.
    If ``debug`` is ``True``, a couple of additional checks will be performed in runtime during preparation and calls to computation.

    .. py:method:: connect(tr, array_arg, new_array_args, new_scalar_args=None)

        Connects a :py:class:`Transformation` instance to the computation.
        After the successful connection the computation resets to teh unprepared state.

        :param array_arg: name of the leaf computation parameter to connect to.
        :param new_array_args: list of the names for the new leaf array parameters.
        :param new_scalar_args: list of the names for the new leaf scalar parameters.

    .. py:method:: set_argnames(outputs, inputs, scalars)

        Set argument names for the computation.
        This method should be called first after the creation of the computation object, and is only available in the computations with variable argument number (for example, :py:class:`~tigger.elementwise.Elementwise`.)

    .. py:method:: set_basis(**kwds)

        Changes given values in the basis.
        Does not initiate preparation.

    .. py:method:: set_basis_for(*args, **kwds)

        Changes the basis so that the computation could be called with ``args`` supplied to :py:meth:`__call__`.
        Does not initiate preparation.

    .. py:method:: prepare(**kwds)

        Prepare the computation based on the given basis parameters.

    .. py:method:: prepare_for(*args, **kwds)

        Prepare the computation so that it could run with ``args`` supplied to :py:meth:`__call__`.

    .. py:method:: __call__(*args)

        Execute computation with given arguments.
        The order and types of arguments are defined by the base computation and connected transformations.
        The signature can be also viewed by means of :py:meth:`signature_str`.

    .. py:method:: signature_str()

        Returns a string with the signature of the computation, containing argument names, types and shapes (in case of arrays).

    The following methods are for overriding by computations inheriting :py:class:`Computation` class.

    .. py:method:: _get_argnames()

        Must return a tuple ``(outputs, inputs, scalars)``, where each of ``outputs``, ``inputs``, ``scalars`` is a tuple of argument names used by this computation.
        If this method is not overridden, :py:meth:`set_argnames` will have to be called right after creating the computation object.

    .. py:method:: _get_default_basis()

        Must return a dictionary with default values for the computation basis.

    .. py:method:: _get_argvalues(argnames, basis)

        Must return a dictionary with :py:class:`ArrayValue` and :py:class:`ScalarValue` objects assigned to the argument names.

    .. py:method:: _get_basis_for(argnames, *args, **kwds)

        Must return a dictionary with basis values for the computation working with ``args``, given optional parameters ``kwds``.
        If names of positional and keyword arguments are known in advance, it is better to use them explicitly in the signature.

    .. py:method:: _construct_operations(operations, argnames, basis, device_params)

        Must fill the ``operations`` object with actions required to execute the computation.
        See the :py:class:`OperationRecorder` class reference for the list of available actions.

.. py:class:: Transformation(inputs=1, outputs=1, parameters=0, derive_o_from_ip=None,         derive_ip_from_o=None, derive_i_from_op=None, derive_op_from_i=None, code="${store.s1}(${load.l1});")

        Creates an elementwise transformation.

        :param inputs: number of input array values.
        :param outputs: number of output array values.
        :param parameters: number of scalar parameters for the transformation.
        :param derive_o_from_ip: a function taking ``inputs`` + ``parameters`` dtype parameters and returning list with ``outputs`` dtypes.
            Used to derive types in the transformation tree after call to :py:meth:`Computation.prepare_for` when the transformation is connected to the input argument.
        :param derive_ip_from_o: a function taking ``outputs`` dtype parameters and returning tuple of two lists with ``inputs`` and ``parameters`` dtypes.
            Used to derive types in the transformation tree after call to :py:meth:`Computation.prepare` when the transformation is connected to the input argument.
        :param derive_i_from_op: a function taking ``outputs`` + ``parameters`` dtype parameters and returning list with ``inputs`` dtypes.
            Used to derive types in the transformation tree after call to :py:meth:`Computation.prepare_for` when the transformation is connected to the output argument.
        :param derive_op_from_i: a function taking ``inputs`` dtype parameters and returning tuple of two lists with ``outputs`` and ``parameters`` dtypes.
            Used to derive types in the transformation tree after call to :py:meth:`Computation.prepare` when the transformation is connected to the output argument.
        :param code: template source with the transformation code.
            See :ref:`How to write transformations <how-to-transformations>` section for details.

.. py:class:: OperationRecorder

    .. py:method:: add_allocation(name, shape, dtype)

        Adds an allocation to the list of actions.
        The ``name`` can be used later in the list of argument names for kernels.

    .. py:method:: add_kernel(template, defname, argnames, global_size, local_size=None, render_kwds=None)

        Adds kernel execution to the list of actions.
        See the details on how to write kernels in the :ref:`kernel writing guide <guide-contributing>`.

        :param template: Mako template for the kernel.
        :param defname: name of the definition inside the template.
        :param argnames: names of the arguments the kernel takes.
            These must either belong to the list of external argument names, or be allocated by :py:meth:`add_allocation` earlier.
        :param global_size: global size to use for the call.
        :param local_size: local size to use for the call.
            If ``None``, the local size will be picked automatically.
        :param render_kwds: dictionary with additional values used to render the template.

    .. py:method:: add_computation(computation, *argnames)

        Adds a nested computation call. The ``computation`` value must be a computation with necessary basis set and transformations connected. ``argnames`` list specifies which positional arguments will be passed to this kernel.

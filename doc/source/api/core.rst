Core functionality
==================

Under construction.


.. _how-to-transformations:

How to write transformations
----------------------------

Under construction.


Reference
---------

Computations and transformations are based on two classes, which provide the user-side interface of computation objects.

.. py:module:: tigger.core

.. py:class:: Computation(ctx, debug=False)

	Creates a computation class and performs basic initialization for the CLUDA context ``ctx``.
	Note that the computation is unusable until :py:func:`prepare` or :py:func:`prepare_for` is called.
	If ``debug`` is ``True``, a couple of additional checks will be performed in runtime during preparation and calls to computation.

	.. py:method:: connect(tr, array_arg, new_array_args, new_scalar_args=None)

		Connects a :py:class:`Transformation` instance to the computation.
		After the successful connection the computation has to be prepared again.

		:param array_arg: name of the leaf computation parameter to connect to.
		:param new_array_args: list of the names for the new leaf array parameters.
		:param new_scalar_args: list of the names for the new leaf scalar parameters.

	.. py:method:: prepare(**kwds)

		Prepare the computation based on the given basis parameters.

	.. py:method:: prepare_for(*args)

		Prepare the computation so that it could run with the ``args`` supplied to :py:meth:`__call__`.

	.. py:method:: __call__(*args)

		Execute computation with given arguments.
		The order and types of arguments are defined by the base computation and connected transformations.
		The signature can be also viewed by means of :py:meth:`signature_str`.

	.. py:method:: signature_str()

		Returns a string with the signature of the computation, containing argument names, types and shapes (in case of arrays).

.. py:class:: Transformation(load=1, store=1, parameters=0, derive_o_from_ip=None,         derive_ip_from_o=None, derive_i_from_op=None, derive_op_from_i=None, code="${store.s1}(${load.l1});")

		Creates an elementwise transformation.

		:param load: number of input array values.
		:param store: number of output array values.
		:param parameters: number of scalar parameters for the transformation.
		:param derive_o_from_ip: a function taking ``load`` + ``parameters`` dtype parameters and returning list with ``store`` dtypes.
			Used to derive types in the transformation tree after call to :py:meth:`Computation.prepare_for` when the transformation is connected to the input argument.
		:param derive_ip_from_o: a function taking ``store`` dtype parameters and returning tuple of two lists with ``load`` and ``parameters`` dtypes.
			Used to derive types in the transformation tree after call to :py:meth:`Computation.prepare` when the transformation is connected to the input argument.
		:param derive_i_from_op: a function taking ``store`` + ``parameters`` dtype parameters and returning list with ``load`` dtypes.
			Used to derive types in the transformation tree after call to :py:meth:`Computation.prepare_for` when the transformation is connected to the output argument.
		:param derive_op_from_i: a function taking ``load`` dtype parameters and returning tuple of two lists with ``store`` and ``parameters`` dtypes.
			Used to derive types in the transformation tree after call to :py:meth:`Computation.prepare` when the transformation is connected to the output argument.
		:param code: template source with the transformation code.
			See :ref:`How to write transformations <how-to-transformations>` section for details.

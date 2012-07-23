.. _cluda-reference:

CLUDA layer
===========

CLUDA is the foundation of the tigger.
It provides unified access to basic features of CUDA and OpenCL, such as memory operations, compilation and so on.
It can also be used by itself, if you want to write GPU API-independent programs and happen to need the small subset of GPU API.

Root level interface
--------------------

This module contains functions for API discovery.

.. automodule:: tigger.cluda
   :members:

API module
----------

Modules for all APIs have the same generalized interface.
It is referred here (and references from other parts of this documentation) as :py:mod:`tigger.cluda.api`.

.. py:module:: tigger.cluda.api

.. py:attribute:: API_ID

   Identifier of this API.

.. py:class:: Context(self, context, stream=None, fast_math=True, async=True)

   Wraps existing context in the CLUDA context object.

   :param context: a context to wrap
   :type context: :py:class:`pycuda.driver.Context` object for ``API_CUDA``, or :py:class:`pyopencl.Context` object for ``API_OCL``.
   :param stream: a stream to serialize operations to.
                  If not given, a new one will be created internally.
   :type stream: :py:class:`pycuda.driver.Stream` object for ``API_CUDA``, or :py:class:`pyopencl.CommandQueue` object for ``API_OCL``.
   :param fast_math: whether to enable fast mathematical operations during compilation.
   :param async: whether to execute all operations with this context asynchronously.

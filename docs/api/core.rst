Core functionality
==================

Classes necessary to create computations and transformations are exposed from the :py:mod:`~reikna.core` module.

.. module:: reikna.core


Computation signatures
----------------------

.. autoclass:: reikna.core.Type
    :members:
    :special-members: __call__

.. autoclass:: reikna.core.Annotation
    :show-inheritance:
    :members:

.. autoclass:: reikna.core.Parameter
    :show-inheritance:
    :members:

.. autoclass:: reikna.core.Signature
    :show-inheritance:
    :members:


Core classes
------------

.. autoclass:: reikna.core.Computation
    :members:
    :private-members: _build_plan

.. autoclass:: reikna.core.Transformation
    :members:


Result and attribute classes
----------------------------

.. autoclass:: reikna.core.Indices
    :members:
    :special-members: __getitem__

.. automodule:: reikna.core.computation
    :members: ComputationCallable, ComputationParameter, KernelArgument, ComputationPlan

.. automodule:: reikna.core.transformation
    :members: TransformationParameter, KernelParameter

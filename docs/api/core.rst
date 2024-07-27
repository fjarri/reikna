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
    :members:

.. autoclass:: reikna.core.Parameter
    :members:

.. autoclass:: reikna.core.Signature
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
    :special-members: __call__

.. automodule:: reikna.core.transformation
    :members: TransformationParameter, KernelParameter


Array tools
-----------

.. autofunction:: reikna.concatenate

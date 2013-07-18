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

.. autoclass:: reikna.core.Transformation
    :members:


Result and attribute classes
----------------------------

.. automodule:: reikna.core.computation
    :exclude-members: Computation
    :members:
    :special-members: __call__

.. automodule:: reikna.core.transformation
    :exclude-members: Transformation
    :members:

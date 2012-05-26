Codename Tigger
===============

For some time this will be used for a mixture of TODO and proper documentation.

Pre- and post-processing
========================

Elementwise pre- and post-processing can be attached to any kernel (derived from Computation class).
Pre-processing is invoked when kernel reads from memory, and post-processing is invoked when kernel writes to memory.
Pre-processing has to have only one output value, and post-processing has to have only one input value.
They can change variable types as long as there is a function that derives output type from input types (for load) or input types from output types (for store); by default these types are equal.
When computation has some processing attached to it, its signature changes



Contents
========

.. toctree::
   :maxdepth: 2

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


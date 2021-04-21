Installation
===================================

``HipoMap`` supports Python 3.
It requires ``numpy``, ``pandas``, ``tensorflow``, ``scipy``, ``scikit-learn``, ``seaborn``, ``matplotlib``, ``cv2``, ``openslide-python``. Most of them are automatically installed during installation. However, in the case of ``openslide-python``, if it is not installed, you should install it as follows:



Installing openslide-python
-----------------------------------

Before installing ``openslide-python``, ``openslide-tools`` should be installed.

1. Install openslide-tools

.. code-block::

   sudo apt-get update
   sudo apt-get install openslide-tools
   
2. Install openslide-python

.. code-block::
   
   pip install openslide
   

Installing HipoMap
-----------------------------------

``HipoMap`` is easy to install through pip3.

.. code-block::
   
   pip3 install HipoMap
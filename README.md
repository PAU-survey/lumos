# Lumos
Repository for Lumos, a code for extracting photometry from astronomical images with neural networks.

The code performs the following steps:

* Creates image cutouts around the positions given externaly 
* Evaluates Lumos on the galaxy cutout

# Installation
Clone the respository and write

pip install -e .

after entering into the cloned directory.

## Example usage
Examples are available in examples/integration_example_internalDB.ipynb and examples/integration_example_externalDB.ipynb


examples/integration_example_internalDB.ipynb works only for users with access to the PAUS DB
examples/integration_example_externalDB.ipynb works for everybody

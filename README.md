# Lumos
Repository for Lumos, a code for extracting photometry from astronomical images with neural networks.

The code performs the following steps:

* Creates image cutouts around the positions given externaly 
* Evaluates Lumos on the galaxy cutout


# Lumos training
A further script named 'lumos_train' allows to train the network from scratch with any data set.
This requires a set of N training cutouts of 60x60 pixels and their profiles.



# Installation
Clone the respository and write

pip install -e .

after entering into the cloned directory.

## Example usage
Examples are available in examples/integration_example_internalDB.ipynb and examples/integration_example_externalDB.ipynb


examples/integration_example_internalDB.ipynb works only for users with access to the PAUS DB
examples/integration_example_externalDB.ipynb works for everybody

For the example external to the database, the user needs to download a reduced PAUS image example.
This is available at ://www.pausurvey.org/pausurvey/data-processing/ 

An example of trained model is also available at the examples directory.

# Acknowledgement
This project has received funding from the European Union’s Horizon 2020 research
and innovation programme under the grant agreement No
776247 EWC.

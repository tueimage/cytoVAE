# cytoVAE - Extended Variational Auto-Encoder for Single-Cell Representation Learning
This repository contains the code of a tensorflow implementation of the auto-encoder model presented in the following publication:

\[1\] *Capturing Single-Cell Phenotypic Variation via Unsupervised Representation Learning* ;
Maxime W. Lafarge, Juan C. Caicedo, Anne E. Carpenter, Josien P.W. Pluim, Shantanu Singh, Mitko Veta ;
MIDL 2019; PMLR 102:315-325 

## Content
- **training_cytoVAE.py**: Main script to run the training procedure.
- **models/VAEPlus/**: Directory containing the model components (architecture, loss definition, optimizers, training procedure).
- **dataManagers/**: Abstract module to handle a dataset and generate training batches of images.
- **exp_cytoVAE_demo/**: Abstract module to define an experimental setup (imported by the main script).


## Requirements
The current code was developed and tested with the following configuration:
- *python  3.4.5*
- *tensorflow-gpu  1.14.0*
- *numpy  1.16.4*


## Notes
- Dataset and code to import images need to be added in order for the script to run (changes to *dataManagers* and *exp_cytoVAE_demo* must also be adapted to the user's project).

#-*- coding: utf-8 -*-
"""
@author: Maxime W. Lafarge, (mlafarge); Eindhoven University of Technology, The Netherlands
@comment: For more details see "Capturing Single-Cell Phenotypic Variation via Unsupervised Representation Learning"; MW Lafarge et al.; MIDL 2019; PMLR 102:315-325 

Package of the implementation of the tensorflow model.
"""

from . import parameters
from . import loss

from .inference import inference
from .optimization import optimization

from .trainer import Trainer
from .monitor import Monitor
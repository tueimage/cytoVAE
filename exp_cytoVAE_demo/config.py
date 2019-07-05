#-*- coding: utf-8 -*-
"""
@author: Maxime W. Lafarge, (mlafarge); Eindhoven University of Technology, The Netherlands
@comment: For more details see "Capturing Single-Cell Phenotypic Variation via Unsupervised Representation Learning"; MW Lafarge et al.; MIDL 2019; PMLR 102:315-325

Configuration file for a given experiment
Combines:
- local path definition
- dataset parametrization
- model parameterization
"""

#-- Experiment name
name_ext = ""
name = __name__.split(".")[-2] + name_ext

#-- Experiment root path
import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
path_exp_root = os.path.dirname(os.path.realpath(__file__)) #-- Path to the root of the experiment

#-- Experiment storage directories paths
path_logs       = path_exp_root + os.sep + "logs"
path_monitoring = path_exp_root + os.sep + "monitoring"

#-- Local directories creation
def mkDirIfNew(path_dir):
	if not os.path.exists(path_dir):
		os.makedirs(path_dir)
		print(">> New directory created:", path_dir)

mkDirIfNew(path_logs)
mkDirIfNew(path_monitoring)


#-- Dataset manager
""" !!!!!
*** THE COMPONENT TO MANAGE/READ DATASET IS PROJECT-DEPENDENT
*** IT IS NOT PART OF THE SHARED MODEL AND MUST BE IMPLEMENTED BY THE USER
"""
from dataManagers import dataManager_ABSTRACT as dManager

#-- Model
from models import VAEPlus as model


#-- Experiment parameters
gpu_memory_fraction = None
maxIterations    = 80000
validationPeriod = 250

#-- Model recovery
path_to_restore =  None #-- Path to an existing model

print(">> Configuration loaded.")

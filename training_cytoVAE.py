# -*- coding: utf-8 -*-
"""
@author: Maxime W. Lafarge, (mlafarge); Eindhoven University of Technology, The Netherlands
@comment: For more details see "Capturing Single-Cell Phenotypic Variation via Unsupervised Representation Learning"; MW Lafarge et al.; MIDL 2019; PMLR 102:315-325

Master script to run the training procedure of the model.
"""

"""
 1) IMPORT CURRENT EXPERIMENT
"""
from experiment import exp
dManager = exp.config.dManager #-- Data Manager Class
model    = exp.config.model #-- Model

"""
 2) INITIALIZE THE TRAINING CLASS
"""
gpu_memory_fraction = exp.config.gpu_memory_fraction
trainer = model.Trainer (
	name         = exp.config.name,
	path2restore = exp.config.path_to_restore, #-- Model state recovey
	model        = model, #-- Imported model
	
	monitoring  = True,
	is_training = True,
	
	gpu_memory_fraction=gpu_memory_fraction)

"""
 3) TRAINING ITERATIONS
"""
for step in range(exp.config.maxIterations):
	#------
	#-- 0) Booleans of the current iteration
	isValidation = (step+1) % exp.config.validationPeriod == 0
	
	#------
	#-- 1) Extract 2 independent image batches
	tensor_images = dManager.generateBatch()
	tensor_discrimination = dManager.generateBatch()
		
	#------
	#-- 2) Run a training iteration (VAE and Discriminator are trained in parallel)
	trainer.train(
		tensor_images      = tensor_images,
		tensor_images_disc = tensor_discrimination)
	
	#------
	#-- 4) Run a validation iteration
	if isValidation:
		""" USER-FREE VALIDATION PROCEDURE
		"""
		pass
		
		
print("Training done.")

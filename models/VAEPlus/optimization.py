# -*- coding: utf-8 -*-
"""
@author: Maxime W. Lafarge, (mlafarge); Eindhoven University of Technology, The Netherlands
@comment: For more details see "Capturing Single-Cell Phenotypic Variation via Unsupervised Representation Learning"; MW Lafarge et al.; MIDL 2019; PMLR 102:315-325

Optimization functions of the model
"""
import tensorflow as tf

from . import parameters as p_default
from . import flags

DEBUG = 0
def optimization(
		loss_rec_rgb, #-- Loss RGB reconstructions
		loss_rec_gan, #-- Loss Discriminator reconstructions
		loss_disc, #-- CE Loss of the Discriminator
		loss_kl, #-- KL diverence
		
		graph	   = tf.get_default_graph(),
		parameters = p_default):
	
	with graph.as_default() and tf.device(parameters.device):
		#-- Get the global step counter
		global_step = tf.get_collection(flags.GLOBAL_STEP)[0]
		
		##--------
		##-- TRAINING WEIGHTS 
		lr_adam = tf.constant(parameters.sgd_learningRate)
		lr_sgd  = tf.constant(parameters.adam_learningRate)
		mmt		= tf.constant(parameters.momentum)
		epsilon = tf.constant(parameters.adam_epsilon)
		
		##--------
		##-- INDEPENDENT GROUPS OF WEIGHTS TO OPTIMIZE
		variables_trainable	= tf.trainable_variables() #-- All trainables variables		
		variables_kernels	= tf.get_collection(flags.KERNEL)	
		
		variables_encoder = tf.get_collection(flags.ENCODER)  
		variables_decoder = tf.get_collection(flags.DECODER)  
		variables_discriminator = tf.get_collection(flags.DISCRIMINATOR)
		
		_tr_filt = lambda y: list(filter(lambda x: x in variables_trainable, y))
		variables_encoder = _tr_filt(variables_encoder)
		variables_decoder = _tr_filt(variables_decoder)
		variables_discriminator = _tr_filt(variables_discriminator)
		
		_ker_filt = lambda y: list(filter(lambda x: x in variables_kernels, y))
		kernels_encoder = _ker_filt(variables_encoder)
		kernels_decoder = _ker_filt(variables_decoder)
		kernels_discriminator = _ker_filt(variables_discriminator)
		
		
		##--------
		##-- CREATION OF THE DIFFERENT OPTIMIZERS		
		"""
		*** AUTO-ENCODER (CLASSIFICATION PART)
		"""
		opt_encoder = tf.train.AdamOptimizer(
				learning_rate = lr_adam,
				beta1		  = mmt,
				beta2		  = 0.999,
				epsilon		  = epsilon,
				use_locking	  = False)
		
		opt_decoder = tf.train.AdamOptimizer(
				learning_rate = lr_adam,
				beta1		  = mmt,
				beta2		  = 0.999,
				epsilon		  = epsilon,
				use_locking	  = False)
				
		opt_discriminator = tf.train.MomentumOptimizer(
				learning_rate = lr_sgd,
				momentum      = mmt)

		"""
		*** FULL OBJECTIVES
		"""
		beta = parameters.beta
		loss_enc = 1.0 * loss_rec_gan + 1.0 * loss_rec_rgb + beta * loss_kl
		loss_dec = 1.0 * loss_rec_gan + 1.0 * loss_rec_rgb
		
		
		##--------
		##-- CREATION OF THE UPDATE OPERATIONS
		"""
		*** ENCODER
		"""
		var_list = variables_encoder
		loss     = loss_enc
		training_op_encoder = opt_encoder.minimize(
				loss		= loss,
				var_list	= var_list,
				global_step = global_step) #-- GLOBAL_STEP INCREMENT 

		"""
		*** DECODER
		"""
		var_list = variables_decoder
		loss     = loss_dec
		training_op_decoder = opt_decoder.minimize(
				loss		= loss,
				var_list	= var_list)
		
		"""
		*** DISCRIMINATOR
		"""
		var_list = variables_discriminator
		loss     = loss_disc
		training_op_discriminator = opt_discriminator.minimize(
				loss		= loss,
				var_list	= var_list)
		
		"""
		*** EXPLICIT WEIGHT DECAY OPERATORS
		"""
		wd_coeff = parameters.weightDecay
		wd_encoder_op = tf.group(*[tf.assign(kt, (1.0 - wd_coeff)*kt) for kt in kernels_encoder])
		wd_decoder_op = tf.group(*[tf.assign(kt, (1.0 - wd_coeff)*kt) for kt in kernels_decoder])
		wd_discriminator_op = tf.group(*[tf.assign(kt, (1.0 - wd_coeff)*kt) for kt in kernels_discriminator])
		wd_op_all = tf.group(wd_encoder_op, wd_decoder_op, wd_discriminator_op)
		
		"""
		*** OUTPUT
		"""
		training_op_all = tf.group(training_op_encoder, training_op_decoder, training_op_discriminator)
		return training_op_all, wd_op_all
	
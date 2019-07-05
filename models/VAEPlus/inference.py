# -*- coding: utf-8 -*-
"""
@author: Maxime W. Lafarge, (mlafarge); Eindhoven University of Technology, The Netherlands
@comment: For more details see "Capturing Single-Cell Phenotypic Variation via Unsupervised Representation Learning"; MW Lafarge et al.; MIDL 2019; PMLR 102:315-325

Definition of the architecture of the CNN components of the model
"""

import tensorflow as tf
import numpy as np
import math

from . import flags
from . import parameters as p_default
from .layers_base import variable, xavierInit, batchNormalizeTensor, biasTensor

def inference(
		input_tensor_ph, #-- Input tensor
		input_tensor_real_ph, #-- Input tensor of real images for discrimination
		
		input_noise_ph,  #-- Input noise for the embedding
		is_training_ph, #-- Training boolean variable
		
		input_embedding_ph, #-- Input embedding
		
		parameters = p_default,
		graph	   = tf.get_default_graph()):
	""" Master architecture generation function.
		Calls the multiple CNN sub-components of the model
		INPUT:
			- input images (for auto-encoding)
			- input images (for discrimination)
			- noise for latent sampling
		OUTPUT:
			- sampled latent
			- embedding_means
			- embedding_logvars
			- logits of the discriminator
			- internal activations of the discriminator
			- tensorflow operations for BN moving moments
	"""
	with graph.as_default() and tf.device(parameters.device):
		#------------
		# Global Step (keep track of the training iterations)
		with tf.device("/cpu:0"):
			global_step = tf.Variable(0, name="global_step", trainable=False)
			tf.add_to_collection(flags.GLOBAL_STEP, global_step)	
		
		#------------
		# Encoding part
		embedding, embedding_means, embedding_logvars, mvg_mmt_encoder_op = encoder(
			input_tensor_ph,
			input_noise_ph,
			is_training_ph,
			parameters,
			graph)
		
		#------------
		# Optional input Embedding
		# #--> [computed embeddings * n] + [input embeddings * m]
		embedding_extended = tf.concat([embedding, input_embedding_ph], axis=0)
		
		#------------
		# Decoding part
		activations, mvg_mmt_decoder_op = decoder(
			embedding_extended,
			is_training_ph,
			parameters,
			graph)
		
		#------------
		# Reconstruction part
		reconstructions, mvg_mmt_rec_op = reconstructor(
			activations,
			is_training_ph,
			parameters,
			graph)
		
		#------------
		# Discrimination part
		reconstructions_full = reconstructions[-1]
		disc_input_parts = [
			input_tensor_real_ph, #-- Batch of independent real images
			reconstructions_full, #-- Batch of fake reconstructed images
			input_tensor_ph]      #-- Batch of original images to reconstruct
			
		disc_input_tensors = tf.concat(disc_input_parts, axis=0)
		disc_logits, disc_representations, mvg_mmt_discriminator_op = discriminator(
			disc_input_tensors,
			is_training_ph,
			parameters,
			graph)
		
		#------------
		# Output
		mvg_mmt_op = tf.group(mvg_mmt_encoder_op, mvg_mmt_decoder_op, mvg_mmt_rec_op, mvg_mmt_discriminator_op)
		return reconstructions, embedding_means, embedding_logvars, disc_logits, disc_representations, mvg_mmt_op


def encoder(
		input_tensor_ph, #-- Input tensor
		input_noise_ph,  #-- Input noise for the embedding
		is_training_ph, #-- Training boolean variable
		
		parameters = p_default,
		graph		= tf.get_default_graph()):
	""" Encoder CNN: models the posterior of the generative model.
		The CNN estimates Gaussian parameters from which latent are sampled using the reparameterization trick.
		INPUT:
			- input images
			- noise for latent sampling
			
		OUTPUT:
			- sampled latent
			- embedding_means
			- embedding_logvars
			- tensorflow operations for BN moving moments
	"""
	with graph.as_default() and tf.device(parameters.device):
		
		#------------
		# Architecture configuration
		network = parameters.encoder_network
		
		#------------
		# BN moving average operations
		movingMoments_op_list = []
		
		#------------
		# Input Rescaling
		#-- Input expected in range [0,1]
		input_tensor = input_tensor_ph
		
		#------------
		# Expected outputs
		embedding_means   = None
		embedding_logvars = None
		
		#------------
		# Construction of the layers
		last_activation = input_tensor		
		for layer in range(network.nbLayers):
				#-- Parameters of the layer				
				input_units  = network.nbUnits_input[layer]
				output_units = network.nbUnits_output[layer]
				kernel_size  = network.kernel_sizes[layer]
				
				do_bn		= network.batchNorms[layer]
				do_relu		= network.relus[layer]
				
				is_embedding = network.embedding[layer]
				
				do_pooling	= network.maxPoolings[layer]
				
				#-- Scope of the layer
				with tf.variable_scope("HL_{}".format(layer)) as _scope:
					input_activation = last_activation #-- Input of the current layer
							
					"""
					*** CONVOLUTION OPERATION
					"""					
					kernel_shape = [kernel_size, kernel_size, input_units, output_units]
					initializer  = tf.random_normal_initializer(
							mean	= 0.0,
							stddev = xavierInit(kernel_shape[0]*kernel_shape[1]*kernel_shape[2], kernel_shape[3]))
					kernel = variable(
							name		= "kernel",
							shape		= kernel_shape,
							initializer = initializer,
							flags_to_assign = [flags.KERNEL, flags.ENCODER])					
				
					conv = tf.nn.conv2d(
								input	= input_activation, #-- [batchSize, height, width, channels]
								filter  = kernel,
								strides = [1, 1, 1, 1], #-- Fully overlapping convolution
								padding = "VALID") #-- No padding
					averaging_axes = [0,1,2]
					
					"""
					*** BATCH NORMALIZATION
					"""
					conv_normalized = None
					if do_bn:
						conv_normalized, moving_moments_ops = batchNormalizeTensor(
								tensor		= conv, #-- Tensor to normalize
								nbUnits		= output_units, 
								is_training	= is_training_ph,
								
								averaging_axes       = averaging_axes,
								moving_average_coeff = parameters.bn_mvgAverage_coeff,
								
								flags_to_assign = [flags.ENCODER])
						movingMoments_op_list += moving_moments_ops #-- Append operations to the list
					else: #-- Regular bias addition
						conv_normalized, moving_moments_ops = biasTensor(
								tensor		= conv, #-- Tensor to normalize
								nbUnits		= output_units, 
								is_training	= is_training_ph,
								
								averaging_axes       = averaging_axes,
								moving_average_coeff = parameters.bn_mvgAverage_coeff,
								
								flags_to_assign = [flags.ENCODER])				
						movingMoments_op_list += moving_moments_ops #-- Append operations to the list
					
					"""
					*** EMBEDDING PARAMETERIZED SAMPLING
					"""
					conv_embedded = conv_normalized
					if is_embedding:
						embeddingSize = parameters.embeddingSize
						embedding_means   = conv_normalized[:,:,:,:embeddingSize]
						embedding_logvars = conv_normalized[:,:,:,embeddingSize:]
						
						#-- Expand noise to match the target shape (create spatial dimensions)
						embedding_noise = tf.expand_dims(input_noise_ph, axis=1)
						embedding_noise = tf.expand_dims(embedding_noise, axis=1)
						
						embedding_stddev = tf.exp(0.5 * embedding_logvars)
						conv_embedded = embedding_means + (embedding_noise * embedding_stddev)
					
					"""
					*** POOLING
					"""
					conv_pooled = conv_embedded
					if do_pooling:
						conv_pooled = tf.nn.max_pool(
							value   = conv_embedded,
							ksize   = [1, 2, 2, 1],
							strides = [1, 2, 2, 1],
							padding = "VALID")
						
					"""
					*** NON-LINEARITY
					"""
					conv_activated = conv_pooled
					if do_relu:
						leak_coeff = 0.01
						conv_activated = tf.maximum(leak_coeff * conv_pooled, conv_pooled) #-- Leaky ReLU activation
					tf.add_to_collection(flags.COLLECTION_ACTIVATION, conv_activated)
		
					#-- Loop on last activation
					last_activation = conv_activated

		return last_activation, embedding_means, embedding_logvars, tf.group(*movingMoments_op_list)


def decoder(
		embedding, #-- Input tensor
		is_training_ph, #-- Training boolean variable
		
		parameters = p_default,
		graph	   = tf.get_default_graph()):
	""" Construction of the architecture of the model
		INPUT:
			- embedding
			
		OUTPUT:
			- list of activations (to be reconstructed)
			- tensorflow operation for moving BN moments
	"""
	with graph.as_default() and tf.device(parameters.device):
		
		#------------
		# Architecture configuration
		network = parameters.decoder_network #-- ** ARCHITECTURE OF THE MODEL
		
		#------------
		# List of activations
		activations = []
		
		#------------
		# BN moving average operations
		movingMoments_op_list = []
		
		#------------
		# Input Rescaling
		#-- Input expected range of [0,1]
		input_tensor = embedding
		
		#------------
		# Construction of the layers
		last_activation = input_tensor		
		for layer in range(network.nbLayers):
				#-- Parameters of the layer
				input_units  = network.nbUnits_input[layer]
				output_units = network.nbUnits_output[layer]
				kernel_size  = network.kernel_sizes[layer]
				
				do_bn	= network.batchNorms[layer]
				do_relu	= network.relus[layer]
				
				do_padding		= network.paddings[layer]
				do_upsampling  = network.upSampling[layer]
				
				#-- Scope of the layer
				with tf.variable_scope("HL_{}".format(5+layer)) as _scope:
					input_activation = last_activation #-- Input of the current layer
					
					if do_padding:
						""" Spatial padding for transposed convolution
						"""
						pad_val = do_padding
						input_activation = tf.pad(
							tensor   = input_activation,
							paddings = tf.constant([[0,0], [pad_val,pad_val], [pad_val,pad_val], [0,0]]),
							mode     = "CONSTANT",
							constant_values = 0)
							
					"""
					*** CONVOLUTION OPERATION
					"""
					conv = None
					averaging_axes = [] #-- Averaging axes for batch normalization follow up
					
					kernel_shape = [kernel_size, kernel_size, input_units, output_units]
					initializer  = tf.random_normal_initializer(
							mean	= 0.0,
							stddev = xavierInit(kernel_shape[0]*kernel_shape[1]*kernel_shape[2], kernel_shape[3]))
					kernel = variable(
							name		= "kernel",
							shape		= kernel_shape,
							initializer = initializer,
							flags_to_assign = [flags.KERNEL, flags.DECODER])				
				
					conv = tf.nn.conv2d(
								input	= input_activation, #-- [batchSize, height, width, channels]
								filter  = kernel,
								strides = [1, 1, 1, 1], #-- Fully overlapping convolution
								padding = "VALID") #-- No padding
					averaging_axes = [0,1,2]

					"""
					*** BATCH NORMALIZATION
					"""
					conv_normalized = None
					if do_bn:
						conv_normalized, moving_moments_ops = batchNormalizeTensor(
								tensor		= conv, #-- Tensor to normalize
								nbUnits		= output_units, 
								is_training	= is_training_ph,
								averaging_axes 		= averaging_axes,
								moving_average_coeff = parameters.bn_mvgAverage_coeff,
								flags_to_assign = [flags.DECODER])
						movingMoments_op_list += moving_moments_ops #-- Append operations to the list
						
					else: #-- Regular bias addition
						conv_normalized, moving_moments_ops = biasTensor(
								tensor	   = conv, #-- Tensor to normalize
								nbUnits		= output_units, 
								is_training	= is_training_ph,
								averaging_axes 		= averaging_axes,
								moving_average_coeff = parameters.bn_mvgAverage_coeff,
								flags_to_assign = [flags.DECODER])			
						movingMoments_op_list += moving_moments_ops #-- Append operations to the list
					
					"""
					*** EMBEDDING PARAMETERIZED SAMPLING
					"""
					conv_embedded = conv_normalized

					
					"""
					*** NON-LINEARITY
					"""
					conv_activated = conv_embedded
					if do_relu:
						leak_coeff = 0.01
						conv_activated = tf.maximum(leak_coeff * conv_embedded, conv_embedded) #-- Leaky ReLU activation
					
					#-- Store intermediate activations
					activations.append(conv_activated)
					
					
					"""
					*** POOLING / UPSAMPLING
					"""
					conv_up = conv_activated
					if do_upsampling:
						up_size = do_upsampling 
						conv_up = tf.image.resize_nearest_neighbor( #-- NN-UPSAMPLING
							images = conv_activated,
							size   = [up_size]*2)
		
					#-- Loop on last activation
					last_activation = conv_up
		

		return activations, tf.group(*movingMoments_op_list)


def reconstructor(
		activations, #-- Input activations
		is_training_ph, #-- Training boolean variable
		
		parameters = p_default,
		graph	   = tf.get_default_graph()):
	""" Layer of 1x1 convolution to map the decoder output convolutions
		to the target number of channels for the reconstructions to match the shape of the inputs.
		
		INPUT:
			- list of activations to reconstruct
		OUTPUT:
			- list of learned reconstructions
			- tensorflow operation for moving BN moments
	"""
	with graph.as_default() and tf.device(parameters.device):
		
		#------------
		# Architecture configuration
		network = parameters.reconstructor_network #-- ** ARCHITECTURE OF THE MODEL
		
		#------------
		# BN moving average operations
		movingMoments_op_list = []
		
		#------------
		# List of reconstructions
		reconstructions = []
		
		#------------
		# List of activations to reconstruct
		for scale_idx, input_activation in enumerate(activations):
						
			#------------
			# Construction of the layers
			last_activation = input_activation #-- Input activation
			
			for layer in range(network.nbLayers):
				#-- Parameters of the layer			
				input_units  = network.nbUnits_input[layer]
				output_units = network.nbUnits_output[layer]
				kernel_size  = network.kernel_sizes[layer]
			
				do_bn		= network.batchNorms[layer]
				do_relu	= network.relus[layer]
			
			
				#-- Scope of the layer
				with tf.variable_scope("HL_recons_{}".format(scale_idx)) as _scope:
					"""
					*** CONVOLUTION OPERATION
					"""
					conv = None
					averaging_axes = [] #-- Averaging axes for batch normalization follow up
				
					kernel_shape = [kernel_size, kernel_size, input_units, output_units]
					initializer  = tf.random_normal_initializer(
							mean	= 0.0,
							stddev = xavierInit(kernel_shape[0]*kernel_shape[1]*kernel_shape[2], kernel_shape[3]))
					kernel = variable(
							name		= "kernel",
							shape		= kernel_shape,
							initializer = initializer,
							flags_to_assign = [flags.KERNEL, flags.DECODER])				
				
					conv = tf.nn.conv2d(
								input	= input_activation, #-- [batchSize, height, width, channels]
								filter  = kernel,
								strides = [1, 1, 1, 1], #-- Fully overlapping convolution
								padding = "VALID") #-- No padding
					averaging_axes = [0,1,2]
					
					"""
					*** BATCH NORMALIZATION
					"""
					conv_normalized = None
					if do_bn:
						conv_normalized, moving_moments_ops = batchNormalizeTensor(
								tensor		= conv, #-- Tensor to normalize
								nbUnits		= output_units, 
								is_training	= is_training_ph,
								
								averaging_axes = averaging_axes,
								averaging_size = 1 * parameters.batchSize, #-- !! Not on full batch !!
								moving_average_coeff = parameters.bn_mvgAverage_coeff,
								flags_to_assign = [flags.DECODER])				
						movingMoments_op_list += moving_moments_ops #-- Append operations to the list
						
					else: #-- Regular bias addition
						conv_normalized, moving_moments_ops = biasTensor(
								tensor			= conv, #-- Tensor to normalize
								nbUnits		= output_units, 
								is_training	= is_training_ph,
								
								averaging_axes = averaging_axes,
								averaging_size = 1 * parameters.batchSize, #-- !! Not on full batch !!
								moving_average_coeff = parameters.bn_mvgAverage_coeff,
								flags_to_assign = [flags.DECODER])					
						movingMoments_op_list += moving_moments_ops #-- Append operations to the list
					
					"""
					*** NON-LINEARITY
					"""
					conv_activated = conv_normalized
					if do_relu:
						leak_coeff = 0.01
						conv_activated = tf.maximum(leak_coeff * conv_normalized, conv_normalized) #-- Leaky ReLU activation
	
					#-- Loop on last activation
					last_activation = conv_activated
					
			#-- Store the reconstruction
			reconstructions.append(last_activation)


		return reconstructions, tf.group(*movingMoments_op_list)


def discriminator(
		input_tensor_ph, #-- Input tensor [real]+[fake]
		is_training_ph,  #-- Training boolean variable
		
		parameters = p_default,
		graph	   = tf.get_default_graph()):
	""" Discriminator branch for learned similarity metric.
		INPUT:
			- tensor of real images and fake reconstructions
		OUTPUT:
			- discimination logits
			- list of internal activations
			- tensorflow operation for moving BN moments
	"""
	with graph.as_default() and tf.device(parameters.device):
		
		#------------
		# Architecture configuration
		network = parameters.discriminator_network #-- ** ARCHITECTURE OF THE MODEL
		
		#------------
		# BN moving average operations
		movingMoments_op_list = [] #-- Moving moments of the current application
		
		#------------
		# Discriminative representations for learned similarity
		disc_representations = []
		
		#------------
		# Input Rescaling
		#-- Input expected range of [0,1]
		input_tensor = input_tensor_ph
		
		#------------
		# Construction of the layers
		last_activation = input_tensor		
		for layer in range(network.nbLayers):
				#-- Parameters of the layer				
				input_units  = network.nbUnits_input[layer]
				output_units = network.nbUnits_output[layer]
				kernel_size  = network.kernel_sizes[layer]
				
				do_bn	= network.batchNorms[layer]
				do_relu = network.relus[layer]
				
				do_pooling = network.maxPoolings[layer]
				
				#-- Scope of the layer
				with tf.variable_scope("HL_disc_{}".format(layer)) as _scope:
					input_activation = last_activation #-- Input of the current layer
							
					"""
					*** CONVOLUTION OPERATION
					"""
					conv = None
					averaging_axes = [] #-- Averaging axes for batch normalization follow up
					
					kernel_shape = [kernel_size, kernel_size, input_units, output_units]
					initializer  = tf.random_normal_initializer(
							mean   = 0.0,
							stddev = xavierInit(kernel_shape[0]*kernel_shape[1]*kernel_shape[2], kernel_shape[3]))
					kernel = variable(
							name		= "kernel",
							shape		= kernel_shape,
							initializer = initializer,
							flags_to_assign = [flags.KERNEL, flags.DISCRIMINATOR])				
				
					conv = tf.nn.conv2d(
								input	  = input_activation, #-- [batchSize, height, width, channels]
								filter  = kernel,
								strides = [1, 1, 1, 1], #-- Fully overlapping convolution
								padding = "VALID") #-- No padding
					averaging_axes = [0,1,2]
					
					
					"""
					*** BATCH NORMALIZATION
					"""
					conv_normalized = None
					if do_bn:
						conv_normalized, moving_moments_ops = batchNormalizeTensor(
								tensor		= conv, #-- Tensor to normalize
								nbUnits		= output_units, 
								is_training	= is_training_ph,
								
								averaging_axes = averaging_axes,
								averaging_size = 2 * parameters.batchSize, #-- !! Not on full batch !!
								moving_average_coeff = parameters.bn_mvgAverage_coeff,
								flags_to_assign = [flags.DISCRIMINATOR])				
						movingMoments_op_list += moving_moments_ops #-- Append operations to the list
						
					else: #-- Regular bias addition
						conv_normalized, moving_moments_ops = biasTensor(
								tensor		= conv, #-- Tensor to normalize
								nbUnits		= output_units, 
								is_training	= is_training_ph,
								
								averaging_axes = averaging_axes,
								averaging_size = 2 * parameters.batchSize, #-- !! Not on full batch !!
								moving_average_coeff = parameters.bn_mvgAverage_coeff,
								flags_to_assign = [flags.DISCRIMINATOR])					
						movingMoments_op_list += moving_moments_ops #-- Append operations to the list
					
					"""
					*** POOLING
					"""
					conv_pooled = conv_normalized
					if do_pooling:
						conv_pooled = tf.nn.max_pool( #-- MAX-POOL
							value   = conv_normalized,
							ksize   = [1, 2, 2, 1],
							strides = [1, 2, 2, 1],
							padding = "VALID")
						
					"""
					*** NON-LINEARITY
					"""
					conv_activated = conv_pooled
					if do_relu:					
						leak_coeff = 0.01
						conv_activated = tf.maximum(leak_coeff * conv_pooled, conv_pooled) #-- Leaky ReLU activation
					tf.add_to_collection(flags.COLLECTION_ACTIVATION, conv_activated)
		
					#-- Loop on last activation
					disc_representations.append(conv_activated) #-- Store current representation
					last_activation = conv_activated
		
		
		return last_activation, disc_representations, tf.group(*movingMoments_op_list)

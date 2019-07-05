# -*- coding: utf-8 -*-
"""
@author: Maxime W. Lafarge, (mlafarge); Eindhoven University of Technology, The Netherlands
@comment: For more details see "Capturing Single-Cell Phenotypic Variation via Unsupervised Representation Learning"; MW Lafarge et al.; MIDL 2019; PMLR 102:315-325

Loss functions of the model components
"""
import tensorflow as tf

from . import parameters as p_default
from . import flags


def loss_reconstruction_repr(
		representation_GAN_real_list, #-- Representation of the original images
		representation_GAN_fake_list,
		
		base_size = p_default.inputSize[0] * p_default.inputSize[1] * p_default.inputSize[2],
		
		graph	   = tf.get_default_graph(),
		parameters = p_default):
	""" 
		Adversarial-Induced pixel-wise reconstruction loss.
		1) Compute the reconstruction loss using the learned representation
		2) Calculate the weighting coefficients for the total sum loss
		3) Compute the weighted sum loss
	"""
	with tf.device(parameters.device) and graph.as_default():
		global_step = tf.get_collection(flags.GLOBAL_STEP)[0]
		step = tf.cast(global_step, tf.float32) #-- Global step
		
		#-- Schedulling function
		slope = p_default.adv_schedule_slope
		def schedule_weight(delay):
			""" Defines a delayed, linear, saturated schedulling function.
			"""
			step_norm = tf.maximum(0.0, step - delay)
			w = step_norm / slope
			w = tf.maximum(0.0, tf.minimum(1.0, w)) #-- Bounded weight
			return w
		
		discriminator_nbLayers = parameters.discriminator_network.nbLayers - 1
		delays = [slope * (k+1) for k in range(discriminator_nbLayers)] #-- Discriminator Representations
		
		loss_list  = [] #-- Store the final losses
		loss_total = tf.constant(0.0) #-- Total loss
		
		for delay, repr_real, repr_fake in zip(delays, representation_GAN_real_list, representation_GAN_fake_list):
			distance_L2 = tf.square(repr_real - repr_fake)
			distance_L2_batch = 0.5 * base_size * tf.reduce_mean(distance_L2, axis=[1,2,3]) #-- Rescaling to match the input shape
			loss_batch = tf.reduce_mean(distance_L2_batch, name="loss_reconstruction_GAN{}".format(delay)) #-- Averaging of the losses across the batch
			loss_list.append(loss_batch)
			
			loss_weight = schedule_weight(delay) #-- Loss Schedulling
			loss_total += loss_weight * loss_batch #-- Schedule-based weighted average
		
		return loss_total, loss_list
		
		
def loss_reconstruction_rgb(
		input_tensor_ph, #-- Input image
		output_tensor,

		graph	   = tf.get_default_graph(),
		parameters = p_default):
	"""
		Baseline VAE pixel-wise reconstruction loss.
		remark: Maximizing the data likelihood is equivalent to minimizing the L2-distance 
	"""
	with tf.device(parameters.device) and graph.as_default():
		input_maxNorm  = input_tensor_ph
		output_maxNorm = output_tensor
		
		distance_L2 = tf.square(input_maxNorm - output_maxNorm)
		
		#-- Dimension reduction
		distance_L2 = 0.5 * tf.reduce_sum(distance_L2, axis=[1,2,3]) #-- y,x,channels
		
		#-- Averaging of the losses across the batch
		loss_batch = tf.reduce_mean(distance_L2, name="loss_reconstruction_RGB")
	
		return loss_batch
		
				
def loss_KL(
		embedding_mean,
		embedding_logvar,

		graph	   = tf.get_default_graph(),
		parameters = p_default):
	""" KL divergence between the distribution of the embeddings of the VAE and a unit Gaussian prior.
	"""
	with tf.device(parameters.device) and graph.as_default():
		KL_dist = 0.5 * tf.reduce_sum(
				tf.exp(embedding_logvar) + embedding_mean*embedding_mean - 1.0 - embedding_logvar,
				axis=[1,2,3]) #-- y,x,channels
		
		loss_batch = tf.reduce_mean(KL_dist, name="loss_KL") #-- Averaging of the losses across the batch		
		return loss_batch


def loss_discrimination(
		discrimination_logits,

		graph	   = tf.get_default_graph(),
		parameters = p_default):
	""" Cross-Entropy loss for the classification task: reconstructed image vs. real image.
	"""
	with tf.device(parameters.device) and graph.as_default():
		#-- Sigmoid activation
		logits_reduced  = tf.reduce_sum(discrimination_logits, axis=[1,2]) #-- y,x reduction
		discrim_sigmoid = tf.sigmoid(logits_reduced)
		
		#-- Target probabilities [n*real] + [n*fake]
		batchSize = parameters.batchSize
		target_proba = tf.constant([1.0]*batchSize + [0.0]*batchSize)
		target_proba = tf.expand_dims(target_proba, axis=1)
		
		#-- Cross-Entropies
		epsilon = 1e-7
		cross_entropies_pos = -1.0 * target_proba * tf.log(discrim_sigmoid + epsilon)
		cross_entropies_neg = -1.0 * (1.0 - target_proba) * tf.log(1.0 - discrim_sigmoid + epsilon)
		
		cross_entropies_total = cross_entropies_pos + cross_entropies_neg
		
		loss_batch = tf.reduce_mean(cross_entropies_total, name="loss_CE_Discriminator") #-- Averaging of the losses across the batch		
		return loss_batch

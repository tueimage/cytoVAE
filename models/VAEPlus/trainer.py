# -*- coding: utf-8 -*-
"""
@author: Maxime W. Lafarge, (mlafarge); Eindhoven University of Technology, The Netherlands
@comment: For more details see "Capturing Single-Cell Phenotypic Variation via Unsupervised Representation Learning"; MW Lafarge et al.; MIDL 2019; PMLR 102:315-325

Definition of the Trainer class.
The Trainer class instanciate the full tensorflow model.
Methods are defined to run training/validation/test iterations.
"""

#-- Libraries
import numpy as np
import tensorflow as tf

import time
import copy

import os
import sys

				
class Trainer:
	"""
		Class to manage the training of a given parameterized model.
		Stores and manages local parameters and graph information.
	"""
	def __init__(self,
				name,
				
				path2restore = None, #-- Model state recovey
				session		 = None, 
				model        = None,
				
				is_training = True, #-- Training switch
				
				gpu_memory_fraction = None,
				**kwargs):
		
		##--------
		## TRAINER PRIVATE VARIABLES
		self._name = name
		self._step = 0
		self._model = model
		self._parameters = model.parameters
		
		##--------
		## MODEL INITIALIZATION
		self.graph = tf.Graph()
		with self.graph.as_default():
				##
				## 1: INITIALIZATION OF THE PLACEHOLDERS
				self.images_ph = tf.placeholder(
					dtype = tf.float32,
					shape = [None]+self._parameters.inputSize)
				
				self.images_real_ph = tf.placeholder( #-- Batch of real images
					dtype = tf.float32,
					shape = [None]+self._parameters.inputSize)
					
				self.noise_ph = tf.placeholder(
					dtype = tf.float32,
					shape = [None, self._parameters.embeddingSize])
				
				self.embedding_ph = tf.placeholder(
					dtype = tf.float32,
					shape = [None, None, None, self._parameters.embeddingSize])
					
				self.is_training_ph = tf.placeholder(
					dtype = tf.bool)
				
				
				##
				## PRE-PROCESSING NORMALIZATION
				_maxNorm = lambda x, axes=(1,2): x / tf.reduce_max(x, axis=axes, keep_dims=True)
				self.images_ph_norm      = _maxNorm(self.images_ph)
				self.images_real_ph_norm = _maxNorm(self.images_real_ph)
					
				
				##
				## 2: NETWORK INFERENCE
				self.reconstructions_list_op,\
				self.embedding_means, self.embedding_logvars,\
				self.discriminator_logits, self.discriminator_representations,\
				self.movingMoments_op = model.inference(
					input_tensor_ph      = self.images_ph_norm, #-- Input tensor for VAE
					input_tensor_real_ph = self.images_real_ph_norm, #-- Input for Discriminator
					
					input_noise_ph     = self.noise_ph, #-- Input noise for embedding sampling
					input_embedding_ph = self.embedding_ph, #-- Input embedding for application
					
					is_training_ph  = self.is_training_ph,
					parameters      = self._parameters,
					graph			= self.graph)
				
				self.reconstructions_op = self.reconstructions_list_op[-1] #-- RGB reconstruction
				
				
				##
				## 3: NETWORK LOSSES
				
				##
				## 3.1: RGB-RECONSTURUCTION
				self.loss_reconstruction_RGB_op = model.loss.loss_reconstruction_rgb(
					input_tensor_ph = self.images_ph_norm, #-- Original Inputs
					output_tensor   = self.reconstructions_op, #-- Input reconstructions
					graph = self.graph)
				
				##
				## 3.2: ADV-INDUCED RECONSTRUCTIONS
				discriminator_nbLayers = self._parameters.discriminator_network.nbLayers -1
				repr_indices = range(discriminator_nbLayers)
				batchSize    = self._parameters.batchSize
				
				#-- Split up the adversarial-learned representations
				self.representation_GAN_real_list = [
						self.discriminator_representations[r_idx][0:batchSize]
						for r_idx in repr_indices]
						
				self.representation_GAN_fake_list = [
						self.discriminator_representations[r_idx][batchSize:2*batchSize]
						for r_idx in repr_indices]
						
				self.representation_GAN_original_list = [
						self.discriminator_representations[r_idx][2*batchSize:]
						for r_idx in repr_indices]
				
					
				##
				## 3.4: SCHEDULLED RECONSTRUCTION LOSS
				self.loss_reconstruction_GAN_op, self.loss_reconstruction_GAN_list_op = model.loss.loss_reconstruction_repr(
					representation_GAN_real_list = self.representation_GAN_original_list, #-- Representation of the original images
					representation_GAN_fake_list = self.representation_GAN_fake_list, #-- Current representation
					graph = self.graph)
				
				#-- KL prior constraint
				self.loss_kl_op = model.loss.loss_KL(
					self.embedding_means,
					self.embedding_logvars,
					graph = self.graph)
				
				#-- Discriminator loss
				self.loss_discriminator_op = model.loss.loss_discrimination(
					discrimination_logits = self.discriminator_logits[0:2*batchSize], #[real]+[fake]
					graph = self.graph)
				self.discrimination_predictions_op = tf.sigmoid(self.discriminator_logits) #-- Sigmoid activation
				

				##
				## 4: NETWORK OPTIMIZATION
				self.training_op, self.wd_op_all = model.optimization(
					loss_rec_rgb   = self.loss_reconstruction_RGB_op, #-- L2 reconstructions
					loss_kl        = self.loss_kl_op,
					loss_rec_gan = self.loss_reconstruction_GAN_op, #-- Learned similarity
					loss_disc    = self.loss_discriminator_op, #-- Cross-Entropy
					graph	     = self.graph)	
			
				
				##
				## LOGS AND INITIALIZATION
				self.initializer = tf.initialize_all_variables()
				self.saver		 = tf.train.Saver(max_to_keep=None)
	
				##
				## SESSION INITIALIZATION
				gpu_config_kwargs = {}
				if gpu_memory_fraction: #-- GPU configuration
					gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = gpu_memory_fraction)
					gpu_config_kwargs["config"] = tf.ConfigProto(
						log_device_placement = False,
						gpu_options = gpu_options)
		
				self.session = tf.Session(graph=self.graph, **gpu_config_kwargs) #-- Creation of the tensorflow session
				self.session.run(self.initializer) #-- Run initializer
				
					
				##
				## MODEL RECOVERY
				if path2restore: #-- If there is a path to a checkpoint
					with self.graph.as_default():
						print(">> PATH TO BE RETORED: ", path2restore)
						self.saver.restore(self.session, path2restore)
						self._step = self.trainingWeights_op["globalStep"].eval(session=self.session)
						print(">> SESSION RESTORED step[{}].".format(self._step))
				
				print(">> TRAINER is READY.")
			

	def save(self,
			dir_logs= None,
			stepPeriod = 2500): #-- Step period to re-write logs
		""" Method to store on disk network state
		"""
		step_k = self._step // stepPeriod

		with self.graph.as_default():
			#-- Store network state
			if dir_logs:
				path = "{dir}" + os.sep + "checkpoint_{name}_{iter_k}k"
				path = path.format(
					dir     = dir_logs,
					name    = self._name,
					iter_k  = step_k)
				self.saver.save(self.session, path)
				print(">> NETWORK {name} saved at: [{path}]".format(name=self._name, path=path))
					
		return path
	
	
	"""
	*** TRAINING METHODS
	"""
	def train(self,
			tensor_images,      #-- Batch of real images for reconstruction
			tensor_images_disc): #-- Batch of real images for discrimination
		""" Perform a training iteration using an input batch of images
		"""				
		##----------
		## ANNOUNCEMENT!
		self._step += 1
		print(">> MODEL UPDATE (VAEPlus) {}".format(self._step))
		
		##----------
		## TRAINING!
		with self.graph.as_default():
				##--------
				## 1: MITOSIS CLASSIFICATION						
				feed_dict = {}
				feed_dict[self.images_ph]      = tensor_images
				feed_dict[self.images_real_ph] = tensor_images_disc
				
				feed_dict[self.is_training_ph] = True
				
				#-- Embedding noise
				batchSize     = tensor_images.shape[0]
				embeddingSize = self._parameters.embeddingSize
				input_noise = np.stack([np.random.normal(loc=0.0, scale=1.0, size=[embeddingSize]) for _i in range(batchSize)])
				feed_dict[self.noise_ph] = input_noise
				
				#-- No input embedding
				feed_dict[self.embedding_ph] = np.zeros([0,1,1,embeddingSize], dtype=np.float32)
				
				nb_recLosses_GAN = len(self.loss_reconstruction_GAN_list_op)
				
				fetches_training = self.loss_reconstruction_GAN_list_op + [
					self.loss_reconstruction_RGB_op,
					self.loss_kl_op,
					self.loss_discriminator_op,
					self.training_op,
					self.wd_op_all,
					self.movingMoments_op]

				fetches = self.session.run(fetches_training, feed_dict=feed_dict)
				reconstructionLosses_GAN = fetches[:nb_recLosses_GAN]; nbT = nb_recLosses_GAN
				reconstructionLoss_RGB, _KLloss, _GAN_loss, _training, _wd, _mvgMoments = fetches[nbT:]
					
				##
				##-- DISPLAY INFO
				print(">> RGB RECONSTRUCTION LOSS: ", reconstructionLoss_RGB)
				print(">> GAN RECONSTRUCTION LOSS: ", reconstructionLosses_GAN[-1])

				
	def image2embedding(self,
				tensor_images):
		""" Compute embeddings of the input images
		"""
		feed_dict = {}
		feed_dict[self.images_ph]		 = tensor_images
		feed_dict[self.is_training_ph] = False
		
		#-- Empty Sampling Noise
		batchSize     = tensor_images.shape[0]
		embeddingSize = self._parameters.embeddingSize
		input_noise = np.stack([[0.0]*embeddingSize for _i in range(batchSize)])
		feed_dict[self.noise_ph] = input_noise
		
		#-- No input embedding
		feed_dict[self.embedding_ph] = np.zeros([0,1,1,embeddingSize], dtype=np.float32)
		
		#-- No target representation
		feed_dict[self.images_real_ph] = np.zeros([batchSize]+self._parameters.inputSize, dtype=np.float32)
					
		embeddings = self.session.run(
			self.embedding_means,
			feed_dict=feed_dict)
		
		return embeddings
		
	
	def embedding2image(self,
				tensor_embeddings):
		""" Reconstruct images from the input embeddings
		"""
		feed_dict = {}
		#-- Define a "null image" to allow the foward pass through the encoder
		feed_dict[self.images_ph]	   = np.zeros([1]+self._parameters.inputSize, dtype=np.float32) #-- Null image
		feed_dict[self.is_training_ph] = False
		
		#-- Empty Sampling Noise
		batchSize     = 1 #-- 1 null image
		embeddingSize = self._parameters.embeddingSize
		input_noise = np.stack([[0.0]*embeddingSize for _i in range(batchSize)]) #-- Noise for the null image
		feed_dict[self.noise_ph] = input_noise
		
		#-- Input embeddings
		feed_dict[self.embedding_ph] = tensor_embeddings #-- Embeddings of interest
		
		#-- No target representation
		feed_dict[self.images_real_ph] = np.zeros([batchSize]+self._parameters.inputSize, dtype=np.float32)
					
		reconstructions = self.session.run(
			self.reconstructions_op,
			feed_dict=feed_dict)
		
		return reconstructions[1:,] #-- Reconstructions without the null image


	
	def valid(self,tensor_images):
		"""
		User-free method for validation. 
		"""
		return None

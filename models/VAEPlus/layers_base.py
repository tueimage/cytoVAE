# -*- coding: utf-8 -*-
"""
@author: Maxime W. Lafarge, (mlafarge); Eindhoven University of Technology, The Netherlands
@comment: For more details see "Capturing Single-Cell Phenotypic Variation via Unsupervised Representation Learning"; MW Lafarge et al.; MIDL 2019; PMLR 102:315-325

Definition of additional layers.
"""

import tensorflow as tf
import numpy as np
import math

from . import flags
from . import parameters as parameters_d

def variable(name, shape, initializer,
        flags_to_assign = None,
        trainable       = True):
    """ Creates variable given shape and initializer
    """
    kwargs = {"initializer":initializer, "trainable":trainable, "dtype":tf.float32}
    var = tf.get_variable(name, shape, **kwargs) #-- Variable initialization
    for f in flags_to_assign:
        tf.add_to_collection(f, var) #-- Add variable to collection
            
    return var


def xavierInit(n_in, n_out):
    #-- Xavier's initialization formula
    return math.sqrt(2.0/(n_in+n_out))


def batchNormalizeTensor(
        tensor,
        nbUnits,
        is_training,
        
        averaging_axes = [0,1,2],
        averaging_size = None, #-- Key-parameter for BN computation in the Discriminator
        
        moving_average_coeff = parameters_d.bn_mvgAverage_coeff,
        flags_to_assign      = [],
        epsilon = 1e-8):
    """ Batch-Normalization Layer
    """     
    
    #-- 1: Compute the moments of the batch
    with tf.device("/cpu:0"):
        batch_slice = slice(0, averaging_size, 1)
        tensor_reduced = tensor[batch_slice,:]
        batch_mean, batch_var = tf.nn.moments(tensor_reduced, axes=averaging_axes)
        
    
    #-- 2: Initialize non-trainable moving averages    
    current_dict = {}
    current_dict["mean"] = variable(
            name  = "movingBatchMean",
            shape = [1, 1, 1, nbUnits], #-- Explicit shape
            initializer     = tf.constant_initializer(value=0.0),
            flags_to_assign = [flags.MOVING_BATCH_MEAN]+flags_to_assign,
            trainable       = False)
    
    current_dict["var"] = variable(
            name  = "movingBatchVar",
            shape = [1, 1, 1, nbUnits], #-- Explicit shape
            initializer     = tf.constant_initializer(value=1.0),
            flags_to_assign = [flags.MOVING_BATCH_VAR]+flags_to_assign,
            trainable       = False)
        
    #-- 3: Normalize the batch    
    #-- Operations to update the moving averages
    moving_moments_list_op = []  
    moving_moments_list_op.append(
            tf.assign(current_dict["mean"], moving_average_coeff * current_dict["mean"] + (1.0-moving_average_coeff) * batch_mean))
    moving_moments_list_op.append(
            tf.assign(current_dict["var"], moving_average_coeff * current_dict["var"] + (1.0-moving_average_coeff) * batch_var))
    
    
    #-- Different normalization as a function of the training state
    def trainingCase(): return (tensor - batch_mean) / tf.sqrt(batch_var + epsilon)
    def testCase():     return (tensor - current_dict["mean"]) / tf.sqrt(current_dict["var"] + epsilon)
    input_norm = tf.cond(is_training, trainingCase, testCase)      
          
    #-- 4: Learnable rescaling parameters
    current_dict["scale"] = variable(
            name  = "scale",
            shape = [1, 1, 1, nbUnits], #-- Explicit shape
            initializer     = tf.constant_initializer(value=1.0),
            flags_to_assign = [flags.SCALE]+flags_to_assign,
            trainable       = True)            
    
    current_dict["bias"] = variable( #-- Use less since result normalized afterwards
            name  = "bias",
            shape = [1, 1, 1, nbUnits], #-- Explicit shape
            initializer     = tf.constant_initializer(value=0.0),
            flags_to_assign = [flags.BIAS]+flags_to_assign,
            trainable       = True)
    
    tensor_normed_r = current_dict["scale"] * input_norm + current_dict["bias"] #-- Rescaled batch normalized convolution
    return tensor_normed_r, moving_moments_list_op


def biasTensor(
        tensor,
        nbUnits,
        is_training,
        
        averaging_axes = [0,1,2],
        
        moving_average_coeff = parameters_d.bn_mvgAverage_coeff,
        flags_to_assign      = [],
        epsilon = 1e-8):
    """
        Wrapper to perform bias addition to the input tensor
        The goal of the wrapper is to keep track of the moments of the wrapped tensor for monitoring purpose
    """          
    #-- 1: Learnable bias parameter    
    bias = variable( #-- Use less since result normalized afterwards
            name="bias",
            shape=[1, 1, 1, nbUnits],
            initializer=tf.constant_initializer(value=0.01),
            flags_to_assign=[flags.BIAS] + flags_to_assign,
            trainable=True)
    
    tensor_biased = tensor + bias #-- Rescaled batch normalized convolution
    return tensor_biased
    
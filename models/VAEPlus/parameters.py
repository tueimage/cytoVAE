# -*- coding: utf-8 -*-
"""
@author: Maxime W. Lafarge, (mlafarge); Eindhoven University of Technology, The Netherlands
@comment: For more details see "Capturing Single-Cell Phenotypic Variation via Unsupervised Representation Learning"; MW Lafarge et al.; MIDL 2019; PMLR 102:315-325

Parameterization of the model
"""
from . import flags

#-- Tensorflow configuration
device = "/gpu:0"

#-- Training data information
batchSize = 104
inputSize = [68, 68, 3] #-- Dimension of the training samples
embeddingSize = 256 #-- Default embedding size


class Network_p:
    """ abstract network parameterization
    """
    def __init__(self, name="Abstract_Network", nbLayers=1):
        self.name     = name
        self.nbLayers = nbLayers
        
        #-- Layer quantities
        self.nbUnits_input  = [0]*nbLayers
        self.nbUnits_output = [0]*nbLayers
        self.kernel_sizes   = [0]*nbLayers
        
        #-- Layer specificites
        self.paddings    = [None]*nbLayers  #-- Perform padding prior to convolution
        self.embedding   = [False]*nbLayers #-- Flag for embbedding layer
        self.upSampling  = [False]*nbLayers
        self.maxPoolings = [False]*nbLayers
        self.relus       = [False]*nbLayers
        self.batchNorms  = [False]*nbLayers
                
        return None
    
    def __enter__(self):
        return self
    
    def __exit__(self, _type, _value, _traceback):
        pass


nbUnits = 32
#--
#-- 5-layer ENCODER
with Network_p("Encoder", nbLayers=4) as ntwrk:
    encoder_network = ntwrk
    
    #-- Layer 0 [hw_in=68, hw_out=64>32]: INPUT LAYER
    layer_idx = 0
    ntwrk.nbUnits_input[layer_idx]  = inputSize[2]
    ntwrk.nbUnits_output[layer_idx] = nbUnits
    ntwrk.kernel_sizes[layer_idx]   = 5
    
    ntwrk.maxPoolings[layer_idx]    = True
    ntwrk.relus[layer_idx]          = True
    ntwrk.batchNorms[layer_idx]     = True
    
    #-- Layer 1 [hw_in=32, hw_out=28>14]
    layer_idx += 1
    ntwrk.nbUnits_input[layer_idx]  = ntwrk.nbUnits_output[layer_idx-1]
    ntwrk.nbUnits_output[layer_idx] = nbUnits
    ntwrk.kernel_sizes[layer_idx]   = 5
    
    ntwrk.maxPoolings[layer_idx]    = True
    ntwrk.relus[layer_idx]          = True
    ntwrk.batchNorms[layer_idx]     = True

    #-- Layer 2 [hw_in=14, hw_out=10>5]
    layer_idx += 1
    ntwrk.nbUnits_input[layer_idx]  = ntwrk.nbUnits_output[layer_idx-1]
    ntwrk.nbUnits_output[layer_idx] = nbUnits
    ntwrk.kernel_sizes[layer_idx]   = 5
    
    ntwrk.maxPoolings[layer_idx]    = True
    ntwrk.relus[layer_idx]          = True
    ntwrk.batchNorms[layer_idx]     = True
    

    #-- Layer 3 [hw_in=5, hw_out=1]: EMBEDDING
    layer_idx += 1
    ntwrk.embedding[layer_idx]      = True #-- Embedding layer
    
    ntwrk.nbUnits_input[layer_idx]  = ntwrk.nbUnits_output[layer_idx-1]
    ntwrk.nbUnits_output[layer_idx] = 2 * embeddingSize #-- (mean, variance)
    ntwrk.kernel_sizes[layer_idx]   = 5
    
    ntwrk.maxPoolings[layer_idx]    = False #-- No pooling
    ntwrk.relus[layer_idx]          = False #-- No ReLU for sampling parameterization
    ntwrk.batchNorms[layer_idx]     = True #-- Re-center embeddings


#--4-layer DECODER
with Network_p("Decoder", nbLayers=4) as ntwrk:
    decoder_network = ntwrk 
    
    #-- Layer 3+1 [hw_in=1, hw_out=5<10]
    layer_idx = 0
    ntwrk.nbUnits_input[layer_idx]  = embeddingSize
    ntwrk.nbUnits_output[layer_idx] = nbUnits
    ntwrk.kernel_sizes[layer_idx]   = 5
    
    ntwrk.paddings[layer_idx]       = 4 #-- Padding before transposed convolution
    ntwrk.upSampling[layer_idx]     = 10
    ntwrk.relus[layer_idx]          = True
    ntwrk.batchNorms[layer_idx]     = True
  
    #-- Layer 3+2 [hw_in=10<10+8, hw_out=14<28]
    layer_idx += 1
    ntwrk.nbUnits_input[layer_idx]  = ntwrk.nbUnits_output[layer_idx-1]
    ntwrk.nbUnits_output[layer_idx] = nbUnits
    ntwrk.kernel_sizes[layer_idx]   = 5
    
    ntwrk.paddings[layer_idx]       = 4 #-- Padding before transposed convolution
    ntwrk.upSampling[layer_idx]     = 28
    ntwrk.relus[layer_idx]          = True
    ntwrk.batchNorms[layer_idx]     = True
    
    #-- Layer 3+3 [hw_in=28<28+8, hw_out=32<64]
    layer_idx += 1
    ntwrk.nbUnits_input[layer_idx]  = ntwrk.nbUnits_output[layer_idx-1]
    ntwrk.nbUnits_output[layer_idx] = nbUnits
    ntwrk.kernel_sizes[layer_idx]   = 5
    
    ntwrk.paddings[layer_idx]       = 4 #-- Padding before transposed convolution
    ntwrk.upSampling[layer_idx]     = 64
    ntwrk.relus[layer_idx]          = True
    ntwrk.batchNorms[layer_idx]     = True
    
    #-- Layer 3+4 [hw_in=64, hw_out=68]: FINAL ACTIVATION
    layer_idx += 1
    ntwrk.nbUnits_input[layer_idx]  = ntwrk.nbUnits_output[layer_idx-1]
    ntwrk.nbUnits_output[layer_idx] = nbUnits
    ntwrk.kernel_sizes[layer_idx]   = 5
    
    ntwrk.paddings[layer_idx]       = 4 #-- Padding before transposed convolution
    ntwrk.relus[layer_idx]          = True
    ntwrk.batchNorms[layer_idx]     = True


#-- 1-layer RECONSTRUCTOR
with Network_p("Reconstructor", nbLayers=1) as ntwrk:
    reconstructor_network = ntwrk 
    #-- Layer 3+N [hw_in=N, hw_out=N]: OUTPUT ON-PLACE REARRANGING
    layer_idx = 0
    ntwrk.nbUnits_input[layer_idx]  = nbUnits
    ntwrk.nbUnits_output[layer_idx] = 3 #-- recover input-like shape
    ntwrk.kernel_sizes[layer_idx]   = 1 #-- ON-PLACE 
    
    ntwrk.relus[layer_idx]          = True
    ntwrk.batchNorms[layer_idx]     = True #-- Output batch norm


#-- 5-layer DISCRIMINATOR
with Network_p("Discriminator", nbLayers=5) as ntwrk:
    discriminator_network = ntwrk
    
    #-- Layer 0 [hw_in=68, hw_out=64>32]: INPUT LAYER
    layer_idx = 0
    ntwrk.nbUnits_input[layer_idx]  = inputSize[2]
    ntwrk.nbUnits_output[layer_idx] = nbUnits
    ntwrk.kernel_sizes[layer_idx]   = 5
    
    ntwrk.maxPoolings[layer_idx]    = True
    ntwrk.relus[layer_idx]          = True
    ntwrk.batchNorms[layer_idx]     = True
    
    #-- Layer 1 [hw_in=32, hw_out=28>14]
    layer_idx += 1
    ntwrk.nbUnits_input[layer_idx]  = ntwrk.nbUnits_output[layer_idx-1]
    ntwrk.nbUnits_output[layer_idx] = nbUnits
    ntwrk.kernel_sizes[layer_idx]   = 5
    
    ntwrk.maxPoolings[layer_idx]    = True
    ntwrk.relus[layer_idx]          = True
    ntwrk.batchNorms[layer_idx]     = True

    #-- Layer 2 [hw_in=14, hw_out=10>5]
    layer_idx += 1
    ntwrk.nbUnits_input[layer_idx]  = ntwrk.nbUnits_output[layer_idx-1]
    ntwrk.nbUnits_output[layer_idx] = nbUnits
    ntwrk.kernel_sizes[layer_idx]   = 5
    
    ntwrk.maxPoolings[layer_idx]    = True
    ntwrk.relus[layer_idx]          = True
    ntwrk.batchNorms[layer_idx]     = True
    

    #-- Layer 3 [hw_in=5, hw_out=1]: FC + projection
    layer_idx += 1
    ntwrk.nbUnits_input[layer_idx]  = ntwrk.nbUnits_output[layer_idx-1]
    ntwrk.nbUnits_output[layer_idx] = 64
    ntwrk.kernel_sizes[layer_idx]   = 5
    
    ntwrk.relus[layer_idx]          = True
    ntwrk.batchNorms[layer_idx]     = True

    #-- Layer 4 [hw_in=1, hw_out=1]: FC + projection
    layer_idx += 1
    ntwrk.nbUnits_input[layer_idx]  = ntwrk.nbUnits_output[layer_idx-1]
    ntwrk.nbUnits_output[layer_idx] = 1 #-- Sigmoid logit
    ntwrk.kernel_sizes[layer_idx]   = 1
    
    ntwrk.relus[layer_idx]          = False #-- Logits
    ntwrk.batchNorms[layer_idx]     = True #-- Re-center logits


#-- Other training parameters
beta = 2.0
adv_schedule_slope  = 2500.0 
bn_mvgAverage_coeff = 0.9

sgd_learningRate  = 0.01
adam_learningRate = 0.001
adam_epsilon      = 1e-7
momentum          = 0.9

weightDecay = 1e-4



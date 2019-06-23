#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  9 09:50:55 2019

@author: ahmed
"""

from keras.models import load_model
import matplotlib.pyplot as plt
import os
import numpy as np

fl = []
dr  = 'conv/models/'
dri = 'conv/stats/'
n_bins = 100

kwargs = dict(histtype='stepfilled', alpha=0.3, density=True , bins=40)


for file in os.listdir( dr ):
    if file.endswith(".h5"):
        fl.append( file )

for name in fl  :
	
	model = load_model( dr + name )

	nbr_layers =  len( model.layers )   #total number of layers
	nbr_conv = 0  # number of convolution layers

	la = []  # list of layers
	for i in range( nbr_layers ) :
		if( "conv" in model.layers[i].name  ):
			la.append( model.layers[i] )	
			nbr_conv = nbr_conv + 1
 
	weights = [] #get weights only , no biases
	for i in range( nbr_conv ) :
		weights.append( la[i].get_weights()[0] )	

	fig, axs = plt.subplots(nrows=1, ncols=1, sharex=True )

	for i in range( nbr_conv  ) :
		axs.hist( weights[i].flatten(),  **kwargs, label=la[i].name )
		axs.set_xticks( np.arange(-1, 1, step=0.2) )
		axs.set_title( name )
		axs.legend(  )
	
	#break;	
	#plt.show()
	plt.savefig( dri + name + '.png')

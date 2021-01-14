import cv2 
import os 
import matplotlib.pyplot as plt 
import tensorflow as tf 
from tensorflow.keras.layers import Conv2D, Dense, Flatten, BatchNormalization, Conv2DTranspose, MaxPool2D, LeakyReLU, Activation, Reshape, Input
from tensorflow.keras.models import Model 
from tensorflow.keras import backend as K 
import numpy as np 

class Autoencoder:

    @staticmethod
    def build(width, height, depth, filters=(32,64), latentDim=16):
        inputShape = (height, width, depth)
        chanDim = -1
        
        inputs = Input(shape=inputShape)
        x = inputs 

        for f in filters:
            x = Conv2D(f, (3,3), strides=2, padding='same', activation='relu')(x)
            x = Conv2D(f, (3,3), strides=2, padding='same', activation='relu')(x) 
            #x = MaxPool2D((2,2), padding='same')(x)
            x = BatchNormalization(axis=chanDim)(x)
        
        volumeSize = K.int_shape(x)
        x = Flatten()(x)
        latent = Dense(latentDim)(x)

        encoder = Model(inputs, latent, name='encoder')

        latentInputs = Input(shape=(latentDim,))
        x = Dense(np.prod(volumeSize[1:]))(latentInputs)
        x = Reshape((volumeSize[1], volumeSize[2], volumeSize[3]))(x)

        for f in filters[::-1]:
            x = Conv2DTranspose(f, (3,3), strides=2, padding='same', activation='relu')(x)
            x = Conv2DTranspose(f, (3,3), strides=2, padding='same', activation='relu')(x)
            x = BatchNormalization(axis=chanDim)(x)
            #x = Conv2DTranspose(f, (3,3), strides=2, padding='same', activation='relu')(x)
        x = Conv2DTranspose(3, (2,2), strides=1, padding='same', activation='relu')(x)
        outputs = Activation('sigmoid')(x)

        decoder = Model(latentInputs, outputs, name='decoder')

        autoencoder = Model(inputs, decoder(encoder(inputs)), name='autoencoder')
        print(encoder.summary())
        print(decoder.summary())

        return (encoder, decoder, autoencoder)
    

from __future__ import generators
import matplotlib.pyplot as plt
import numpy as np 

from tensorflow.keras.datasets import mnist 
from tensorflow.keras.layers import Dense, Flatten, Reshape
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

# 모델 입력 차원 

img_rows = 28
img_cols = 28
channels = 1 

img_shape = (img_rows, img_cols, channels)

z_dim = 100 

# 생성자 구현 

def build_generator(img_shape, z_dim):
    model = Sequential()
    model.add(Dense(128, input_dim=z_dim))
    model.add(LeakyReLU(alpha=0.01))
    model.add(Dense(28 * 28 * 1, activation='tanh'))
    model.add(Reshape(img_shape))
    return model

# 판별자 구현 

def build_discriminator(img_shape):
    model = Sequential()
    model.add(Flatten(input_shape=img_shape))
    model.add(Dense(128))
    model.add(LeakyReLU(alpha==0.01))
    model.add(Dense(1, activation='sigmoid'))
    return model
# 모델 생성

def build_gan(generator, discriminator):
    
    model = Sequential()
    
    model.add(generator)
    model.add(discriminator)

    return model

discriminator = build_discriminator(img_shape)
discriminator.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['acuraccy'])

generator = build_generator(img_shape, z_dim)
discriminator.trainable = False

gan = build_gan(generator, discriminator)
gan.compile(loss='binary_crossentropy')
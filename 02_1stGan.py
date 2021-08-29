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
    model.add(LeakyReLU(alpha=0.01))
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

# GAN 훈련 반복

losses = []
accuracies = []
iteration_checkpoints = []

def train(iterations, batch_size, sample_interval):
    (X_train, _), (_, _) = mnist.load_data()
    
    X_train = X_train / 127.5 - 1.0 # 스케일 조정 
    X_train = np.expand_dims(X_train, axis=3)

    real = np.ones((batch_size, 1)) # 진짜 이미지 레이블 : 모두 1
    fake = np.zeros((batch_size, 1)) # 가짜 이미지 레이블 : 모두 0 

    for iteration in range(iterations):
        idx = np.random.randint(0, X_train.shape[0], batch_size) # 진짜 이미지에서 랜덤 배치 가져오기 
        imgs = X_train[idx]
        

        z = np.random.normal(0, 1, (batch_size, 100)) # 가짜 이미지 배치 생성 
        d_loss_real = discriminator.train_on_batch(imgs, real)
        d_loss_fake = discriminator.train_on_batch(gen_imgs, fake)
        d_loss, accuracy = 0.5 * np.add(d_loss_real, d_loss_fake)

        z = np.random.normal(0, 1, (batch_size, 100))
        gen_imgs = generator.predict(z)
        
        g_loss = gan.train_on_batch(z, real) # 생성자 훈련 
        
        if (iteration + 1) % sample_interval == 0:
            losses.append((d_loss, g_loss))
            accuracies.append(100.0 * accuracy)
            iteration_checkpoints.append(iteration + 1)
            
            print("%d [ 손실 : %f, 정확도 : %.2f%%] [G 손실 : %f]"%
                    (iteration + 1, d_loss, 100.0 * accuracy, g_loss))

            sample_images(generator)

        
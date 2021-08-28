from json import decoder, encoder
from tensorflow.keras.layers import Input, Dense, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras import metrics
from tensorflow.keras.datasets import mnist
from tensorflow.python.keras.backend_config import epsilon
import numpy as np

# hyperparameter
batch_size = 100
original_dim = 784 # mnist 이미지의 높이*너비
latent_dim = 2
intermediate_dim = 256
epochs = 50
epsilon_std = 1.0

# encoder
x = Input(shape=(original_dim,), name="input") # 인코더 입력 
h = Dense(intermediate_dim, activation='relu', name="encoding")(x) # 중간층
z_mean = Dense(latent_dim, name="mean")(h) # 잠재 공간의 로그 분산을 정의
z_log_var = Dense(latent_dim, name="log-variance") # 잠재 공간의 로그 분산을 정의
z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var]) # 텐서플로 백엔드를 사용할 때는 ouuput_shape이 꼭 필요한 것은 아니다.
encoder = Model(x, [z_mean, z_log_var, z], name="encoder")

# sampling 
def sampling(*args: tuple):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape = (K.shape(z_mean)[0], latent_dim), mean=0. , stddev=epsilon_std)
    return z_mean + K.exp(z_log_var / 2 ) + epsilon

input_decoder = Input(shape=(latent_dim,), name="decoder_input") # 디코더 입력 
decoder_h = Dense(intermediate_dim, activation='relu', name="decoder_h")(input_decoder) # 잠재 공간을 중간층의 차원으로 변환합니다.
x_decoded = Dense(original_dim, activation='sigmoid', name="flat_decoded")(decoder_h) # 원본 차원으로 변환한다.
decoder = Model(input_decoder, x_decoded, name="decoder") # 케라스 모델로 디코더를 정의한다. 

# model combine 

output_combined = decoder(encoder(x)[2]) # 인코더 출력을 디코더에 사용한다. 인코더의 3번째 반환 값이 z이다
vae = Model(x, output_combined) # 입력과 출력을 연결한다.
vae.summary()

# def loss fucntion
kl_loss = -0.5 * K.sum(1 + z_log_var - K.exp(z_log_var)- K.square(z_mean), axis=-1)

vae.add_loss(K.mean(kl_loss / 794.))
vae.compile(optimizer='rmsprop', loss='binary_crossentropy') # 마지막으로 모델을 컴파일한다. 

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype('float32') /255.
x_test = x_test.astype('float32') /255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.hape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.hape[1:])))

vae.fit(x_train, x_train,shuffle=True, epochs=epochs, batch_size=batch_size)
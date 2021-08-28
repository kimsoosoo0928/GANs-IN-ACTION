from json import encoder
from tensorflow.keras.layers import Input, Dense, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras import metrics
from tensorflow.keras.datasets import mnist
from tensorflow.python.keras.backend_config import epsilon

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

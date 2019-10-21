from keras.layers import Dense
from keras.models import Sequential

model = Sequential()
model.add(Dense(4, input_dim=3, kernel_initializer="random_uniform"))

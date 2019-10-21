from __future__ import print_function
import numpy as np
from keras import regularizers
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD, Adam
from keras.utils import np_utils
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

np.random.seed(1671)
NB_EPOCH = 20
BATCH_SIZE = 128
VERBOSE = 1
NB_CLASSES = 10
OPTIMIZER = Adam(lr=0.001)
N_HIDDEN = 128
VALIDATION_SPLIT = 0.2
DROPOUT = 0.3

# 加载数据
(X_train, y_train), (X_test, y_test) = mnist.load_data("../data/mnist.npz")
# print(1111)
# print(X_train)
# print(y_train)
RESHAPED = 784
X_train = X_train.reshape(60000, RESHAPED)
X_test = X_test.reshape(10000, RESHAPED)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train /= 255
X_test /= 255
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')
Y_train = np_utils.to_categorical(y_train, NB_CLASSES)
Y_test = np_utils.to_categorical(y_test, NB_CLASSES)
# 构建模型
model = Sequential()
model.add(Dense(N_HIDDEN, input_shape=(RESHAPED,), activation='relu', kernel_regularizer=regularizers.l2(0.01)))
model.add(Dropout(DROPOUT))
model.add(Dense(N_HIDDEN, activation='relu'))
model.add(Dropout(DROPOUT))
model.add(Dense(NB_CLASSES, activation='softmax'))
model.summary()
model.compile(loss='categorical_crossentropy', optimizer=OPTIMIZER, metrics=['accuracy'])
# 训练模型
history = model.fit(X_train, Y_train, batch_size=BATCH_SIZE, epochs=NB_EPOCH, verbose=VERBOSE,
                    validation_split=VALIDATION_SPLIT)
# 评价模型
score = model.evaluate(X_test, Y_test, verbose=VERBOSE)
print("test score:", score[0])
print("test accuracy:", score[1])
# 预测
pred_c = model.predict_classes(X_test[0:1])
pred_p = model.predict_proba(X_test[0:1])
print("predict result:")
print("to predict:")
print(X_test[0:1])
print("class:")
print(pred_c)
print("p:")
print(pred_p)

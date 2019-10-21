import keras
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
# 这是Sequential模型：
model = Sequential()

# 堆叠层非常简单.add()：
model.add(Dense(units=64, activation='relu', input_dim=100))
model.add(Dense(units=10, activation='softmax'))

# 一旦您的模型看起来不错，请配置其学习过程.compile()：
model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])

# 如果需要，您可以进一步配置优化程序。Keras的核心原则是使事情变得相当简单，同时允许用户在需要时完全控制（最终控制是源代码的易扩展性）。
# model.compile(loss=keras.losses.categorical_crossentropy,
#               optimizer=keras.optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True))

x_train = np.random.random((100, 100))
y_train = np.random.random((100, 10))
# 您现在可以批量迭代您的训练数据：
# x_train and y_train are Numpy arrays --just like in the Scikit-Learn API.
model.fit(x_train, y_train, epochs=5, batch_size=32)
# # 或者，您可以手动将批次提供给模型：
#
# model.train_on_batch(x_batch, y_batch)
# # 在一行中评估您的表现：
print("--------------eval-----------------")
x_test = np.random.random((100, 100))
y_test = np.random.random((100, 10))

loss_and_metrics = model.evaluate(x_test, y_test, batch_size=128)
# # 或者生成对新数据的预测：

print("--------------predict-----------------")
classes = model.predict(x_test, batch_size=128)
print(classes)

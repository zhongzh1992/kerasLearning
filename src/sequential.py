import warnings
from keras.models import Sequential
from keras.layers import Dense, Activation

warnings.filterwarnings("ignore")

# 方式1
model = Sequential([
    Dense(32, input_shape=(784,)),
    Activation('relu'),
    Dense(10),
    Activation('softmax'),
])

# 方式2
model = Sequential()
# batch_size = 32
model.add(Dense(32, input_dim=784))
model.add(Activation('relu'))
model.add(Dense(10))
model.add(Activation('softmax'))

# 编译模型
model.compile(optimizer='rmsprop',  # 优化器
              loss='categorical_crossentropy',  # 交叉熵损失函数
              metrics=['accuracy'])  # 准确率

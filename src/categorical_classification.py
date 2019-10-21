from keras.models import Sequential
from keras.layers import Dense, Activation
import numpy as np
import keras

# For a single-input model with 10 classes (categorical classification):
model = Sequential()
model.add(Dense(32, activation='relu', input_dim=100))
model.add(Dense(10, activation='softmax'))
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Generate dummy data
data = np.random.random((1000, 100))
labels = np.random.randint(10, size=(1000, 1))
print("labels")
print(labels)

# Convert labels to categorical one-hot encoding
one_hot_labels = keras.utils.to_categorical(labels, num_classes=10)
print("one-hot")
print(one_hot_labels)

# Train the model, iterating on the data in batches of 32 samples
model.fit(data, one_hot_labels, epochs=10, batch_size=32)

# importing the libraries
import os
import random
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from keras import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten

from keras.datasets import fashion_mnist
from keras.utils import to_categorical

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

SEED_VALUE = 42

# Fix seed to make training deterministic.
random.seed(SEED_VALUE)
np.random.seed(SEED_VALUE)
tf.random.set_seed(SEED_VALUE)

# loading the dataset
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()


print(X_train.shape)
print(X_test.shape)

# displaying sample images from the dataset
plt.figure(figsize=(18, 9))

num_rows = 4
num_cols = 8

# plot each of the images in the batch and the associated ground truth labels.
for i in range(num_rows * num_cols):
    ax = plt.subplot(num_rows, num_cols, i + 1)
    plt.imshow(X_train[i, :, :])
    plt.axis("off")
plt.show()

# Normalize images to the range [0, 1].
X_train = X_train.astype("float32") / 255
X_test = X_test.astype("float32") / 255

# Change the labels from integer to categorical data.
print('Original (integer) label for the first training sample: ', y_train[0])

# Convert labels to one-hot encoding.
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print('After conversion to categorical one-hot encoded labels: ', y_train[0])


# defining a function to work on the cnn model
def cnn_model(input_shape=(28, 28, 1)):
    model = Sequential()

    # ------------------------------------
    # Conv Block 1: 32 Filters, MaxPool.
    # ------------------------------------
    model.add(Conv2D(filters=32, kernel_size=3, padding='same', activation='relu', input_shape=input_shape))
    model.add(Conv2D(filters=32, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))

    # ------------------------------------
    # Conv Block 2: 64 Filters, MaxPool.
    # ------------------------------------
    model.add(Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'))
    model.add(Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))

    # ------------------------------------
    # Conv Block 3: 64 Filters, MaxPool.
    # ------------------------------------
    model.add(Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'))
    model.add(Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))

    # ------------------------------------
    # Flatten the convolutional features.
    # ------------------------------------
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(10, activation='softmax'))

    return model

# Create the model.
model = cnn_model()
model.summary()

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'],
             )

#training the model
history = model.fit(X_train,
                    y_train,
                    batch_size=32,
                    epochs=10,
                    verbose=1,
                    validation_split=.2,
                   )
# Evaluate the model
loss,accuracy= model.evaluate(X_test, y_test)

# Retrieve training results.
epochs = range(1, len(history.history['loss']) + 1)
train_loss = history.history["loss"]
train_acc  = history.history["accuracy"]
val_loss = history.history["val_loss"]
val_acc  = history.history["val_accuracy"]

# Plot Loss
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs, train_loss, label='Training Loss', marker='o')
plt.plot(epochs, val_loss, label='Validation Loss', marker='s')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss vs Epoch')
plt.legend()
plt.grid(True)

# Plot Accuracy
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs, train_acc, label='Training Accuracy', marker='o')
plt.plot(epochs, val_acc, label='Validation Accuracy', marker='s')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Epoch')
plt.legend()
plt.grid(True)


plt.tight_layout()
plt.show()

print(f'Model Accuracy: {accuracy}')
print(f'Model Loss: {loss}')

# Predict data for first 5
y_pred = model.predict(X_test)
print(f'Predictions: {y_pred[:5]}')

# Using the save() method, the model will be saved to the file system in the 'SavedModel' format.
model.save('image_classification_model')
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist


def get_mnist_model():
    """

    :return:
    """
    inputs = keras.Input(shape=(28 * 28))
    features = layers.Dense(512, activation='relu')(inputs)
    features = layers.Dropout(0.5)(features)
    outputs = layers.Dense(10, activation='softmax')(features)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


(images, labels), (test_images, test_labels) = mnist.load_data()
images = images.reshape(60000, 28 * 28).astype('float32') / 255
test_images = test_images.reshape(10000, 28 * 28).astype('float32') / 255
train_images, val_images = images[10000:], images[:10000]
train_labels, val_labels = labels[10000:], labels[:10000]

model = get_mnist_model()
model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
tensorboard = keras.callbacks.TensorBoard(
    log_dir='/home/karvsmech/PycharmProjects/Tensorflow_projects'
)
model.fit(train_images, train_labels,
          epochs=10,
          validation_data=(val_images, val_labels), callbacks=[tensorboard])

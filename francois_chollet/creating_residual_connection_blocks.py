import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from keras.datasets import mnist

"""
# To implement the following with mnist dataset image classification
1) Adding dropout layers
2) Implementing residual connects
3) Implementing batch nomralization
4) Implementing data augmentation
"""

(train_val_images, train_val_labels), (test_images, test_labels) = mnist.load_data()
train_val_images = train_val_images.reshape((len(train_val_images), 28, 28, 1))
train_val_images = train_val_images.astype('float32') / 255
test_images = test_images.reshape(len(test_images), 28, 28, 1)
test_images = test_images.astype('float32') / 255

# creating train and validation set by splitting the train_data

# shuffling the data just to ensure complete randomness
train_val_images = tf.random.shuffle(train_val_images, seed=42)
train_val_labels = tf.random.shuffle(train_val_labels, seed=42)

# splitting the train data to create a validation set.
validation_data = train_val_images[:10000]
train_data = train_val_images[10000:]
validation_labels = train_val_labels[:10000]
train_labels = train_val_labels[10000:]


def residual_block(inputs, filters, is_pooling=False):
    """

    :param inputs:
    :param is_pooling:
    :return:
    """
    residual = inputs
    x = layers.Conv2D(filters=filters, kernel_size=3, activation='relu', padding='same')(inputs)
    x = layers.Conv2D(filters=filters, kernel_size=3, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    if is_pooling:
        x = layers.MaxPool2D(pool_size=2, padding='same')(x)
        # we use strides in a 1x1 linear dimensional Conv2D if the spatial height and width are different.
        # and the filters to change the channel depth. no activation function is required.
        residual = layers.Conv2D(filters=filters, kernel_size=1, strides=2)(residual)

    if residual.shape[-1] != filters:
        residual = layers.Conv2D(filters=filters, kernel_size=1)(residual)
    x = layers.add([x, residual])
    return x


# convolutional neural networks take tensor of rank 3. (spatial height, spatial width, and channel depth)
# using functional api
inputs = keras.Input(shape=(28, 28, 1))
x = residual_block(inputs, 32, is_pooling=True)
x = residual_block(x, 64, is_pooling=True)
x = residual_block(x, 128, is_pooling=True)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.5)(x)
x = layers.Flatten()(x)
outputs = layers.Dense(10, activation='softmax')(x)
model = keras.Model(inputs=inputs, outputs=outputs)

model.summary()

model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# defining callbacks
# using tensorboard in the callbacks
callbacks_list = [
    keras.callbacks.ModelCheckpoint(filepath='./model_checkpoints/mnist_with_just_2_block_dropout.keras',
                                    save_best_only=True,
                                    monitor='val_loss'),
    keras.callbacks.TensorBoard(log_dir='./tensor_board/mnist_just_2_block_dropout')
]

history = model.fit(x=train_data,
                    y=train_labels,
                    batch_size=512,
                    epochs=10,
                    callbacks = callbacks_list,
                    validation_data=(validation_data, validation_labels)
                    )

train_accuracy = history.history['accuracy']
train_loss = history.history['loss']
validation_accuracy = history.history['val_accuracy']
validation_loss = history.history['val_loss']
epochs = list(range(1, len(train_accuracy)+1))

# plotting the train accuracy
figs, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))
ax1.plot(epochs, train_loss, 'bo', label='training_loss')
ax1.plot(epochs, validation_loss, 'b', label='validation_loss')
ax1.set_title('training loss vs. validation loss')
ax1.grid(True)
ax1.legend()
ax2.plot(epochs, train_accuracy, 'bo', label='training_accuracy')
ax2.plot(epochs, validation_accuracy, 'b', label='validation_accuracy')
ax2.set_title('training accuracy vs. validation accuracy')
ax2.grid(True)
ax2.legend()
plt.savefig('./plots/mnist_with_just_2_block_dropout.png')

model = keras.models.load_model('./model_checkpoints/mnist_with_just_2_block_dropout.keras')

# evaluating the model on the test set.
model.evaluate(x=test_images, y=test_labels)


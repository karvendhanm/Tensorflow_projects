import matplotlib.pyplot as plt
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
model.summary()

callbacks_list = [
    keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=2),
    keras.callbacks.ModelCheckpoint(filepath='checkpoint_path.keras', monitor='val_loss', save_best_only=True)
]

# model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# model.fit(x=train_images, y=train_labels, epochs=20,
#           callbacks=callbacks_list,
#           validation_data=(val_images, val_labels))

# saving the model manually
model.save('my_checkpoint_path')

# loading the model
model = keras.models.load_model('checkpoint_path.keras')

# writing your own callbacks:
class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs):
        self.per_batch_losses = []

    def on_batch_end(self,  batch, logs):
        self.per_batch_losses.append(logs.get('loss'))

    def on_epoch_end(self, epoch, logs):
        plt.clf()
        plt.plot(range(len(self.per_batch_losses)), self.per_batch_losses, label='training loss for each batch')
        plt.xlabel(f'Batch (epoch: {epoch})')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(f'plot at epoch_{epoch}')
        self.per_batch_losses = []

model = get_mnist_model()
model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x=train_images, y=train_labels, epochs=10,
          callbacks=[LossHistory()],
          validation_data=(val_images, val_labels))





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

model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x=train_images, y=train_labels, epochs=3, validation_data=(val_images, val_labels))
test_metrics = model.evaluate(x=test_images, y=test_labels)
predictions = model.predict(test_images)

# 7_3_1 writing your own metrics
class RootMeanSquaredError(keras.metrics.Metric):

    def __init__(self, name='rmse', **kwargs):
        super().__init__(name=name, **kwargs)
        self.mse_sum = self.add_weight(name='mse_sum', initializer='zeros')
        self.total_samples = self.add_weight(name='total_samples', initializer='zeros', dtype='int32')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.one_hot(y_true, depth=tf.shape(y_pred)[1])
        mse = tf.reduce_sum(tf.square(y_true - y_pred))
        self.mse_sum.assign_add(mse)
        num_samples = tf.shape(y_pred)[0]
        self.total_samples.assign_add(num_samples)

    def result(self):
        return tf.sqrt(self.mse_sum/tf.cast(self.total_samples, tf.float32))

    def reset_state(self):
        self.mse_sum.assign(0.)
        self.total_samples.assign(0)


model = get_mnist_model()
model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['accuracy', RootMeanSquaredError()])
model.fit(x=train_images, y=train_labels, epochs=3, validation_data=(val_images, val_labels))
test_metrics = model.evaluate(x=test_images, y=test_labels)
predictions = model.predict(test_images)


























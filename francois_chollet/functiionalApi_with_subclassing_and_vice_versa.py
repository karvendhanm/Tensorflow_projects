import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

input_ = keras.Input(shape=(3,))
features = layers.Dense(64, activation='relu')(input_)
output = layers.Dense(10, activation='softmax')(features)
model = keras.Model(inputs=[input_], outputs=[output])

model.summary()
keras.utils.plot_model(model, 'graphical_representation.png', show_shapes=True)


class Classifier(keras.Model):
    def __init__(self, num_classes):
        """

        :param num_classes:
        """
        super().__init__()
        if num_classes == 2:
            num_units = 1
            activation = 'sigmoid'
        else:
            num_units = num_classes
            activation = 'softmax'
        self.dense = layers.Dense(num_units, activation=activation)

    def call(self, inputs):
        """

        :param inputs:
        :return:
        """
        return self.dense(inputs)


input_ = keras.Input(shape=(3,))
features = layers.Dense(64, activation='relu')(input_)
outputs = Classifier(num_classes=10)(features)
model = keras.Model(inputs=input_, outputs=outputs)

model.summary()
keras.utils.plot_model(model, 'graphical_representation_model.png', show_shapes=True)


inputs = keras.Input(shape=(64,))
outputs = layers.Dense(1, activation='sigmoid')(inputs)
binary_classifier = keras.Model(inputs=inputs, outputs=outputs)

class MyModel(keras.Model):

    def __init__(self, num_classes=2):
        super().__init__()
        self.dense = layers.Dense(64, activation='relu')
        self.classifier = binary_classifier

    def call(self, inputs):
        features = self.dense(inputs)
        return self.classifier(features)

model = MyModel()








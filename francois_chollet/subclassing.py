import numpy as np
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers


# subclassing a layer
class DenseLayer(keras.layers.Layer):
    def __init__(self, units, activation=None):
        """

        :param units:
        :param activation:
        """
        super().__init__()
        self.W = None
        self.b = None
        self.units = units
        self.activation = activation

    def build(self, input_shape):
        """

        :param input_dim:
        :return:
        """
        input_dim = input_shape[-1]
        self.W = self.add_weight(shape=(input_dim, self.units), initializer='random_normal')
        self.b = self.add_weight(shape=(self.units,), initializer='zeros')

    def call(self, inputs):
        """

        :param inputs:
        :return:
        """
        y = tf.matmul(inputs, self.W) + self.b
        if self.activation is not None:
            y = self.activation(y)
        return y

    def __call__(self, inputs):
        if not self.built:
            self.build(inputs.shape)
            self.built = True
        return self.call(inputs)


denselayer = DenseLayer(32, tf.nn.relu)
input_tensor = tf.ones(shape=(2, 784))
output_tensor = denselayer(input_tensor)
print(output_tensor.shape)

# subclassing the model class:
# model has 3 inputs and 2 outputs

# inputs
# 1) title of the ticket (text input)
# 2) the text body of the ticket (text input)
# 3) tags added by the user (categorical input assumed to be one-hot encoded)

# outputs
# 1) priority score of the ticket between 0 and 1
# 2) the department that should handle the ticket

vocabulary_size = 10000  # size of the vocabulary of the text_input
num_tags = 100  # assumed to be one-hot encoded
num_department = 4

num_samples = 1280

title_data = np.random.randint(0, 2, size=(num_samples, vocabulary_size))
text_body_data = np.random.randint(0, 2, size=(num_samples, vocabulary_size))
tags_data = np.random.randint(0, 2, size=(num_samples, num_tags))

priority_data = np.random.random(size=(num_samples, 1))
department_data = np.random.randint(0, 2, size=(num_samples, num_department))


class CustomerTicketModel(keras.Model):
    def __init__(self, num_departments):
        """

        :param num_departments:
        """
        super().__init__()
        self.concat_layer = layers.Concatenate()
        self.mixing_layer = layers.Dense(64, activation='relu')
        self.priority_scorer = layers.Dense(1, activation='sigmoid')
        self.department_classifier = layers.Dense(num_departments, activation='softmax')

    def call(self, inputs):
        """

        :param inputs:
        :return:
        """
        title = inputs['title']
        text_body = inputs['text_body']
        tags = inputs['tags']

        features = self.concat_layer([title, text_body, tags])
        features = self.mixing_layer(features)
        priority = self.priority_scorer(features)
        department = self.department_classifier(features)
        return priority, department

model = CustomerTicketModel(num_departments=num_department)
# priority, department = model({'title':title_data, 'text_body':text_body_data, 'tags':tags_data})

model.compile(optimizer='rmsprop',
              loss=['mean_squared_error', 'categorical_crossentropy'],
              metrics=[['mean_absolute_error'], ['accuracy']])

model.fit(x={'title': title_data, 'text_body': text_body_data, 'tags': tags_data},
          y=[priority_data, department_data],
          epochs=1)
model.evaluate(x={'title': title_data, 'text_body': text_body_data, 'tags': tags_data},
               y=[priority_data, department_data])
priority_preds, department_preds = model.predict({'title': title_data, 'text_body': text_body_data,
                                                  'tags': tags_data})









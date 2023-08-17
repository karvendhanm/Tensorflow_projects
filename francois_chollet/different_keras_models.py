"""
There are 3 different ways to building keras models.
    1) The Sequential model - mostly used by Novice users
    2) Functional API - mostly used by engineers
    3) model subclassing - mostly used by researchers
"""

import numpy as np

from tensorflow import keras
from tensorflow.keras import layers

# the Sequential model/sequential API example

model = keras.Sequential([
    layers.Dense(64, activation='relu', name='first_layer'),
    layers.Dense(10, activation='softmax', name='second_layer')
], name='my sequential model')

# model.weights

# building the model
model.build(input_shape=(None, 3))

# model.weights

model.summary()

# building the same sequential model incrementally
model_ = keras.Sequential()
model_.add(layers.Dense(64, activation='relu'))
model_.add(layers.Dense(10, activation='softmax'))

model_.build(input_shape=(None, 3))

model_.weights

model_.summary()

# building model on the fly
model_v1 = keras.Sequential(name='sequential_model')
model_v1.add(keras.Input(shape=(3,)))
model_v1.add(layers.Dense(64, activation='relu'))
model_v1.summary()
model_v1.weights

# adding the next layer
model_v1.add(layers.Dense(10, activation='softmax'))
model_v1.weights
model_v1.summary()

# the Functional API
inputs = keras.Input(shape=(3,), name='my_input')

# inputs is a symbolic tensor
inputs.shape

inputs.dtype

features = layers.Dense(64, activation='relu')(inputs)
outputs = layers.Dense(10, activation='softmax')(features)
model = keras.Model(inputs=inputs, outputs=outputs)

model.weights
model.summary()

# Functional API - multi-input, multi-output models.
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

title = keras.Input(shape=(vocabulary_size,), name='title')
text_body = keras.Input(shape=(vocabulary_size,), name='text_body')
tags = keras.Input(shape=(num_tags,), name='tags')

features = layers.Concatenate()([title, text_body, tags])
features = layers.Dense(64, activation='relu')(features)

priority = layers.Dense(1, activation='sigmoid', name='priority')(features)
department = layers.Dense(num_department, activation='softmax', name='department')(features)
model = keras.Model(inputs=[title, text_body, tags], outputs=[priority, department])

# graphing/plotting the structure of the model
keras.utils.plot_model(model, 'ticket_classifier.png')
keras.utils.plot_model(model, 'ticket_classifier_with_shape_info.png', show_shapes=True)

# list of layers that make up the model.
model.layers

# for each layer we can query the input and the output
model.layers[0].input   # pass through layer
model.layers[0].output  # pass through layer

model.layers[3].input
model.layers[3].output

model.weights
model.summary()

# training multi-input, multi-output keras model using functional-api
num_samples = 1280

title_data = np.random.randint(0, 2, size=(num_samples, vocabulary_size))
text_body_data = np.random.randint(0, 2, size=(num_samples, vocabulary_size))
tags_data = np.random.randint(0, 2, size=(num_samples, num_tags))

priority_data = np.random.random(size=(num_samples, 1))
department_data = np.random.randint(0, 2, size=(num_samples, num_department))

model.compile(optimizer='rmsprop',
              loss=['mean_squared_error', 'categorical_crossentropy'],
              metrics=[['mean_absolute_error'], ['accuracy']])

model.fit(x=[title_data, text_body_data, tags_data],
          y=[priority_data, department_data],
          epochs=1)
model.evaluate([title_data, text_body_data, tags_data], [priority_data, department_data])
priority_preds, department_preds = model.predict([title_data, text_body_data, tags_data])

# leveraging the names of the input.
model.compile(optimizer='rmsprop',
              loss={'priority': 'mean_squared_error', 'department': 'categorical_crossentropy'},
              metrics={'priority': ['mean_absolute_error'], 'department': ['accuracy']})
model.fit(x={'title': title_data, 'text_body': text_body_data, 'tags': tags_data},
          y={'priority': priority_data, 'department': department_data},
          epochs=1)
model.evaluate(x={'title': title_data, 'text_body': text_body_data, 'tags': tags_data},
               y={'priority': priority_data, 'department': department_data})
priority_preds, department_preds = model.predict({'title': title_data, 'text_body': text_body_data,
                                                  'tags': tags_data})




























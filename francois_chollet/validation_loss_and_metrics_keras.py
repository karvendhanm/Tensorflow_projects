import numpy as np

from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential([
        layers.Dense(1)
])

# compilation step (loss, optimizer and metric)
model.compile(optimizer=keras.optimizers.RMSprop(learning_rate=0.1),
              loss = keras.losses.MeanSquaredError(),
              metrics = [keras.metrics.BinaryAccuracy()]
              )

# another compilation method
model.compile(optimizer= 'rmsprop',
              loss = 'mean_squared_error',
              metrics = ['accuracy']
              )







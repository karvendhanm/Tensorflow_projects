# regression model
import keras

from tensorflow.keras import layers
from tensorflow.keras.datasets import boston_housing

# loading the data
(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()

# feature-wise normalization of the data
# since all the feature in the dataset are in different scales, it would make learning more difficult for
# the model.

# calculating the mean and standard deviation
mean = train_data.mean(axis=0)
# wonder what happens if standard deviation was calculated after subtracting the mean from the feature.
std = train_data.std(axis=0)


# normalization
"""
note that only the metrics computed on training data was used on test data as well.
no metrics computed on test data should be used in the workflow
"""
train_data -= mean
test_data -= mean

train_data /= std
test_data /= std

# smaller the training data, worse overfitting will be, and using a small model with less number of
# layers is one way of mitigating that.

def build_model():
    model = keras.Sequential([
        layers.Dense(64, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(1) # no activation function for regression models
    ])

    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model





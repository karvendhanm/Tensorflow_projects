# regression model
import keras
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from tensorflow.keras import layers
from tensorflow.keras.datasets import boston_housing

matplotlib.use('agg')

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


"""
since the training data 400 odd samples, splitting a validation set would from this, will result
in high variance. solution: use k-fold cross validation
"""
k = 4 # typically the number of splits in 4 or 5.
num_val_samples = len(train_data) // k
num_epochs = 100
all_scores = []
for i in range(k):
    print(f'processing fold #{i}')
    val_data = train_data[i * num_val_samples : (i + 1) * num_val_samples]
    val_target = train_targets[i * num_val_samples : (i + 1) * num_val_samples]
    partial_train_data = np.concatenate(
        [train_data[:i * num_val_samples],
         train_data[(i + 1) * num_val_samples:]],
        axis=0)
    partial_train_targets = np.concatenate(
        [train_targets[:i * num_val_samples],
         train_targets[(i + 1) * num_val_samples:]],
        axis=0)
    model = build_model()
    # train the model in silent mode. verbose=0
    model.fit(x=partial_train_data, y=partial_train_targets, batch_size=16, epochs=num_epochs, verbose=0)
    val_mse, val_mae = model.evaluate(val_data, val_target, verbose=0)
    all_scores.append(val_mae)

print(all_scores)
print(np.mean(all_scores))

"""
increasing the epochs to 500 and few other changes
"""

k = 4  # typically the number of splits in 4 or 5.
num_val_samples = len(train_data) // k
num_epochs = 500
all_mae_history = []
for i in range(k):
    print(f'processing fold #{i}')
    val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
    val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]
    partial_train_data = np.concatenate(
        [train_data[:i * num_val_samples],
         train_data[(i + 1) * num_val_samples:]],
        axis=0)
    partial_train_targets = np.concatenate(
        [train_targets[:i * num_val_samples],
         train_targets[(i + 1) * num_val_samples:]],
        axis=0)
    model = build_model()
    # train the model in silent mode. verbose=0
    history = model.fit(x=partial_train_data, y=partial_train_targets,
              batch_size=16, epochs=num_epochs, verbose=0,
              validation_data=(val_data, val_targets))
    mae_history = history.history['val_mae']
    all_mae_history.append(mae_history)

# compute the average of the per-epoch MAE scores of all the k-folds.
average_mae_history = [np.mean([x[i] for x in all_mae_history]) for i in range(num_epochs)]

# lets plot it out
plt.figure()
plt.plot(range(1, len(average_mae_history)+1), average_mae_history)
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.savefig('./data/images/regression_MAE_per_epoch.png')


# first 10 values dominate. so lets omit them and plot again
plt.figure()
truncated_mae_history = average_mae_history[10:]
plt.plot(range(1, len(truncated_mae_history)+1), truncated_mae_history)
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.savefig('./data/images/truncated_regression_MAE_per_epoch.png')

# using the whole of the training data.
model = build_model()
model.fit(x=train_data, y=train_targets, epochs=130, batch_size=16, verbose=0)
test_mse_score, test_mae_acore = model.evaluate(test_data, test_targets)

print(test_mae_acore)

predictions = model.predict(test_data)
predictions[0]










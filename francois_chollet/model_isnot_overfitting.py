"""
In a ML/DL problem there are 3 common types of problems while training the model
    1) training loss doesn't go down, it stalls
    2) training loss goes down, but the model doesn't generalize. Unable to beat a trivial baseline.
    3) training and validation loss both goes down, but the model isn't overfitting.

    things to look out for when confronted with problem number 1:
    it is usually the case of 1) choice of optimizer
                              2) the distribution of initial values in the weights of the model.
                              3) learning rate
                              4) batch size

        Even as the 4 aforementioned parameters are interdependent, usually we can tune learning rate and batch size
        and get the training loss to go down.
"""
# contrived example for a model that doesn't overfit
# contrived example for stalled training loss
import matplotlib
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist

matplotlib.use('agg')

# loading the mnist dataset
(train_data, train_labels), _ = mnist.load_data()

# reshaping and normalizing training data
train_data = train_data.reshape(60000, 28 * 28)
train_data = train_data.astype('float32')/255


def load_model(lr = 1e-2):
    """

    :param lr: learning rate
    :return:
    """
    model = keras.Sequential([
        layers.Dense(10, activation='softmax') # purposefully reducing the representation power of the model.
    ])

    model.compile(optimizer=keras.optimizers.RMSprop(lr),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


model = load_model()
history = model.fit(x=train_data, y=train_labels,
          batch_size=128,
          epochs=20,
          validation_split=0.2)

# plotting the path of training and validation loss with epochs
training_loss = history.history['loss']
validation_loss = history.history['val_loss']
epochs = range(1, len(training_loss) +  1)

plt.figure(figsize = (5, 7))
plt.plot(epochs, training_loss, 'b-', label='training loss')
plt.plot(epochs, validation_loss, 'b--',  label='validation loss')
plt.title('effect of insufficient model capacity on validation loss')
plt.xlabel('Epochs')
plt.ylabel('loss')
plt.legend()
plt.savefig('./data/images/small_model.png')

# Increasing the model capacity to get the model to overfit (adding a layer to the model arcitecture)
def load_model(lr = 1e-2):
    """

    :param lr: learning rate
    :return:
    """
    model = keras.Sequential([
        layers.Dense(96, activation='relu'),
        layers.Dense(96, activation='relu'),
        layers.Dense(10, activation='softmax') # purposefully reducing the representation power of the model.
    ])

    model.compile(optimizer=keras.optimizers.RMSprop(lr),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

model = load_model()
history = model.fit(x=train_data, y=train_labels,
          batch_size=128,
          epochs=20,
          validation_split=0.2)

# plotting the path of training and validation loss with epochs
training_loss = history.history['loss']
validation_loss = history.history['val_loss']
epochs = range(1, len(training_loss) +  1)

plt.figure(figsize = (5, 7))
plt.plot(epochs, training_loss, 'b-', label='training loss')
plt.plot(epochs, validation_loss, 'b--',  label='validation loss')
plt.title('effect of insufficient model capacity on validation loss')
plt.xlabel('Epochs')
plt.ylabel('loss')
plt.legend()
plt.savefig('./data/images/bigger_model.png')



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
        layers.Dense(512, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer=keras.optimizers.RMSprop(lr), # absurdly large learning rate of 1.
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

model = load_model(1.)
history = model.fit(x=train_data, y=train_labels,
          batch_size=128,
          epochs=10,
          validation_split=0.2)

# plotting the path of training and validation loss with epochs
# since the initial loss in the 1st epoch(epoch 0) is a huge number, leaving it out
training_loss = history.history['loss'][1:]
validation_loss = history.history['val_loss'][1:]
epochs = range(1, len(training_loss) +  1)

plt.figure(figsize = (5, 7))
plt.plot(epochs, training_loss, 'b-', label='training loss')
plt.plot(epochs, validation_loss, 'b--',  label='validation loss')
plt.xlabel('Epochs')
plt.ylabel('loss')
plt.legend()
plt.savefig('./data/images/stalled_training_loss.png')

# the same model with a more appropriate learning rate
model = load_model(1e-2)
history = model.fit(x=train_data, y=train_labels,
          batch_size=128,
          epochs=10,
          validation_split=0.2)

# plotting the path of training and validation loss with epochs
training_loss = history.history['loss']
validation_loss = history.history['val_loss']
epochs = range(1, len(training_loss) +  1)

plt.figure(figsize = (5, 7))
plt.plot(epochs, training_loss, 'b-', label='training loss')
plt.plot(epochs, validation_loss, 'b--',  label='validation loss')
plt.xlabel('Epochs')
plt.ylabel('loss')
plt.legend()
plt.savefig('./data/images/training_loss_with_appropriate_learning_rate.png')






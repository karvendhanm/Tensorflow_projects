import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist

(train_data, train_labels), (test_data, test_labels) = mnist.load_data()

train_data = train_data.reshape(60000, 28 * 28).astype('float32') / 255
test_data = test_data.reshape(10000, 28 * 28).astype('float32') / 255

# creating a validation set from the training data
validation_set_data, train_set_data = train_data[:10000], train_data[10000:]
validation_set_labels, train_set_labels = train_labels[:10000], train_labels[10000:]

# ##########
# # SEQUENTIAL API STARTS
# model = keras.Sequential([
#     layers.Dense(512, activation='relu', name='first_hidden_layer'),
#     layers.Dense(10, activation='softmax', name='output_layer')
# ], name='my_model')
# # SEQUENTIAL API ENDS
# #########

# ##########
# FUNCTIONAL API STARTS
# using functional api
inputs = keras.Input(shape=(28 * 28))
features = layers.Dense(512, activation='relu', name='first_hidden_layer')(inputs)
outputs = layers.Dense(10, activation='softmax', name='output_layer')(features)
model = keras.Model(inputs=inputs, outputs=outputs)

# model.summary()
# model.weights

# plotting model structure
keras.utils.plot_model(model, show_shapes=True, to_file='./model_structure/mnist.png')
# FUNCTIONAL API ENDS
# ##########


# to use tensorboard seperately:
# tensorboard is a form of callback which we need to use during fit
# tensorboard = keras.callbacks.TensorBoard(log_dir='/home/karvsmech/PycharmProjects/Tensorflow_projects')

# early stopping, model checkpoints and tensorboard
callbacks_list = [
    keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=2),
    keras.callbacks.ModelCheckpoint(
        filepath='/home/karvsmech/PycharmProjects/Tensorflow_projects/model_checkpoints/mnist',
        monitor='val_loss',
        save_best_only=True),
    keras.callbacks.TensorBoard(log_dir='/home/karvsmech/PycharmProjects/Tensorflow_projects/tensor_board/mnist')
]

# compile, fit, evaluate and predict
model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = model.fit(x=train_set_data,
                    y=train_set_labels,
                    epochs=20,
                    callbacks=callbacks_list,
                    validation_data=(validation_set_data, validation_set_labels))
model.evaluate(x=test_data, y=test_labels)
model.predict(x=test_data)

# plotting the training loss and the validation loss
history_dict = history.history
training_loss = history_dict['loss']
training_accuracy = history_dict['accuracy']
validation_loss = history_dict['val_loss']
validation_accuracy = history_dict['val_accuracy']
epochs = list(range(1, len(training_loss) + 1))
plt.plot(epochs, training_loss, color='red', label='training loss', linestyle='dotted')
plt.plot(epochs, validation_loss, color='blue', label='validation loss', linestyle='solid')
plt.xlabel('epochs')
plt.legend()
plt.show()

plt.clf()
plt.plot(epochs, training_accuracy, color='red', label='training accuracy', linestyle='dotted')
plt.plot(epochs, validation_accuracy, color='blue', label='validation accuracy', linestyle='solid')
plt.xlabel('epochs')
plt.legend()
plt.show()


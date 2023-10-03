from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist

(train_data, train_labels), (test_data, test_labels) = mnist.load_data()

train_data = train_data.reshape(60000, 28, 28, 1).astype('float32') / 255
test_data = test_data.reshape(10000, 28, 28, 1).astype('float32') / 255

# using functional api
# convnet takes as an Input tensor of shape (image height, image width, and image channel).
inputs = keras.Input(shape=(28, 28, 1), name='input_layer')
x = layers.Conv2D(filters=32, kernel_size=3, activation='relu', name='first_convolutional_layer')(inputs)
x = layers.MaxPooling2D(pool_size=2, name='first_maxpool_layer')(x)
x = layers.Conv2D(filters=64, kernel_size=3, activation='relu', name='second_convolutional_layer')(x)
x = layers.MaxPooling2D(pool_size=2, name='second_maxpool_layer')(x)
x = layers.Conv2D(filters=128, kernel_size=3, activation='relu', name='third_convolutional_layer')(x)
# x = layers.MaxPooling2D(pool_size=2, name='third_maxpool_layer')(x)
x = layers.Flatten(name='flattening_layer')(x)
outputs = layers.Dense(10, activation='softmax', name='output_layer')(x)
model = keras.Model(inputs=inputs, outputs=outputs)

model.summary()
model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x=train_data, y=train_labels, batch_size=512, epochs=5)
test_loss, test_accuracy = model.evaluate(x=test_data, y=test_labels)



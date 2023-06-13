from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

model = keras.Sequential([
    layers.Dense(512, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='rmsprop',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

train_images = train_images.reshape(60000, 28 * 28)
train_images = train_images.astype('float32') / 255
test_images = test_images.reshape(10000, 28 * 28)
test_images = test_images.astype('float32') / 255

model.fit(train_images, train_labels, batch_size=128, epochs=5)

test_digits = test_images[:10]
predictions = model.predict(test_digits)
predictions[0].argmax()

test_loss, test_accuracy = model.evaluate(test_images, test_labels)
print(f"test accuracy is: {test_accuracy}")
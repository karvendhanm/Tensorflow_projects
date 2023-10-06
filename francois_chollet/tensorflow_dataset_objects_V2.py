import numpy as np
import tensorflow as tf

random_numbers = np.random.normal(size=(1000, 16))
dataset = tf.data.Dataset.from_tensor_slices(random_numbers)

for i, element in enumerate(dataset):
    print(f'the data at index: {i} is {element}')
    break

# creating batches
batched_dataset = dataset.batch(4)
for i, element in enumerate(batched_dataset):
    print(f'the batch at index: {i} is {element}')
    print(f'the shape of the data in the first batch is: {element.shape}')
    break

shuffled_dataset = dataset.shuffle(1000)
for i, element in enumerate(shuffled_dataset):
    print(f'the data at index: {i} is {element}')
    break

reshaped_dataset = dataset.map(lambda x: tf.reshape(x, (4, 4)))
for i, element in enumerate(reshaped_dataset):
    print(f'the batch at index: {i} is {element}')
    print(f'the shape of the data in the first batch is: {element.shape}')
    break

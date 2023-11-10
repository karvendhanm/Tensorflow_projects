import matplotlib.pyplot as plt
import numpy as np

from tensorflow import keras
from tensorflow.keras import layers

model = keras.models.load_model('./model_checkpoints/convnet_from_scratch_with_augmentation.keras')
model.summary()

img_path = keras.utils.get_file(
    fname='cat.jpg',
    origin='https://img-datasets.s3.amazonaws.com/cat.jpg'
)

def get_img_array(img_path, target_size):
    """

    :param img_path:
    :param target_size:
    :return:
    """
    img = keras.utils.load_img(img_path, target_size=target_size)
    array = keras.utils.img_to_array(img)
    array = np.expand_dims(array, axis=0)
    return array


img_tensor = get_img_array(img_path, target_size=(180, 180))

# displaying the picture
plt.axis('off')
plt.imshow(img_tensor[0].astype('uint8'))
plt.savefig('./plots/visualizing_intermediate_activations.png')

layer_outputs = []
layer_names = []
for layer in model.layers:
    if isinstance(layer, (layers.Conv2D, layers.MaxPooling2D)):
        layer_outputs.append(layer.output)
        layer_names.append(layer.name)
activation_model = keras.Model(inputs=model.input, outputs=layer_outputs)

activations = activation_model.predict(img_tensor)

# output/activation of first convolutional layer
first_layer_activation = activations[0]
print(first_layer_activation.shape)

plt.axis('off')
plt.imshow(first_layer_activation[0, :, :, 6], cmap='viridis')
plt.savefig('./plots/first_layer_activation.png')









"""
Fine-tuning a pre-trained model.

Here a densely connected classifier is added on top of the pre-trained model(VGG16, a convolutional base).

then entire convolutional base is freezed, and the added densely connected classifier is trained.

then after the densely connected classifier is trained, few top layers (specialized layers) of the

convolutional base is unfreezed, and we again train both the newly unfrozen layers in the convolutional base

and the densely connected classifier.

"""
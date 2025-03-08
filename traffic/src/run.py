import readTrafficSigns as rd
import prepare
import numpy as np

images, labels = rd.readTrafficSigns('/Users/lianglihang/Desktop/OCR/traffic/GTSRB')
image_shape = images[0].shape
image_size = image_shape[0]
sign_classes, class_indices, class_counts = np.unique(labels, return_index = True, return_counts = True)
n_classes = class_counts.shape[0]
print("Number of classes =", n_classes)
print("Image data shape =", image_shape)
print("Image size =", image_size)
#images, labels = prepare.prepareDateset(images, labels)
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()  # 自动识别本地文件

# Normalize the images
train_images = (train_images / 255) - 0.5
test_images = (test_images / 255) - 0.5

# Flatten the images
train_images = train_images.reshape((-1, 784))
test_images = test_images.reshape((-1, 784))

# Build the model here
model = Sequential([
    # layers...
    # 由于我们只是在构建一个标准的前馈网络，我们只需要密集层
    Dense(64, activation='relu', input_shape=(784,)),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax'),
])

# 编译
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy'],# 指标列表
)

# 训练模型
model.fit(
    train_images,
    to_categorical(train_labels),
    epochs=5,
    batch_size=32,
)

model.evaluate(
    test_images,
    to_categorical(test_labels)
)

model.save_weights('model.weights.h5')
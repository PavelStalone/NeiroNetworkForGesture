import math

import matplotlib.pyplot as plt
import os

import numpy as np
import tensorflow as tf
import pandas as pd
from keras.layers import Dense, Flatten, MaxPool2D, Conv2D, Dropout, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model, Sequential
from keras.preprocessing.image import image_utils
import keras

IMAGE_SIZE = 255
BATCH_SIZE = 8
AUTOTUNE = tf.data.AUTOTUNE
SEED = 321
DATA_SET_PATH = "NewData"
NAME_MODEL = "TestModel"

classes = os.listdir(DATA_SET_PATH)
train_df = pd.DataFrame(columns=['image', 'class'])
for label in classes:
    images = os.listdir(f"{DATA_SET_PATH}/{label}")
    for image in images[:-200]:
        train_df = pd.concat(
            [train_df, pd.DataFrame.from_records([{'image': f"./{DATA_SET_PATH}/{label}/{image}", 'class': label}])],
            ignore_index=True)

model = tf.keras.models.load_model(NAME_MODEL)
model.summary()

def showLayers(imageFromDS, model):
    img = image_utils.load_img(imageFromDS, target_size=(IMAGE_SIZE, IMAGE_SIZE))
    img_tensor = image_utils.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    img_tensor /= 255.

    layer_outputs = [layer.output for layer in model.layers[2:16]]
    activation_model = tf.keras.models.Model(inputs=model.input, outputs=layer_outputs)
    acts = activation_model.predict(img_tensor)

    images_per_row = 16

    i = 1

    for layer_output, layer_activation in zip(layer_outputs, acts):
        if not layer_output.name.startswith('max') and not layer_output.name.startswith('average'):
            # Количество фильтров в карте признаков
            n_features = layer_activation.shape[-1]

            # Признак имеет форму: (1, высота, ширина, n_features)
            size = layer_activation.shape[1]

            # Разместим результаты активации в виде сетки.
            # На каждой строке будет по images_per_row (16)
            n_cols = n_features // images_per_row
            display_grid = np.zeros((size * n_cols, images_per_row * size))

            for col in range(n_cols):
                for row in range(images_per_row):
                    # Фильтр:
                    channel_image = layer_activation[0,
                                    :, :,
                                    col * images_per_row + row]
                    # Постобработка, чтобы получить приемлимую визуализацию
                    channel_image -= channel_image.mean()
                    channel_image /= channel_image.std()
                    channel_image *= 64
                    channel_image += 128
                    channel_image = np.clip(channel_image, 0, 255).astype('uint8')
                    display_grid[col * size: (col + 1) * size,
                    row * size: (row + 1) * size] = channel_image

            # Визуализация результатов активации модели TensorFlow
            scale = 1. / size
            plt.figure(figsize=(scale * display_grid.shape[1],
                                scale * display_grid.shape[0]))
            plt.title(f"Conv2D number {i}")
            plt.grid(False)
            plt.imshow(display_grid, aspect='auto', cmap='viridis')
            i += 1
    plt.show()


showLayers(train_df.iloc[0].image, model)

TEST_PATH = "TestDataSet"
test_df = pd.DataFrame(columns=['image', 'class'])
for image in os.listdir(TEST_PATH):
    test_df = pd.concat([test_df, pd.DataFrame.from_records([{'image': f"./{TEST_PATH}/{image}", 'class': image}])],
                        ignore_index=True)

IMAGE_IN_ROW = 5
plt.figure(figsize=(10,10))
for i in range(len(test_df)):
    img = image_utils.load_img(test_df.iloc[i].image, target_size=(IMAGE_SIZE, IMAGE_SIZE))

    img_tensor = image_utils.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    img_tensor /= 255.
    prediction = model.predict(img_tensor)

    plt.subplot(math.ceil(len(test_df) / IMAGE_IN_ROW), IMAGE_IN_ROW, i + 1)
    plt.imshow(img)
    plt.title(test_df['class'].iloc[i])
    plt.xlabel(classes[np.argmax(prediction)])
    plt.grid(False)
plt.show()

showLayers(test_df.iloc[0].image, model)

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
from tensorflow import keras
from keras.constraints import maxnorm

IMAGE_SIZE = 255
BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE
SEED = 321
DATA_SET_PATH = "DataSetFromVideo"
NAME_MODEL = "TestModel"
TEST_PATH = "TestDataSet"

classes = os.listdir(DATA_SET_PATH)
print(classes)

train_df = pd.DataFrame(columns=['image', 'class'])
val_df = pd.DataFrame(columns=['image', 'class'])
test_df = pd.DataFrame(columns=['image', 'class'])

for image in os.listdir(TEST_PATH):
    test_df = pd.concat([test_df, pd.DataFrame.from_records([{'image': f"./{TEST_PATH}/{image}", 'class': image}])],
                        ignore_index=True)

for label in classes:
    images = os.listdir(f"{DATA_SET_PATH}/{label}")
    for image in images[:-213]:
        train_df = pd.concat(
            [train_df, pd.DataFrame.from_records([{'image': f"./{DATA_SET_PATH}/{label}/{image}", 'class': label}])],
            ignore_index=True)

    for image in images[-213:]:
        val_df = pd.concat(
            [val_df, pd.DataFrame.from_records([{'image': f"./{DATA_SET_PATH}/{label}/{image}", 'class': label}])],
            ignore_index=True)

print(train_df.head())

train_datagen = ImageDataGenerator(
    rotation_range=5,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.1,
    horizontal_flip=True,
    rescale=1 / 255,
)

val_datagen = ImageDataGenerator(
    horizontal_flip=True,
    rescale=1 / 255
)

train_generator = train_datagen.flow_from_dataframe(train_df, x_col='image',
                                                    y_col='class',
                                                    classes=classes,
                                                    shuffle=True,
                                                    batch_size=BATCH_SIZE)
val_generator = val_datagen.flow_from_dataframe(val_df, x_col='image',
                                                y_col='class',
                                                classes=classes,
                                                shuffle=True,
                                                batch_size=BATCH_SIZE)
layers = tf.keras.layers

rescale = tf.keras.Sequential([
    layers.Resizing(IMAGE_SIZE, IMAGE_SIZE),
])
redraw = tf.keras.Sequential([
    layers.Lambda(lambda x: tf.image.rgb_to_grayscale(x))
])
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal", seed=SEED, input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)),
    layers.RandomRotation(0.1, seed=SEED, fill_mode="reflect"),
])

# test_model = tf.keras.models.Sequential([
#     rescale,
#     data_augmentation,
#
#     layers.Conv2D(32, (5, 5), padding="same", activation='relu'),
#     keras.layers.BatchNormalization(),
#     layers.Conv2D(64, (5, 5), padding="same", activation='relu'),
#     keras.layers.BatchNormalization(),
#     layers.SeparableConv2D(12, (3, 3), padding="same", activation='relu'),
#     keras.layers.BatchNormalization(),
#     layers.SeparableConv2D(12, (3, 3), padding="same", activation='relu'),
#     keras.layers.BatchNormalization(),
#     layers.Conv2D(128, (3, 3), padding="same", activation='relu'),
#     layers.MaxPooling2D((2, 2)),
#     keras.layers.BatchNormalization(),
#     # layers.Add(),
#     layers.SeparableConv2D(256, (3, 3), padding="same", activation='relu'),
#     keras.layers.BatchNormalization(),
#     layers.SeparableConv2D(256, (3, 3), padding="same", activation='relu'),
#     keras.layers.BatchNormalization(),
#     layers.Conv2D(256, (3, 3), padding="same", activation='relu'),
#     layers.MaxPooling2D((2, 2)),  # [!]
#     keras.layers.BatchNormalization(),
#     #     layers.Add(),
#     # layers.SeparableConv2D(728, (3, 3), padding="same", activation='relu'),
#     #     keras.layers.BatchNormalization(),
#
#     # layers.Conv2D(64, (3, 3), padding="same", activation='relu'),
#     # layers.MaxPooling2D((2, 2)),
#     # layers.Dropout(0.2),
#     # keras.layers.BatchNormalization(),
#     # layers.Conv2D(128, (3, 3), padding="same", activation='relu'),
#     # layers.Dropout(0.2),
#     # keras.layers.BatchNormalization(),
#
#     layers.Flatten(),
#     layers.Dropout(0.2),
#     layers.Dense(256, activation='relu', kernel_constraint=maxnorm(3)),
#     layers.Dropout(0.2),
#     keras.layers.BatchNormalization(),
#     layers.Dense(128, activation='relu'),
#     layers.Dropout(0.2),
#     keras.layers.BatchNormalization(),
#     layers.Dense(len(classes), activation="softmax")
# ])
#
# opt = keras.optimizers.Adadelta(learning_rate=0.01)
# test_model.compile(loss='categorical_crossentropy', optimizer=opt,
#                    metrics=['acc'])
# history = test_model.fit(train_generator, epochs=1, validation_data=val_generator)
#
# test_model.save(NAME_MODEL)

from keras.applications.xception import Xception

xception = Xception(include_top=False, input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3), classes=len(classes))
xception.trainable = False
last_layer = xception.layers[-1].output
x = GlobalAveragePooling2D()(last_layer)
x = Dense(len(classes), activation='softmax')(x)
model = Model(xception.inputs, x)

model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.RMSprop(learning_rate=0.001), metrics=['acc'])
history = model.fit(train_generator, epochs=2, validation_data=val_generator)

model.save(NAME_MODEL)

IMAGE_IN_ROW = 5
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

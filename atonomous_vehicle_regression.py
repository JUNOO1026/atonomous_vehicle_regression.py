import os, glob, sys
import numpy as np
from sklearn.model_selection import train_test_split
from PIL import Image
from tensorflow import keras
from pathlib import Path    #path를 객체 처럼 사용(즉, 디렉토리를 왔다갔다 할 수 있음.)
import tensorflow as tf
import pandas as pd
import cv2
import time
import os
from keras.models import load_model
np.set_printoptions(threshold=sys.maxsize)

print('Python version', sys.version)
print('Tensorflow version', tf.__version__)
print('keras version', keras.__version__)

path = os.getcwd()
print(path)

folder_directory_path = './regression_length_direction_data/'

image_dir = Path(folder_directory_path)

filepaths = pd.Series(list(image_dir.glob(r'**/*.bmp')), name='Filepath').astype(str)
# print(filepaths)
#
# print(filepaths.values)
length = pd.Series(filepaths.apply(lambda x: os.path.split(os.path.split(x)[0])[1]), name='length').astype(np.int)

# print(type(filepaths.apply(lambda x: os.path.split(os.path.split(x)[0])[1]).astype(np.int)))
# # print(os.path.split(os.path.split(filepaths.values[0])[0])[1])
#
# print("##########################")
# print(length)

images = pd.concat([filepaths, length], axis=1)

# print(images)

train_df, test_df = train_test_split(images, train_size=0.8, test_size=0.2, shuffle=True, random_state=1)
#
# print(type(train_df))
# print(train_df.columns)
# print(train_df.shape)
# print(type(test_df))
# print(test_df.shape)

train_generator = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

test_generator = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255
)

train_images = train_generator.flow_from_dataframe(
    dataframe=train_df,
    x_col='Filepath',
    y_col= 'length',
    target_size=(160, 90),
    color_mode='grayscale',
    class_mode='raw',
    batch_size=32,
    seed= 42,
    shuffle=True,
    subset='training'
)

val_images = train_generator.flow_from_dataframe(
    dataframe=train_df,
    x_col='Filepath',
    y_col= 'length',
    target_size=(160, 90),
    color_mode='grayscale',
    class_mode='raw',
    batch_size=32,
    seed=42,
    shuffle=True,
    subset='validation'
)

test_images = test_generator.flow_from_dataframe(
    dataframe=test_df,
    x_col='Filepath',
    y_col='length',
    target_size=(160, 90),
    color_mode='grayscale',
    class_mode='raw',
    batch_size=32,
    shuffle=False
)

print(type(train_images))
print(train_images)

inputs = tf.keras.Input(shape=(160, 90, 1))
print(inputs)
x = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(inputs)
x = tf.keras.layers.MaxPooling2D()(x)
x = tf.keras.layers.Dropout(rate=0.4)(x)
x = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu')(x)
# x = tf.keras.layers.Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(x)
x = tf.keras.layers.MaxPooling2D()(x)
x = tf.keras.layers.Dropout(rate=0.3)(x)
x = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), activation='relu')(x)
x = tf.keras.layers.MaxPooling2D()(x)
x = tf.keras.layers.Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(x)
x = tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), activation='relu')(x)
x = tf.keras.layers.MaxPooling2D()(x)
x = tf.keras.layers.Conv2D(filters=256, kernel_size=(1, 1), activation='relu')(x)
x = tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), activation='relu')(x)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
print(x)
x = tf.keras.layers.Dropout(rate=0.5)(x)
x = tf.keras.layers.Dense(1000, activation='relu')(x)
outputs = tf.keras.layers.Dense(1, activation='linear')(x)

model = tf.keras.Model(inputs=inputs, outputs=outputs)

model.compile(
    optimizer='adam',
    loss='mse',
    metrics=['accuracy']
)

model.summary()

# history = model.fit(
#     train_images,
#     validation_data=val_images,
#     epochs=10,
#     # callbacks=[
#     #     tf.keras.callbacks.EarlyStopping(
#     #         monitor='val_loss',
#     #         patience=5,
#     #         restore_best_weights=True
#     # )]
# )
#
# #predicted_length = np.squeeze(model.predict(test_images))
# true_length = test_images.labels
#
# print(true_length)
#
# print(model.predict(test_images, verbose=1))
#
# model.save('atonomous_vehicle_regression_amolang_01_02.h5')
#





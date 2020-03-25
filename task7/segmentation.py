import os
import numpy as np
from skimage.io import imread, imshow
from skimage.transform import resize

import matplotlib.pyplot as plt

import keras
from keras.layers import Concatenate, Conv2D, UpSampling2D, concatenate
from keras.models import Model, load_model
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

BATCH_SIZE = 16
IMAGE_SIZE = 224

def normalize_image(img_as_array):
    return (img_as_array - 127.5) / 127.5

def predict(model, img_path):
    image = imread(img_path)
    h, w = image.shape[:2]

    image = normalize_image(image)
    image = resize(image, (IMAGE_SIZE, IMAGE_SIZE, 3))

    image_with_batch_dim = np.array([image])
    pred = model.predict(image_with_batch_dim)[0]
    if len(pred.shape) == 3:
        pred = np.reshape(pred, pred.shape[:2])
    if h is not None and w is not None:
         pred = resize(pred, (h, w))
    #plt.imshow(pred)
    #plt.waitforbuttonpress()
    return np.clip(pred, 0, 1).astype(np.float32)

def data_generators(train_data_dir, gt_data_dir):
    data_generator = ImageDataGenerator(preprocessing_function=normalize_image).flow_from_directory(train_data_dir, batch_size=BATCH_SIZE,
                                                              class_mode=None, target_size=(IMAGE_SIZE, IMAGE_SIZE),
                                                              color_mode="rgb", seed=42)
    gt_generator = ImageDataGenerator(rescale=1./255.).flow_from_directory(gt_data_dir, batch_size=BATCH_SIZE,
                                                                         class_mode=None, target_size=(IMAGE_SIZE, IMAGE_SIZE),
                                                                         color_mode="grayscale", seed=42)
    train_data_generator = zip(data_generator, gt_generator)
    train_data_generator = (pair for pair in train_data_generator)
    return train_data_generator


def build_model():
    encoder = MobileNetV2((224, 224, 3), alpha=1.0, include_top=False)

    encoder_output = encoder.output

    up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(encoder_output))
    block14 = encoder.get_layer('block_13_expand_BN').output
    merge6 = concatenate([block14, up6], axis=3)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

    up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv6))
    block28 = encoder.get_layer('block_6_expand_BN').output
    merge7 = concatenate([block28, up7], axis=3)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

    up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv7))
    block56 = encoder.get_layer('block_3_expand_BN').output
    merge8 = concatenate([block56, up8], axis=3)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

    up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv8))
    block112 = encoder.get_layer('block_1_expand_BN').output
    merge9 = concatenate([block112, up9], axis=3)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)

    up10 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv9))
    conv10 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(up10)
    conv10 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv10)
    #conv10 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv10)
    conv11 = Conv2D(1, 3, activation='sigmoid', padding='same')(conv10)

    model = Model(encoder.input, conv11)

    model.compile(optimizer=Adam(lr=3e-4), loss='binary_crossentropy', metrics=['mse'])

    model.summary()

    return model


def train_model(train_data_path):
    images_train = os.path.join(train_data_path, 'images')
    gt = os.path.join(train_data_path, 'gt')
    train_data_generator = data_generators(images_train, gt)

    #x, y = next(train_data_generator)
    #x, y = x[0], y[0][:, :, 0]
    #print(x)
    #plt.imshow((x * 127.5 + 127.5).astype(np.uint8))
    #plt.waitforbuttonpress()


    #model = build_model()
    model = load_model('segmentation_model.hdf5')
    model.fit_generator(train_data_generator, steps_per_epoch=524, epochs=1)
    model.save('segmentation_model.hdf5')
    return model


if __name__ == '__main__':
    train_model('D:\\CV2020\\task7\\tests\\00_test_val_input\\train')
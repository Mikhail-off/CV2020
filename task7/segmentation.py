# coding=utf-8
import keras
from keras.layers import Input, Conv2D, Concatenate, UpSampling2D
from keras.preprocessing.image import ImageDataGenerator

from keras.applications import MobileNetV2
from keras.models import Model
from keras.optimizers import Adam

from skimage.io import imread
from skimage.transform import resize

import numpy as np

batch_size = 4
inp_sz = (224, 224, 3)


def unprocess_mask(mask, shape):
    h, w = shape[:2]
    mask = mask.reshape(mask.shape[:2])
    mask = resize(mask, (h, w))
    return mask


def UNet():
    mobileNet = MobileNetV2(input_shape=inp_sz, include_top=False, alpha=1.0)

    filters = 256
    cur = mobileNet.outputs[0]
    cur = UpSampling2D()(cur)
    cur = Conv2D(filters, 3, padding='same', activation='relu')(cur)
    cur = Concatenate()([cur, mobileNet.get_layer('block_13_expand_relu').output])

    cur = UpSampling2D()(cur)
    cur = Conv2D(filters // 2, 3, padding='same', activation='relu')(cur)
    cur = Concatenate()([cur, mobileNet.get_layer('block_6_expand_relu').output])

    cur = UpSampling2D()(cur)
    cur = Conv2D(filters // 4, 3, padding='same', activation='relu')(cur)
    cur = Concatenate()([cur, mobileNet.get_layer('block_3_expand_relu').output])

    cur = UpSampling2D()(cur)
    cur = Conv2D(filters // 8, 3, padding='same', activation='relu')(cur)
    cur = Concatenate()([cur, mobileNet.get_layer('block_1_expand_relu').output])

    cur = UpSampling2D()(cur)
    cur = Conv2D(filters // 16, 3, padding='same', activation='relu')(cur)
    cur = Concatenate()([cur, mobileNet.get_layer('input_1').output])


    cur = Conv2D(filters // 32, 3, padding='same')(cur)
    cur = Conv2D(1, 3, padding='same', activation='sigmoid')(cur)

    model = Model(mobileNet.inputs[0], cur)
    model.summary()

    model.compile(optimizer=Adam(learning_rate=3e-3), loss=['binary_crossentropy'], metrics=['mse', 'mae'])

    return model


def train_model(train_data_path):
    images = train_data_path + 'images\\'
    masks = train_data_path + 'gt\\'

    # mobileNet обучен на картинках [-1; 1], поэтому испоьзуем препроцессинг
    image_gen = ImageDataGenerator(preprocessing_function=lambda im: (im - 127.5) / 127.5).flow_from_directory(images,
                        batch_size=batch_size,
                        seed=111,
                        class_mode=None,
                        target_size=inp_sz[:2],
                        color_mode="rgb")
    mask_gen = ImageDataGenerator(rescale=1/255.).flow_from_directory(masks,
                        batch_size=batch_size,
                        seed=111,
                        class_mode=None,
                        target_size=inp_sz[:2],
                        color_mode="grayscale")

    train_data_generator = zip(image_gen, mask_gen)
    model = UNet()
    imgs_count = 8382
    model.fit_generator(train_data_generator, steps_per_epoch=imgs_count // batch_size, epochs=20)
    model.save("segmentation_model.hdf5")
    return model


def predict(model, img_path):
    img = imread(img_path)
    predict_image = (img - 127.5) / 127.5
    predict_image = resize(img_as_array, inp_sz)
    predict_image = np.array([predict_image])
    mask = model.predict(predict_image)[0]
    return unprocess_mask(mask, img.shape[:2])
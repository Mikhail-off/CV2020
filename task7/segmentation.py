import keras
from keras.layers import Input, Conv2D, Concatenate, BatchNormalization, Dropout, Dense, Flatten,\
    UpSampling2D, Conv2DTranspose
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from tqdm import tqdm
from keras.applications import MobileNetV2
from skimage.transform import resize
from keras.models import Model, Sequential, load_model
from keras.optimizers import Adam
import os
from skimage.io import imread, imsave
from keras.utils import plot_model

train_data_path = 'C:\\CV2020\\task7\\tests\\00_test_val_input\\train\\'
model_name = "segmentation_model.hdf5"

BATCH_SIZE = 4
EPOCHS = 20
LR = 3e-3
SEED = 42
INPUT_SHAPE = (224, 224, 3)
ALPHA = 1.0



def resize_to_shape(img_as_array, shape=INPUT_SHAPE):
    h, w = shape[:2]

    if h is not None and w is not None:
        img_as_array = resize(img_as_array, shape)

    return img_as_array


def preprocess_image(img_as_array):
    return (img_as_array - 127.5) / 127.5


def deprocess_mask(mask, shape):
    h, w = shape[:2]
    if len(mask.shape) == 3:
        assert mask.shape[-1] == 1
        mask = mask.reshape(mask.shape[:2])
    if h is not None and w is not None:
        mask = resize(mask, (h, w))
        assert mask.shape == (h, w)
    return mask


def prepare_data_generators(x_data_dir, y_data_dir):
    x_generator = ImageDataGenerator(preprocessing_function=preprocess_image).flow_from_directory(x_data_dir, batch_size=BATCH_SIZE, seed=SEED,
                                                                                                  class_mode=None, target_size=INPUT_SHAPE[:2],
                                                                                                  color_mode="rgb")
    y_generator = ImageDataGenerator(rescale=1/255.).flow_from_directory(y_data_dir, batch_size=BATCH_SIZE, seed=SEED,
                                                                         class_mode=None, target_size=INPUT_SHAPE[:2],
                                                                         color_mode="grayscale")
    train_data_generator = zip(x_generator, y_generator)

    return train_data_generator


def UNet():
    mobileNet = MobileNetV2(input_shape=INPUT_SHAPE, include_top=False, alpha=1.0)

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

    model.compile(optimizer=Adam(learning_rate=LR), loss=['binary_crossentropy'], metrics=['mse', 'mae'])

    return model


def train_model(train_data_path):
    x_data_dir = train_data_path + 'images\\'
    y_data_dir = train_data_path + 'gt\\'
    train_data_generator = prepare_data_generators(x_data_dir, y_data_dir)
    model = UNet()

    #x_length = len(x_generator)
    #assert x_length == len(y_generator)
    #steps_per_epoch = (x_length - 1) // BATCH_SIZE + 1



    model.fit_generator(train_data_generator, steps_per_epoch=8382 // BATCH_SIZE, epochs=EPOCHS)
    model.save(model_name)
    return model


def predict(model, img_path):
    img = imread(img_path)
    h, w = img.shape[:2]
    img_as_array = preprocess_image(img)
    img_as_array = resize_to_shape(img_as_array, INPUT_SHAPE)
    img_as_array = np.array([img_as_array])
    mask = model.predict(img_as_array)[0]
    mask = deprocess_mask(mask, (h, w))
    return mask

def main():
    train_model(train_data_path)


if __name__ == '__main__':
    main()
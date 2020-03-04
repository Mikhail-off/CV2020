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
#from keras.utils import plot_model

train_data_path = 'D:\\CV2020\\task7\\tests\\00_test_val_input\\train'

model_name = "segmentation_model.hdf5"

BATCH_SIZE = 16
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

    x_length = len(x_generator)
    assert x_length == len(y_generator)
    steps_per_epoch = (x_length - 1) // BATCH_SIZE + 1
    return train_data_generator, steps_per_epoch


def build_model(compiled=True):
    mobileNet = MobileNetV2(input_shape=INPUT_SHAPE, include_top=False, alpha=ALPHA)
    mobileNet.summary()

    inp = mobileNet.inputs[0]
    output = mobileNet.outputs[0]

    output_shape = output.get_shape().as_list()[-3:]

    layers_to_concat = []
    prev_layer = inp
    for layer in mobileNet.layers[1:]:
        layer_input = layer.input
        layer_output = layer.output

        # попали на слой с несколькими входами или выходами
        if isinstance(layer_input, list) or isinstance(layer_output, list):
            prev_layer = layer
            continue

        layer_input_shape = layer_input.get_shape().as_list()[1:3]
        expected_shape = list(map(lambda x: x // 2, layer_input_shape))
        layer_output_shape = layer_output.get_shape().as_list()[1:3]

        # надо запомнить слой. Мы тут наткнулись на ZeroPadding, запомнить надо тензор перед ним
        if expected_shape == layer_output_shape:
            assert not (isinstance(layer_input, list))
            layers_to_concat.append(prev_layer.input)

        prev_layer = layer


    layers_to_concat = layers_to_concat[::-1]
    filters = output_shape[-1] // 2
    kernel = (3, 3)
    cur_filters = filters
    cur = output
    for layer in layers_to_concat:
        cur_filters //= 2
        cur = UpSampling2D()(cur)  # x2
        cur = Conv2D(cur_filters, kernel_size=kernel, padding='same', activation='relu')(cur)
        cur = Concatenate()([cur, layer])

    cur = Conv2D(cur_filters // 2, kernel_size=kernel, padding='same', activation='relu')(cur)
    cur = Conv2D(1, kernel_size=kernel, padding='same', activation='sigmoid')(cur)

    model = Model(inp, cur)
    model.summary()
    """
    print('Concated Layers')
    print(*layers_to_concat, sep='\n')
    """
    if compiled:
        opt = Adam(learning_rate=LR)
        model.compile(optimizer=opt, loss=['binary_crossentropy'], metrics=['mse', 'mae'])
    #plot_model(model, 'model.png')

    return model


def train_model(train_data_path):
    x_data_dir = os.path.join(train_data_path,'images')
    y_data_dir = os.path.join(train_data_path, 'gt')
    train_data_generator, steps_per_epoch = prepare_data_generators(x_data_dir, y_data_dir)

    if os.path.exists(model_name):
        model = load_model(model_name)
    else:
        model = build_model()

    model.fit_generator(train_data_generator, steps_per_epoch, epochs=EPOCHS)
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
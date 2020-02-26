import keras
from keras import backend as K
from keras.layers import Conv2D, Dropout, MaxPool2D, Input,\
    BatchNormalization, ReLU, Flatten, Dense, ELU, Concatenate
from keras.models import Model
import keras.optimizers as opt
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
import pandas as pd
import skimage
from skimage.io import imread
from skimage.transform import resize
from skimage.color import gray2rgb
from random import random
import matplotlib.pyplot as plt

IMG_SIZE = 100
POINTS = 14
BATCH_SIZE = 64
LR = 1e-3

class DataGenerator:
    def __init__(self, images_path, gt=None, aug_prob=0.0):
        self.__aug_prob = aug_prob
        self.__image_path = images_path
        self._gt_mode = False
        if isinstance(gt, dict):
            self._gt_mode = True
            self.__image_name_to_target = gt
            self.image_names = list(sorted(self.__image_name_to_target.keys()))
        elif isinstance(gt, str):
            self.__gt_mode = True
            self.__image_name_to_target = dict()
            df = pd.read_csv(gt)
            for row in df.values:
                self.__image_name_to_target[row[0]] = np.array(row[1:], dtype=np.int32)
            self.image_names = list(sorted(self.__image_name_to_target.keys()))
        else:
            self.image_names = list(sorted(list(os.listdir(images_path))))

    def get_steps_per_epoch(self, batch_size):
        return (len(self.image_names) - 1) // batch_size + 1

    def generate(self, batch_size, val_mode=False):
        cur_ind = 0

        while True:
            batch_images = None
            y_batch = None
            if cur_ind >= len(self.image_names):
                cur_ind = 0
                if val_mode:
                    break

            if cur_ind + batch_size <= len(self.image_names):
                batch_images = self.image_names[cur_ind:cur_ind + batch_size]
                if self._gt_mode:
                    y_batch = np.array([self.__image_name_to_target[img_name] for img_name in batch_images])
            else:
                batch_images = self.image_names[cur_ind:]
                if self._gt_mode:
                    y_batch = np.array([self.__image_name_to_target[img_name] for img_name in batch_images])
            if self._gt_mode:
                y_batch = y_batch.astype(int)
            x_batch = []
            #print(batch_images)
            for i, img_name in enumerate(batch_images):
                # print(img_name)
                img_path = os.path.join(self.__image_path, img_name)
                img_as_array = open_image(img_path, aug_prob=self.__aug_prob)
                if self._gt_mode:
                    img_as_array, y_batch[i] = preprocess(img_as_array, target=y_batch[i])
                else:
                    img_as_array = preprocess(img_as_array)

                x_batch += [img_as_array]

            x_batch = np.array(x_batch)
            #print(x_batch.shape)
            cur_ind += len(batch_images)
            # print('x_batch:', x_batch.size(), '\ty_batch:' ,y_batch.size())
            if self._gt_mode:
                yield x_batch, y_batch
            else:
                yield x_batch

        # мешаем данные каждую эпоху
        ind = np.random.permutation(len(self.image_names))
        self.image_names = self.image_names[ind]
        self.__labels = self.__labels[ind]

    def __len__(self):
        return len(self.image_names)


def open_image(image_path, aug_prob=0.0):
    img_as_array = skimage.img_as_float32(imread(image_path))
    if len(img_as_array.shape) == 2 or img_as_array.shape[-1] == 1:
        img_as_array = gray2rgb(img_as_array)

    #print(img_as_array.shape, image_path)
    assert len(img_as_array.shape) == 3
    assert img_as_array.shape[-1] == 3
    return img_as_array


def augment(img, target=None):
    if target is None:
        return img
    else:
        return img, target


def preprocess(img, target=None):
    h, w = img.shape[:2]
    img = resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.
    if target is None:
        return img
    else:
        coefs = np.zeros((POINTS, 2))
        coefs[:, 0] = IMG_SIZE / w
        coefs[:, 1] = IMG_SIZE / h
        coefs = coefs.reshape((-1))
        target = (target * coefs).astype(np.int32)

        if random() < 0.5:
            img = img[:, ::-1, :]
            target = target.reshape((POINTS, 2))
            target[:, 0] = IMG_SIZE - target[:, 0]

        target = target.flatten()
        return img, target

def build_model():
    inp = Input((IMG_SIZE, IMG_SIZE, 3))

    filters = 32
    kernel_size = (3, 3)

    cur = inp

    cur = Conv2D(filters, kernel_size=(7, 7), activation='relu')(cur)
    cur = Conv2D(filters, kernel_size=kernel_size, activation='relu')(cur)
    cur = Conv2D(filters, kernel_size=kernel_size, activation='relu', strides=2)(cur)
    cur = BatchNormalization()(cur)

    cur = Conv2D(2 * filters, kernel_size=kernel_size, activation='relu')(cur)
    cur = Conv2D(2 * filters, kernel_size=kernel_size, activation='relu')(cur)
    cur = BatchNormalization()(cur)

    cur = Conv2D(4 * filters, kernel_size=kernel_size, activation='relu', strides=2)(cur)
    cur = Conv2D(4 * filters, kernel_size=kernel_size, activation='relu')(cur)
    cur = Conv2D(4 * filters, kernel_size=kernel_size, activation='relu')(cur)
    cur = BatchNormalization()(cur)

    cur = Conv2D(8 * filters, kernel_size=kernel_size, activation='relu', strides=2)(cur)
    cur = Conv2D(8 * filters, kernel_size=kernel_size, activation='relu')(cur)
    cur = Conv2D(8 * filters, kernel_size=kernel_size, activation='relu')(cur)
    cur = BatchNormalization()(cur)

    cur = Flatten()(cur)
    cur = Dense(128, activation='relu')(cur)
    #cur = Dropout(0.2)(cur)
    cur = Dense(64, activation='relu')(cur)
    #cur = Dropout(0.2)(cur)

    #cur = BatchNormalization()(cur)
    cur = Dense(2 * POINTS, activation=None)(cur)
    model = Model(inputs=inp, outputs=cur)
    model.summary()
    return model


def train_detector(imgs, gts, fast_train=False):
    model = build_model()

    #model.load_weights('facepoints_model.hdf5')

    opimizer = opt.Adam(learning_rate=LR)
    model.compile(opimizer, loss=['mse'], metrics=['mse', 'mae', 'accuracy'])

    train_generator = DataGenerator(images_path=gts, gt=imgs)
    """
    x, y = next(train_generator.generate(batch_size=1))
    plt.figure()
    plt.imshow(x[0] * 255)
    plt.scatter(y[0].reshape(14, 2)[:, 0], y[0].reshape(14, 2)[:, -1], marker='o', c='r')
    plt.show()
    plt.waitforbuttonpress()
    """
    model.fit_generator(train_generator.generate(batch_size=BATCH_SIZE),
                        steps_per_epoch=train_generator.get_steps_per_epoch(BATCH_SIZE),
                        epochs=10)
    model.save_weights('facepoints_model.hdf5')
    model = build_model()
    model.load_weights('facepoints_model.hdf5')
    model.save('facepoints_model.hdf5')
    return model

def detect(model, imgs, gts=None):
    data_generator = DataGenerator(imgs)
    image_names = list(sorted(data_generator.image_names))

    predicted_coords = Model.predict_generator(model, data_generator.generate(BATCH_SIZE, True),
                                   steps=data_generator.get_steps_per_epoch(BATCH_SIZE),
                                   verbose=1)
    predicted_coords = np.clip(predicted_coords, 0, IMG_SIZE - 1)
    image_name_to_coords = dict(zip(image_names, predicted_coords))
    return image_name_to_coords
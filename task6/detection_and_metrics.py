import keras
from keras import backend as K
from keras.layers import Conv2D, MaxPool2D, BatchNormalization, Dense, Activation, Input, Flatten, Dropout
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

# ============================== 1 Classifier model ============================

INPUT_SHAPE = (40, 100, 1)
LR = 3e-4
BATCH_SIZE = 64
EPOCHS = 50

def get_cls_model(input_shape):
    """
    :param input_shape: tuple (n_rows, n_cols, n_channgels)
            input shape of image for classification
    :return: nn model for classification
    """
    # your code here \/
    # your code here /\
    inp = Input(shape=INPUT_SHAPE)

    filters = 8

    cur = inp
    cur = Conv2D(filters, kernel_size=3, activation='relu')(cur)
    cur = Conv2D(filters, kernel_size=3, activation='relu', strides=2)(cur)
    cur = Conv2D(2 * filters, kernel_size=3, activation='relu')(cur)
    cur = Conv2D(2 * filters, kernel_size=3, activation='relu', strides=2)(cur)
    cur = Conv2D(4 * filters, kernel_size=3, activation='relu')(cur)
    cur = Conv2D(4 * filters, kernel_size=3, activation='relu', strides=2)(cur)
    cur = Flatten()(cur)
    cur = Dense(4 * filters, activation='relu')(cur)
    cur = Dropout(0.2)(cur)
    cur = Dense(2, activation='softmax')(cur)
    model = Model(inputs=inp, outputs=[cur])
    model.summary()
    return model


def fit_cls_model(X, y):
    """
    :param X: 4-dim ndarray with training images
    :param y: 2-dim ndarray with one-hot labels for training
    :return: trained nn model
    """

    model = get_cls_model((40, 100, 1))
    model.compile(loss='binary_crossentropy',
                  optimizer=Adam(learning_rate=LR),
                  metrics=["accuracy"])

    image_generator = ImageDataGenerator(horizontal_flip=True,
                                         brightness_range=(0.5, 1.5),
                                         rotation_range=90,
                                         zoom_range=(0.7, 1.3),
                                         )

    image_generator.flow(X, y)

    image_generator = ImageDataGenerator()
    image_generator.fit(X)


    steps_per_epoch = (len(X)  - 1) // BATCH_SIZE + 1
    model.fit_generator(image_generator.flow(X, y, batch_size=BATCH_SIZE),
                        steps_per_epoch=steps_per_epoch,
                        epochs=EPOCHS)
    model.save('classifier_model.h5')
    return model


# ============================ 2 Classifier -> FCN =============================
def get_detection_model(cls_model):
    """
    :param cls_model: trained cls model
    :return: fully convolutional nn model with weights initialized from cls
             model
    """
    # your code here \/
    # detection_model = ...
    # return detection_model
    # your code here /\


# ============================ 3 Simple detector ===============================
def get_detections(detection_model, dictionary_of_images):
    """
    :param detection_model: trained fully convolutional detector model
    :param dictionary_of_images: dictionary of images in format
        {filename: ndarray}
    :return: detections in format {filename: detections}. detections is a N x 5
        array, where N is number of detections. Each detection is described
        using 5 numbers: [row, col, n_rows, n_cols, confidence].
    """
    # your code here \/
    return {}
    # your code here /\


# =============================== 5 IoU ========================================
def calc_iou(first_bbox, second_bbox):
    """
    :param first bbox: bbox in format (row, col, n_rows, n_cols)
    :param second_bbox: bbox in format (row, col, n_rows, n_cols)
    :return: iou measure for two given bboxes
    """
    # your code here \/
    return 1
    # your code here /\


# =============================== 6 AUC ========================================
def calc_auc(pred_bboxes, gt_bboxes):
    """
    :param pred_bboxes: dict of bboxes in format {filename: detections}
        detections is a N x 5 array, where N is number of detections. Each
        detection is described using 5 numbers: [row, col, n_rows, n_cols,
        confidence].
    :param gt_bboxes: dict of bboxes in format {filenames: bboxes}. bboxes is a
        list of tuples in format (row, col, n_rows, n_cols)
    :return: auc measure for given detections and gt
    """
    # your code here \/
    return 1
    # your code here /\


# =============================== 7 NMS ========================================
def nms(detections_dictionary, iou_thr):
    """
    :param detections_dictionary: dict of bboxes in format {filename: detections}
        detections is a N x 5 array, where N is number of detections. Each
        detection is described using 5 numbers: [row, col, n_rows, n_cols,
        confidence].
    :param iou_thr: IoU threshold for nearby detections
    :return: dict in same format as detections_dictionary where close detections
        are deleted
    """
    # your code here \/
    return {}
    # your code here /\

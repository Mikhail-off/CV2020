import keras
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from keras.models import Sequential
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

# ============================== 1 Classifier model ============================
def get_cls_model(input_shape):
    """
    :param input_shape: tuple (n_rows, n_cols, n_channgels)
            input shape of image for classification
    :return: nn model for classification
    """
    model = Sequential([
        Conv2D(filters=8, kernel_size=3, activation=None, input_shape=(40, 100, 1)),
        Conv2D(filters=8, kernel_size=3, activation='relu'),
        MaxPooling2D(),
        Conv2D(filters=16, kernel_size=3, activation='relu'),
        MaxPooling2D(),
        Conv2D(filters=32, kernel_size=3, activation='relu'),
        MaxPooling2D(),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(2, activation='softmax')
    ])
    model.summary()
    model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=1e-4), metrics=["accuracy"])
    return model


def fit_cls_model(X, y):
    """
    :param X: 4-dim ndarray with training images
    :param y: 2-dim ndarray with one-hot labels for training
    :return: trained nn model
    """

    model = get_cls_model((40, 100, 1))
    image_generator = ImageDataGenerator(horizontal_flip=True)
    image_generator.fit(X)
    batch_size = 64
    steps_per_epoch = (len(X) - 1) // batch_size + 1
    model.fit_generator(image_generator.flow(X, y, batch_size=batch_size),
                        steps_per_epoch=steps_per_epoch,
                        epochs=25)
    #model.save('classifier_model.h5')
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
    return cls_model


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
def calc_square(points):
    return max(0, points[2] - points[0]) * max(0, points[3] - points[1])

def calc_iou(first_bbox, second_bbox):
    """
    :param first_bbox: bbox in format (row, col, n_rows, n_cols)
    :param second_bbox: bbox in format (row, col, n_rows, n_cols)
    :return: iou measure for two given bboxes
    """
    def bbox2quadrangle(bbox):
        quadrangle = bbox.copy()
        quadrangle[2] += quadrangle[0]
        quadrangle[3] += quadrangle[1]
        return quadrangle

    first_points = first_bbox.copy()
    first_points[2] = first_points[2] + first_points[0]
    first_points[3] = first_points[3] + first_points[1]
    second_points = second_bbox.copy()
    second_points[2] = second_bbox[2] + second_bbox[0]
    second_points[3] = second_bbox[3] + second_bbox[1]

    I_points = [
        max(first_points[0], second_points[0]),
        max(first_points[1], second_points[1]),
        min(first_points[2], second_points[2]),
        min(first_points[3], second_points[3])
    ]

    return calc_square(I_points) / (calc_square(first_points) + calc_square(first_points) - calc_square(I_points))


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

import numpy as np
from scipy.signal import convolve2d

from skimage.transform import resize
from skimage.io import imread, imshow

from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import classification_report

MAX_ITEMS = 10**10
HOG_RANGE = (0, np.pi)
EPS = 1e-6

BIN_COUNT = 12
IMG_SIZE = (64, 64)
CELL_ROWS,  CELL_COLS = 4, 4
BLOCK_ROWS, BLOCK_COLS = 4, 4


def calc_gradient(img):
    dx_c = np.zeros_like(img)
    dy_c = np.zeros_like(img)
    n_channels = img.shape[-1]

    x_kernel = np.array([[-1, 0, 1],
                         [-2, 0, 2],
                         [-1, 0, 1]])
    y_kernel = x_kernel.T
    
    for channel in range(n_channels):
        dx_c[:, :, channel] = convolve2d(img[:, :, channel], y_kernel, mode='same', boundary='symm')
        dy_c[:, :, channel] = convolve2d(img[:, :, channel], y_kernel, mode='same', boundary='symm')
    
    dx = np.sum(dx_c, axis=-1)
    dy = np.sum(dx_c, axis=-1)
    
    grad_magnitude = np.sqrt(dx**2 + dy**2)
    
    grad_angle = np.arctan2(dx, dy)
    #grad_angle[grad_angle < HOG_RANGE[0]] += np.pi

    return grad_magnitude, grad_angle


def calc_hist(img, cell_rows, cell_cols, bin_count):
    num_cells_h = img.shape[0] // cell_rows
    num_cells_w = img.shape[1] // cell_cols
    
    bin_hist = np.zeros((num_cells_h, num_cells_w, bin_count))
    
    grad_magnitude, grad_angle = calc_gradient(img)
    
    for y in range(num_cells_h):
        for x in range(num_cells_w):
            y_left = y * cell_rows
            y_right = y_left + cell_rows
            x_left = x * cell_cols
            x_right = x_left + cell_cols

            cell_magnitudes = grad_magnitude[y_left: y_right, x_left: x_right].flatten()
            cell_angles = grad_angle[y_left: y_right, x_left: x_right].flatten()

            bin_hist[y, x], _ = np.histogram(cell_angles, bin_count, range=HOG_RANGE, weights=cell_magnitudes)
    
    return bin_hist


def get_blocks(img, block_row, block_col, cell_rows, cell_cols, bin_count):
    hist = calc_hist(img, cell_rows, cell_cols, bin_count)
    #print(hist.shape)
    num_block_h = img.shape[0] // (block_row * cell_rows) 
    num_block_w = img.shape[1] // (block_col * cell_cols)
    
    blocks = np.zeros((num_block_h, num_block_w, block_row * block_col * bin_count))
    
    for y in range(num_block_h):
        for x in range(num_block_w):
            y_left = y * block_row
            y_right = y_left + block_row
            x_left = x * block_col
            x_right = x_left + block_col
            block_elem = hist[y_left: y_right, x_left:x_right, :].flatten()
            blocks[y, x] = block_elem / (np.linalg.norm(block_elem) + EPS)
    
    return blocks
    

def extract_hog(img):
    preprocessed_image = np.array(resize(img, IMG_SIZE))
    hog_features = get_blocks(preprocessed_image, BLOCK_ROWS, BLOCK_COLS, CELL_ROWS, CELL_COLS, BIN_COUNT).flatten()
    return hog_features


def fit_and_classify(train_features, train_labels, test_features):
    if len(train_features) > MAX_ITEMS:
        train_features = train_features[:MAX_ITEMS]
        train_labels = train_labels[:MAX_ITEMS]
    print('--------------------------------------------------------------------------')
    print('Train shape:', train_features.shape, train_labels.shape)
    print('Test shape:', test_features.shape)
    model = LinearSVC(C=0.3, verbose=0)
    """
    x_train, x_test, y_train, y_test = train_test_split(train_features, train_labels, test_size=0.3)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    print(classification_report(y_test, y_pred))
    print('Average score is', (y_test == y_pred).mean())
    """
    model.fit(train_features, train_labels)
    print('--------------------------------------------------------------------------')
    return model.predict(test_features)

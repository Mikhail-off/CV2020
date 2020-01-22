import numpy as np
from skimage.transform import resize


MIN_IMAGE_SIZE = 500


def convert_to_coloted_img(img):
    assert len(img.shape) == 2
    height = img.shape[0] // 3
    return np.dstack((img[:height, :], img[height:2 * height, :], img[2 * height: 3 * height, :]))


def cut_image(image, percent=0.05):
    h, w = image.shape[:2]
    delta_h = int(h * percent)
    delta_w = int(w * percent)
    return image[delta_h: h - delta_h, delta_w: w - delta_w, :]


def mse(img_1, img_2):
    assert img_1.shape == img_2.shape
    return ((img_1 - img_2)**2).mean()


# возвращаем отрицательное значение, чтобы была задача минимизации
def cross_corelation(img_1, img_2):
    square_sum_1 = (img_1**2).sum()
    square_sum_2 = (img_2**2).sum()
    return -(img_1 * img_2).sum() / np.sqrt(square_sum_1 * square_sum_2)


def find_best_shift(img_1, img_2, metrics, shift_0=(0, 0), shift_range=(-15, 16)):
    assert img_1.shape == img_2.shape
    best_shift = shift_0
    best_metrics = None

    for y_shift in range(shift_0[0] + shift_range[0], shift_0[0] + shift_range[1]):
        for x_shift in range(shift_0[1] + shift_range[0], shift_0[1] + shift_range[1]):
            shifted_img_1 = img_1[max(0, -y_shift): min(img_1.shape[0], img_1.shape[0] - y_shift),
                                          max(0, -x_shift): min(img_1.shape[1], img_1.shape[1] - x_shift)]

            shifted_img_2 = img_2[max(0, y_shift): min(img_2.shape[0], img_2.shape[0] + y_shift),
                                          max(0, x_shift): min(img_2.shape[1], img_2.shape[1] + x_shift)]

            cur_metrics = metrics(shifted_img_1, shifted_img_2)
            if best_metrics is None or cur_metrics < best_metrics:
                best_metrics = cur_metrics
                best_shift = (y_shift, x_shift)

    return best_shift


def find_best_shift_pyramid(img_1, img_2, metrics):
    assert img_1.shape == img_2.shape
    cur_image_size = img_1.shape
    image_sizes = [cur_image_size]
    while cur_image_size[0] > MIN_IMAGE_SIZE or cur_image_size[1] > MIN_IMAGE_SIZE:
        cur_image_size = cur_image_size[0] // 2, cur_image_size[1] // 2
        image_sizes.append(cur_image_size)

    cur_img_1 = img_1
    cur_img_2 = img_2
    images = [(img_1, img_2)]
    for sz in image_sizes[1:]:
        images += [(resize(cur_img_1, sz), resize(cur_img_2, sz))]
        cur_img_1, cur_img_2 = images[-1]

    image_sizes = image_sizes[::-1]
    images = images[::-1]


    best_shift = find_best_shift(images[0][0], images[0][1], metrics)
    for (img_1, img_2) in images[1:]:
        best_shift = 2 * best_shift[0], 2 * best_shift[1]
        best_shift = find_best_shift(img_1, img_2, metrics, shift_0=best_shift, shift_range=(-2, 3))

    return best_shift


def roll_image(image, shift):
    img = np.roll(image, shift=-shift[0], axis=0)
    img = np.roll(img, shift=-shift[1], axis=1)
    return img


def align(img, g_coords):
    colored_img = convert_to_coloted_img(img)
    height, width = colored_img.shape[:2]
    colored_img = cut_image(colored_img)

    b_img, g_img, r_img = colored_img[:, :, 0], colored_img[:, :, 1], colored_img[:, :, 2]

    metrics = cross_corelation
    r_shift = find_best_shift_pyramid(g_img, r_img, metrics=metrics)
    b_shift = find_best_shift_pyramid(g_img, b_img, metrics=metrics)

    r_img = roll_image(r_img, shift=r_shift)
    b_img = roll_image(b_img, shift=b_shift)

    res_image = (np.dstack((r_img, g_img, b_img)) * 255).astype(np.uint8)

    r_coords = g_coords[0] + height + r_shift[0], g_coords[1] + r_shift[1]
    b_coords = g_coords[0] - height + b_shift[0], g_coords[1] + b_shift[1]

    return res_image, b_coords, r_coords

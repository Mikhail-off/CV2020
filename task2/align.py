import numpy as np

MIN_SHIFT = -15
MAX_SHIFT = 16

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


# return minus
def cross_corelation(img_1, img_2):
    square_sum_1 = (img_1**2).sum()
    square_sum_2 = (img_2**2).sum()
    return -(img_1 * img_2).sum() / np.sqrt(square_sum_1 * square_sum_2)


from PIL import Image
from time import sleep

def find_best_shift(img_1, img_2, metrics):
    best_shift = (0, 0)
    best_metrics = None

    #sleep(5)
    for y_shift in range(MIN_SHIFT, MAX_SHIFT):
        for x_shift in range(MIN_SHIFT, MAX_SHIFT):
            shifted_img_1 = img_1[max(0, -y_shift): min(img_1.shape[0], img_1.shape[0] - y_shift),
                                          max(0, -x_shift): min(img_1.shape[1], img_1.shape[1] - x_shift)]

            #shifted_img_2 = np.roll(img_2, shift=(y_shift, x_shift))
            shifted_img_2 = img_2[max(0, y_shift): min(img_2.shape[0], img_2.shape[0] + y_shift),
                                          max(0, x_shift): min(img_2.shape[1], img_2.shape[1] + x_shift)]

            cur_metrics = metrics(shifted_img_1, shifted_img_2)
            if best_metrics is None or cur_metrics < best_metrics:
                best_metrics = cur_metrics
                best_shift = (y_shift, x_shift)
                #shifted_img_1 = img_1
                #shifted_img_2 = img_2
                image_1 = Image.fromarray((shifted_img_1 * 255).astype(np.uint8))
                image_2 = Image.fromarray((shifted_img_2 * 255).astype(np.uint8))
                image_1.save('D:\\temp\\1.png')
                image_2.save('D:\\temp\\2.png')

    #print('finish')
    #print(best_shift)
    #sleep(5)
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
    r_shift = find_best_shift(g_img, r_img, metrics=metrics)
    b_shift = find_best_shift(g_img, b_img, metrics=metrics)

    r_img = roll_image(r_img, shift=r_shift)
    b_img = roll_image(b_img, shift=b_shift)

    res_image = (np.dstack((r_img, g_img, b_img)) * 255).astype(np.uint8)

    r_coords = g_coords[0] + height + r_shift[0], g_coords[1] + r_shift[1]
    b_coords = g_coords[0] - height + b_shift[0], g_coords[1] + b_shift[1]
    #print(b_coords)
    #print(g_coords)
    #print(r_coords)
    return res_image, b_coords, r_coords
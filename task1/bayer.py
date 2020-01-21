import numpy as np


def get_bayer_masks(n_rows, n_cols):
    cell = np.zeros((2, 2, 3), dtype=np.bool)
    cell[0, 1, 0] = 1
    cell[0, 0, 1] = 1
    cell[1, 1, 1] = 1
    cell[1, 0, 2] = 1

    reps = (n_rows + 1) // 2, (n_cols + 1) // 2, 1
    mask = np.tile(cell, reps)[:n_rows, :n_cols, :]
    return mask


def get_colored_img(raw_img):
    mask = get_bayer_masks(*raw_img.shape[:2])
    colored_img = np.zeros_like(mask, dtype=raw_img.dtype)
    for i in range(3):
        colored_img[:, :, i] = raw_img * mask[:, :, i]
    return colored_img


def bilinear_interpolation(colored_img):
    pad_width = ((1, 1),) * (len(colored_img.shape) - 1) + ((0, 0),)
    padded_img = np.pad(colored_img, pad_width=pad_width, mode='constant', constant_values=0)

    mask = get_bayer_masks(*colored_img.shape[:2])
    mask = np.pad(mask, pad_width=pad_width, mode='constant', constant_values=0)

    interpolated_image = np.copy(colored_img)

    for i in range(1, padded_img.shape[0] - 1):
        for j in range(1, padded_img.shape[1] - 1):
            for k in range(padded_img.shape[2]):
                if mask[i, j, k]:
                    continue

                pixel_sum = padded_img[i - 1:i + 2, j - 1:j + 2, k].sum()
                n_known = mask[i - 1:i + 2, j - 1:j + 2, k].sum()
                interpolated_image[i - 1, j - 1, k] = pixel_sum / n_known

    return interpolated_image


def improved_interpolation(raw_img):
    pad_width = ((2, 2),) * (len(raw_img.shape) - 1) + ((0, 0),)
    padded_img = np.pad(raw_img, pad_width=pad_width, mode='constant', constant_values=0)

    mask = get_bayer_masks(*raw_img.shape[:2])
    mask = np.pad(mask, pad_width=pad_width, mode='constant', constant_values=0)

    interpolated_image = np.copy(raw_img).astype(int)

    for i in range(2, padded_img.shape[0] - 2):
        for j in range(2, padded_img.shape[1] - 2):
            for k in range(padded_img.shape[2]):
                if mask[i, j, k]:
                    continue


    return interpolated_image
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
    pass

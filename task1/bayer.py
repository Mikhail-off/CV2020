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
    raw_img = get_colored_img(raw_img)
    pad_width = ((2, 2),) * (len(raw_img.shape) - 1) + ((0, 0),)
    padded_img = np.pad(raw_img, pad_width=pad_width, mode='constant', constant_values=0)

    mask = get_bayer_masks(*raw_img.shape[:2])
    mask = np.pad(mask, pad_width=pad_width, mode='constant', constant_values=0)

    interpolated_image = np.copy(raw_img).astype(int)

    g_at_r = np.zeros((5, 5, 3), dtype=float)
    g_at_r[:, :, 0] = np.array([[0, 0, -1, 0, 0],
                                [0, 0, 0, 0, 0],
                                [-1, 0, 4, 0, -1],
                                [0, 0, 0, 0, 0],
                                [0, 0, -1, 0, 0]])
    g_at_r[:, :, 1] = np.array([[0, 0, 0, 0, 0],
                                [0, 0, 2, 0, 0],
                                [0, 2, 0, 2, 0],
                                [0, 0, 2, 0, 0],
                                [0, 0, 0, 0, 0]])
    g_at_b = np.zeros_like(g_at_r)
    g_at_b[:, :, 1] = g_at_r[:, :, 1]
    g_at_b[:, :, 2] = g_at_r[:, :, 0]


    r_at_g_in_r_row = np.zeros_like(g_at_b)
    r_at_g_in_r_row[:, :, 0] = np.array([[0, 0, 0, 0, 0],
                                         [0, 0, 0, 0, 0],
                                         [0, 4, 0, 4, 0],
                                         [0, 0, 0, 0, 0],
                                         [0, 0, 0, 0, 0]])
    r_at_g_in_r_row[:, :, 1] = np.array([[0, 0, 0.5, 0, 0],
                                         [0, -1, 0, -1, 0],
                                         [-1, 0, 5, 0, -1],
                                         [0, -1, 0, -1, 0],
                                         [0, 0, 0.5, 0, 0]])

    r_at_g_in_b_row = np.zeros_like(g_at_b)
    r_at_g_in_b_row[:, :, 0] = r_at_g_in_r_row[:, :, 0].T
    r_at_g_in_b_row[:, :, 1] = r_at_g_in_r_row[:, :, 1].T

    r_at_b = np.zeros_like(g_at_b)
    r_at_b[:, :, 0] = np.array([[0, 0, 0, 0, 0],
                                [0, 2, 0, 2, 0],
                                [0, 0, 0, 0, 0],
                                [0, 2, 0, 2, 0],
                                [0, 0, 0, 0, 0]])
    r_at_b[:, :, 2] = np.array([[0, 0, -1.5, 0, 0],
                                [0, 0, 0, 0, 0],
                                [-1.5, 0, 6, 0, -1.5],
                                [0, 0, 0, 0, 0],
                                [0, 0, -1.5, 0, 0]])

    b_at_g_in_b_row = np.zeros_like(g_at_b)
    b_at_g_in_b_row[:, :, 1] = r_at_g_in_r_row[:, :, 1]
    b_at_g_in_b_row[:, :, 2] = r_at_g_in_r_row[:, :, 0]

    b_at_g_in_r_row = np.zeros_like(g_at_b)
    b_at_g_in_r_row[:, :, 1] = r_at_g_in_b_row[:, :, 1]
    b_at_g_in_r_row[:, :, 2] = r_at_g_in_b_row[:, :, 0]

    b_at_r = np.zeros_like(g_at_b)
    b_at_r[:, :, 0] = r_at_b[:, :, 2]
    b_at_r[:, :, 2] = r_at_b[:, :, 0]


    for i in range(2, padded_img.shape[0] - 2):
        for j in range(2, padded_img.shape[1] - 2):
            for k in range(padded_img.shape[2]):
                if mask[i, j, k]:
                    continue

                kernel = np.zeros((5, 5, 3), dtype=int)
                kernel[2, 2, :] = np.array([1, 1, 1])
                # r
                if k == 0:
                    if mask[i, j, 1] and mask[i, j - 1, 0] and mask[i, j + 1, 0]:
                        kernel = r_at_g_in_r_row
                    elif mask[i, j, 1] and mask[i, j - 1, 2] and mask[i, j + 1, 2]:
                        kernel = r_at_g_in_b_row
                    elif mask[i, j, 2]:
                        kernel = r_at_b
                # g
                elif k == 1:
                    if mask[i, j, 0]:
                        kernel = g_at_r
                    elif mask[i, j, 2]:
                        kernel = g_at_b
                # b
                elif k == 2:
                    if mask[i, j, 1] and mask[i, j - 1, 2] and mask[i, j + 1, 2]:
                        kernel = b_at_g_in_b_row
                    elif mask[i, j, 1] and mask[i, j - 1, 0] and mask[i, j + 1, 0]:
                        kernel = b_at_g_in_r_row
                    elif mask[i, j, 0]:
                        kernel = b_at_r

                else:
                    assert False
                pixel_sum = (kernel * padded_img[i - 2: i + 3, j - 2: j + 3, :]).sum()
                norm_coeff = kernel.sum()
                interpolated_image[i - 2, j - 2, k] = pixel_sum / norm_coeff


    interpolated_image = np.clip(interpolated_image, 0, 255).astype(np.uint8)

    return interpolated_image


def compute_psnr(img_pred, img_gt):
    img_gt = img_gt.astype(np.float64)
    img_pred = img_pred.astype(np.float64)
    mse = ((img_gt - img_pred)**2).sum() / img_pred.size

    if mse == 0:
        raise ValueError

    psnr = 10 * np.log10(img_gt.max()**2 / mse)
    return psnr

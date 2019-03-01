import image
import blocks
import matplotlib.pyplot as plt
import numpy as np


def upsample(arr, up_h, up_w):
    """ Duplicate rows `up_h` times and columns `up_w` times

    ex:
    upsample( [[1,2],[3,4], 2 ,3 )
    ->[[1, 1, 1, 2, 2, 2],
       [1, 1, 1, 2, 2, 2],
       [3, 3, 3, 4, 4, 4],
       [3, 3, 3, 4, 4, 4]]
    """

    return np.array(np.repeat(
        [list(np.repeat(row, up_w)) for row in arr],
        repeats=up_h, axis=0))


def scheme_subsample(ycbcr_img, scheme):
    if scheme == (4, 2, 0):
        SSH, SSW = 2, 2
        Y_img = ycbcr_img[:, :, 0]
        Cb_img = ycbcr_img[::SSH, ::SSW, 1]
        Cr_img = ycbcr_img[::SSH, ::SSW, 2]
    else:
        raise NotImplementedError

    return {'Y': Y_img, 'Cb': Cb_img, 'Cr': Cr_img, 'SSH': SSH, 'SSW': SSW, 'scheme': scheme}


def upsample_and_assemble(subsampled_object):
    if subsampled_object['scheme'] == (4, 2, 0):
        SSH, SSW = 2, 2
        up_Y = subsampled_object['Y']
        up_Cb = upsample(subsampled_object['Cb'], SSH, SSW)
        up_Cr = upsample(subsampled_object['Cr'], SSH, SSW)
    else:
        raise NotImplementedError

    assembled_array = np.zeros((up_Y.shape) + (3,))

    assembled_array[:, :, 0] = up_Y[:, :]
    assembled_array[:, :, 1] = up_Cb[:, :]
    assembled_array[:, :, 2] = up_Cr[:, :]

    return assembled_array


def subsample(arr, SSH, SSW):
    return arr[::SSH, ::SSW]


if __name__ == "__main__":
    pass

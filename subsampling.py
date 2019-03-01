import image
import blocks
import matplotlib.pyplot as plt
import numpy as np


def upsample(arr, up_h, up_w) -> []:
    """retourner un array qui a la shape (2d_arr.shape[0] * up_h, 2d_arr.shape[1] * up[w])
    et remplir le vide avec des 0.
    """

    return np.repeat(
        [list(np.repeat(row, up_w)) for row in arr],
        repeats=up_h, axis=0)


if __name__ == "__main__":
    img = image.get_test_image()
    green_channel = img[:,:,1]

    # Our friendly block (30,40)
    green_8x8_blocks = blocks.split_8x8(green_channel)
    that_block_green = green_8x8_blocks[30,40]
    plt.imshow(that_block_green, cmap=plt.get_cmap('Greens'))
    plt.title("green 8x8 block (30,40)")
    plt.show()

    ########### THIS IS THE SUBSAMPLING PART
    # NOTE We are not going to subsample the green, we're going to subsample the
    # from the YCbCr
    green_channel = green_channel[::2,::2]
    green_8x8_blocks = blocks.split_8x8(green_channel)

    # Print the sub sampled block, just the part corresponding
    # to the block above
    that_block_green = green_8x8_blocks[15,20]
    that_block_sub_part = that_block_green[:4,:4]
    plt.imshow(that_block_sub_part, cmap=plt.get_cmap('Greens'))
    plt.title("upper left part of green subsampled 8x8 block (15,20)")
    plt.show()

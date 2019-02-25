from image import open_image_as_ndarray, get_test_image
import numpy as np


def split_8x8(img):
    rows = np.array(np.split(img, 75, 0))
    blocks = np.array(
        [np.split(row, 75, 1) for row in rows]
    )
    print(blocks.shape)
    return blocks

def open_image_as_8x8_blocks(filename):
    return split_8x8(open_image_as_ndarray(filename))

def split_rgb(blocks):
    red_blocks = blocks[:,:,:,:,0]
    green_blocks = blocks[:,:,:,:,1]
    blue_blocks = blocks[:,:,:,:,2]
    return red_blocks, green_blocks, blue_blocks

def get_test_image_as_8x8_blocks():
    img = get_test_image()
    return split_8x8(img)


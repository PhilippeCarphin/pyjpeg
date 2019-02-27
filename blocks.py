from image import open_image_as_ndarray, get_test_image
import numpy as np

N = 8

def split_NxN(img):
    h = img.shape[0]
    w = img.shape[1]
    rows = np.array(np.split(img, h//N, 0))
    blocks = np.array(
        [np.split(row, w//N, 1) for row in rows]
    )
    return blocks

def open_image_as_NxN_blocks(filename):
    return split_NxN(open_image_as_ndarray(filename))

def split_rgb(blocks):
    red_blocks = blocks[:,:,:,:,0]
    green_blocks = blocks[:,:,:,:,1]
    blue_blocks = blocks[:,:,:,:,2]
    return red_blocks, green_blocks, blue_blocks

def get_test_image_as_NxN_blocks():
    img = get_test_image()
    return split_NxN(img)

def get_one_test_NxN_rgb_block():
    return get_test_image_as_NxN_blocks()[30,40,:,:,:]

def get_one_test_NxN_block():
    return get_one_test_NxN_rgb_block()[:,:,0]

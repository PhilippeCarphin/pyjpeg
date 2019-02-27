import numpy as np
# import scipy.fftpack as dctpack
from skimage import io
import matplotlib.pyplot as plt

"""
Toutes des vieilles affaires pas rapport
"""

img = io.imread('input_image.png')
height = img.shape[0]
width = img.shape[1]

assert(height % 8 == 0 and width % 8 == 0), "on a juste à traiter des images 8x8"

# block i = 30, j =40
a_block = img[30 * 8:30*8 +8, 40*8:40*8 + 8, :]

blocks_h = height // 8
blocks_w = width // 8

i_axis = 0
j_axis = 1

rows = np.array(np.split(img, 75, i_axis))
row_of_blocks = np.split(rows[30], 75, j_axis)
plt.imshow(rows[30])
plt.show()

assert np.array_equal(row_of_blocks[40], a_block), "Should be equal"

plt.imshow(row_of_blocks[40])
plt.show()

rows = np.array(np.split(img, 75, i_axis))
blocks = np.array(
    [np.split(row, 75, j_axis) for row in rows]
)


plt.imshow(blocks[30,40])
plt.show()


# Could have split red, green, blue before splitting into blocks but whatever
red_blocks = blocks[:,:,:,:,0]
green_blocks = blocks[:,:,:,:,1]
blue_blocks = blocks[:,:,:,:,2]

print(red_blocks.shape)

plt.imshow(red_blocks[30,40], cmap = plt.get_cmap('gray_r'))
plt.show()
plt.imshow(green_blocks[30,40], cmap = plt.get_cmap('gray_r'))
plt.show()
plt.imshow(blue_blocks[30,40], cmap = plt.get_cmap('gray_r'))
plt.show()








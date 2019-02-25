import numpy as np
import scipy.fftpack as dctpack
from skimage import io
import matplotlib.pyplot as plt
####################### DCT TOWN ###################################

# RECAP : Opening the image and splitting it into blocks
img = (io.imread('input_image.png').astype('int16') - 128).astype('int8')
rows = np.array(np.split(img, 75, 0))
blocks = np.array(
    [np.split(row, 75, 1) for row in rows]
)
red_blocks = blocks[:,:,:,:,0]
green_blocks = blocks[:,:,:,:,1]
blue_blocks = blocks[:,:,:,:,2]

my_block = red_blocks[30,40]
plt.imshow(my_block, cmap=plt.get_cmap('gray'))
plt.show()
print(my_block)

# Actual DCT
first_pass_dct = dctpack.dct(my_block, axis=0, norm='ortho')
dct_block = dctpack.dct(first_pass_dct, axis=1, norm='ortho')

print(dct_block)
plt.imshow(dct_block.astype('uint8'), cmap=plt.get_cmap('gray'))
plt.show()

first_pass_idct = dctpack.idct(dct_block, axis=0, norm='ortho')
idct_block = dctpack.idct(first_pass_idct, axis=1, norm='ortho')

# print(idct_block)
plt.imshow(idct_block, cmap=plt.get_cmap('gray'))
plt.show()


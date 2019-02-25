from image import open_image_as_ndarray
import blocks
import numpy as np
import scipy.fftpack as dctpack
import matplotlib.pyplot as plt

def encode_dct(block_8x8):
    first_pass_dct = dctpack.dct(block_8x8, axis=0, norm='ortho')
    dct_block_8x8 = dctpack.dct(first_pass_dct, axis=1, norm='ortho')
    return dct_block_8x8

def decode_dct(dct_block_8x8):
    first_pass_idct = dctpack.idct(dct_block_8x8, axis=0, norm='ortho')
    idct_block_8x8 = dctpack.idct(first_pass_idct, axis=1, norm='ortho')
    return idct_block_8x8

def dct_basis_element(i,j):
    block = np.zeros((8,8))
    block[i,j] = 1
    first_pass_idct = dctpack.idct(block, axis=0, norm='ortho')
    idct_block = dctpack.idct(first_pass_idct, axis=1, norm='ortho')
    return idct_block

def show_basis_element(i,j):
    basis = dct_basis_element(i,j)
    plt.imshow(basis, cmap = plt.get_cmap('gray'))
    plt.show()
    return basis

def shift_image_by_128_for_dct(image):
    return (image.astype('int16') - 128).astype('int8')

if __name__ == "__main__":
    the_blocks = blocks.get_test_image_as_8x8_blocks()
    the_blocks = shift_image_by_128_for_dct(the_blocks)
    red_blocks, green_blocks, blue_blocks = blocks.split_rgb(the_blocks)


    ################### 1 Bloc original
    my_block = red_blocks[30,40]
    print("Bloc original\n", my_block)
    plt.imshow(my_block, cmap=plt.get_cmap('gray'))
    plt.show()

    ################### 2 Bloc encodé
    encoded_block = encode_dct(my_block)
    print("Bloc encodé\n", encoded_block)
    print(encoded_block.dtype)
    plt.imshow(encoded_block, cmap=plt.get_cmap('gray'))
    plt.show()

    ################### 3 Combinaison Linéaire avec deux coefficients
    # On fait une combinaison linéaire avec les coefficients du coin en haut à
    # gauche qui ont l'air les plux forts
    comb = np.zeros((8,8))
    b1 = dct_basis_element(0,1)
    comb += b1 * encoded_block[0,1] / 255
    b2 = dct_basis_element(1,0)
    comb += b2 * encoded_block[1,0] / 255
    plt.imshow(comb, cmap=plt.get_cmap('gray'))
    plt.show()

    ################### 4 Combinaison Linéaire avec 16 coefficients
    comb = np.zeros((8,8))
    for i in range(4):
        for j in range(4):
            comb += dct_basis_element(i,j) * encoded_block[i,j] / 255
    plt.imshow(comb, cmap=plt.get_cmap('gray'))
    plt.show()

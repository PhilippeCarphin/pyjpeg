from image import open_image_as_ndarray
import blocks
import numpy as np
import scipy.fftpack as dctpack
import matplotlib.pyplot as plt
import quantize

def encode_dct(block_8x8):
    first_pass_dct = dctpack.dct(block_8x8, axis=0, norm='ortho')
    dct_block_8x8 = dctpack.dct(first_pass_dct, axis=1, norm='ortho')
    return dct_block_8x8

def decode_dct(dct_block_8x8):
    first_pass_idct = dctpack.idct(dct_block_8x8, axis=0, norm='ortho')
    idct_block_8x8 = dctpack.idct(first_pass_idct, axis=1, norm='ortho')
    return idct_block_8x8

def dct_basis_element(i,j):
    """ Returns the inverse discrete cosine transform of a canonical basis
    element (i.e. a vector with a '1' in a single position and zeroes everywhere
    else."""
    block = np.zeros((8,8))
    block[i,j] = 1
    first_pass_idct = dctpack.idct(block, axis=0, norm='ortho')
    idct_block = dctpack.idct(first_pass_idct, axis=1, norm='ortho')
    return idct_block

def dct_encode_blocks(blocks):
    # Le astype('float') est ben important ici, sinon les
    # tests ont l'air de pas marcher
    n_blocks_h = blocks.shape[0]
    n_blocks_w = blocks.shape[1]
    dct_blocks = np.empty_like(blocks).astype('float')
    for i in range(n_blocks_h):
        for j in range(n_blocks_w):
            dct_blocks[i,j,:,:] = encode_dct(blocks[i,j,:,:])
    return dct_blocks


if __name__ == "__main__":
    the_blocks = blocks.get_test_image_as_8x8_blocks()
    red_blocks, green_blocks, blue_blocks = blocks.split_rgb(the_blocks)

    ################### Bloc original
    my_block = red_blocks[30,40]
    print("Bloc original\n", my_block)
    plt.imshow(my_block, cmap=plt.get_cmap('gray'))
    plt.title("Bloc")
    plt.show()

    ################### Bloc encodé
    encoded_block = encode_dct(my_block)
    print("Bloc encodé :\n", encoded_block)
    plt.imshow(encoded_block, cmap=plt.get_cmap('gray'))
    plt.title("Bloc > dct")
    plt.show()

    ################### Bloc DCT Quantifié
    quantized_block = quantize.quantize_one_block(encoded_block)
    print("Bloc encodé, quantifié :\n", quantized_block)
    plt.imshow(quantized_block, cmap=plt.get_cmap('gray'))
    plt.title("Bloc > dct > quantifié")
    plt.show()

    ################### Combinaison Linéaire avec deux coefficients
    # On fait une combinaison linéaire avec les coefficients du coin en haut à
    # gauche qui ont l'air les plux forts
    comb_lin = np.zeros((8,8))
    comb_lin += dct_basis_element(0,1) * encoded_block[0,1] / 255
    comb_lin += dct_basis_element(1,0) * encoded_block[1,0] / 255
    plt.imshow(comb_lin, cmap=plt.get_cmap('gray'))
    plt.title("Bloc > dct > quantification AGRESSIVE > idct")
    plt.show()

    ################### Combinaison Linéaire avec 16 coefficients
    comb_lin = np.zeros((8,8))
    for i in range(4):
        for j in range(4):
            comb_lin += dct_basis_element(i,j) * encoded_block[i,j] / 255
    plt.imshow(comb_lin, cmap=plt.get_cmap('gray'))
    plt.title("Bloc > dct > quantification moins agressive > idct")
    plt.show()

    idct_block = decode_dct(quantized_block)
    plt.imshow(idct_block, cmap=plt.get_cmap('gray'))
    plt.title("Bloc > dct > la vraie quantification > idct")
    plt.show()

    idct_block = decode_dct(encoded_block)
    plt.imshow(idct_block, cmap=plt.get_cmap('gray'))
    plt.title("Bloc > dct > PAS de quantification > idct")
    plt.show()



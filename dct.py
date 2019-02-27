from image import open_image_as_ndarray
import blocks
import numpy as np
import scipy.fftpack as dctpack
import matplotlib.pyplot as plt
import quantize

def encode_dct(block_NxN):
    first_pass_dct = dctpack.dct(block_NxN, axis=0, norm='ortho')
    dct_block_NxN = dctpack.dct(first_pass_dct, axis=1, norm='ortho')
    return dct_block_NxN

def decode_dct(dct_block_NxN):
    first_pass_idct = dctpack.idct(dct_block_NxN, axis=0, norm='ortho')
    idct_block_NxN = dctpack.idct(first_pass_idct, axis=1, norm='ortho')
    return idct_block_NxN

def dct_basis_element(i,j):
    """ Returns the inverse discrete cosine transform of a canonical basis
    element (i.e. a vector with a '1' in a single position and zeroes everywhere
    else."""
    block = np.zeros((blocks.N,blocks.N))
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
    the_blocks = blocks.get_test_image_as_NxN_blocks()
    red_blocks, green_blocks, blue_blocks = blocks.split_rgb(the_blocks)

    ################### Bloc original
    my_block = red_blocks[30,40]
    print("Bloc original\n", my_block)
    plt.imshow(my_block, cmap=plt.get_cmap('gray_r'))
    plt.title("Bloc")
    plt.show()

    ################### Bloc encodé
    encoded_block = encode_dct(my_block)
    print("Bloc encodé :\n", encoded_block)
    plt.imshow(encoded_block, cmap=plt.get_cmap('gray_r'))
    plt.title("Bloc > dct")
    plt.show()

    ################### Bloc DCT Quantifié
    quantized_block = quantize.quantize_one_block(encoded_block)
    print("Bloc encodé, quantifié :\n", quantized_block)
    plt.imshow(quantized_block, cmap=plt.get_cmap('gray_r'))
    plt.title("Bloc > dct > quantifié")
    plt.show()

    fig = plt.figure(figsize=(4,4))
    fig.add_subplot(2,2,1)
    plt.imshow(dct_basis_element(0,0), cmap=plt.get_cmap('gray_r'))

    fig.add_subplot(2,2,2)
    plt.imshow(dct_basis_element(0,1), cmap=plt.get_cmap('gray_r'))

    fig.add_subplot(2,2,3)
    plt.imshow(dct_basis_element(1,0), cmap=plt.get_cmap('gray_r'))

    fig.add_subplot(2,2,4)
    plt.imshow(dct_basis_element(1,1), cmap=plt.get_cmap('gray_r'))
    plt.show()

    ################### Combinaison Linéaire avec deux coefficients
    # On fait une combinaison linéaire avec les coefficients du coin en haut à
    # gauche qui ont l'air les plux forts
    comb_lin = np.zeros((blocks.N,blocks.N))
    comb_lin += dct_basis_element(0,1) * encoded_block[0,1] / 255
    comb_lin += dct_basis_element(1,0) * encoded_block[1,0] / 255
    plt.imshow(comb_lin, cmap=plt.get_cmap('gray_r'))
    plt.title("Bloc > dct > quantification AGRESSIVE > idct")
    plt.show()

    ################### Combinaison Linéaire avec 16 coefficients
    fig = plt.figure(figsize=(blocks.N,blocks.N))
    plt.title("16 vecteurs de la base DCT")
    for k in range(16):
        i = k // 4;
        j = k % 4;
        fig.add_subplot(4,4,k+1)
        plt.imshow(dct_basis_element(i,j), cmap=plt.get_cmap('gray'))
    plt.show()
    i = 0
    j = 0
    print(f'bloc_dct({i},{j}) : \n{dct_basis_element(i,j)}')
    i = 1
    j = 0
    print(f'bloc_dct({i},{j}) : \n{dct_basis_element(i,j)}')

    comb_lin = np.zeros((blocks.N,blocks.N))
    for i in range(4):
        for j in range(4):
            comb_lin += dct_basis_element(i,j) * encoded_block[i,j] / 255
    plt.imshow(comb_lin, cmap=plt.get_cmap('gray_r'))
    plt.title("Bloc > dct > quantification moins agressive > idct")
    plt.show()

    idct_block = decode_dct(quantized_block)
    plt.imshow(idct_block, cmap=plt.get_cmap('gray_r'))
    plt.title("Bloc > dct > la vraie quantification > idct")
    plt.show()

    idct_block = decode_dct(encoded_block)
    plt.imshow(idct_block, cmap=plt.get_cmap('gray_r'))
    plt.title("Bloc > dct > PAS de quantification > idct")
    plt.show()


# Accent, Accent_r, Blues, Blues_r, BrBG, BrBG_r, BugqGn, BuGn_r, BuPu, BuPu_r,\
# CMRmap, CMRmap_r, Dark2, Dark2_r, GnBu, GnBu_r, Greens, Greens_r, Greys,\
# Greys_r, OrRd, OrRd_r, Oranges, Oranges_r, PRGn, PRGn_r, Paired, Paired_r,\
# Pastel1, Pastel1_r, Pastel2, Pastel2_r, PiYG, PiYG_r, PuBu, PuBuGn, PuBuGn_r,\
# PuBu_r, PuOr, PuOr_r, PuRd, PuRd_r, Purples, Purples_r, RdBu, RdBu_r, RdGy,\
# RdGy_r, RdPu, RdPu_r, RdYlBu, RdYlBu_r, RdYlGn, RdYlGn_r, Reds, Reds_r, Set1,\
# Set1_r, Set2, Set2_r, Set3, Set3_r, Spectral, Spectral_r, Wistia, Wistia_r, \
# YlGn, YlGnBu, YlGnBu_r, YlGn_r, YlOrBr, YlOrBr_r, YlOrRd, YlOrRd_r, afmhot, \
# afmhot_r, autumn, autumn_r, binary, binary_r, bone, bone_r, brg, brg_r, bwr, \
# bwr_r, cividis, cividis_r, cool, cool_r, coolwarm, coolwarm_r, copper, copper_r,\
# cubehelix, cubehelix_r, flag, flag_r, gist_earth, gist_earth_r, gist_gray, \
# gist_gray_r, gist_heat, gist_heat_r, gist_ncar, gist_ncar_r, gist_rainbow, \
# gist_rainbow_r, gist_stern, gist_stern_r, gist_yarg, gist_yarg_r, gnuplot, \
# gnuplot2, gnuplot2_r, gnuplot_r, gray, gray_r, hot, hot_r, hsv, hsv_r, inferno, \
# inferno_r, jet, jet_r, magma, magma_r, nipy_spectral, nipy_spectral_r, ocean, \
# ocean_r, pink, pink_r, plasma, plasma_r, prism, prism_r, rainbow, rainbow_r, \
# seismic, seismic_r, spring, spring_r, summer, summer_r, tab10, tab10_r, tab20, \
# tab20_r, tab20b, tab20b_r, tab20c, tab20c_r, terrain, terrain_r, twilight, \
# twilight_r, twilight_shifted, twilight_shifted_r, viridis, viridis_r, winter, winter_r


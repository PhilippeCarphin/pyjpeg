import dct
import matplotlib.pyplot as plt
import numpy as np
import blocks


def demo():
    the_blocks = blocks.get_test_image_as_NxN_blocks()
    red_blocks, green_blocks, blue_blocks = blocks.split_rgb(the_blocks)

    ################### Bloc original
    my_block = red_blocks[30,40]
    print("Bloc original\n", my_block)
    plt.imshow(my_block, cmap=plt.get_cmap('gray_r'))
    plt.title("Bloc")
    plt.show()

    ################### Bloc encodé
    encoded_block = dct.encode_dct(my_block)
    print("Bloc encodé :\n", encoded_block)
    plt.imshow(encoded_block, cmap=plt.get_cmap('gray_r'))
    plt.title("Bloc > dct")
    plt.show()

    ################### Bloc DCT Quantifié
    quantized_block = dct.quantize.quantize_one_block(encoded_block)
    print("Bloc encodé, quantifié :\n", quantized_block)
    plt.imshow(quantized_block, cmap=plt.get_cmap('gray_r'))
    plt.title("Bloc > dct > quantifié")
    plt.show()

    fig = plt.figure(figsize=(4,4))
    fig.add_subplot(2,2,1)
    plt.imshow(dct.dct_basis_element(0,0), cmap=plt.get_cmap('gray_r'))

    fig.add_subplot(2,2,2)
    plt.imshow(dct.dct_basis_element(0,1), cmap=plt.get_cmap('gray_r'))

    fig.add_subplot(2,2,3)
    plt.imshow(dct.dct_basis_element(1,0), cmap=plt.get_cmap('gray_r'))

    fig.add_subplot(2,2,4)
    plt.imshow(dct.dct_basis_element(1,1), cmap=plt.get_cmap('gray_r'))
    plt.show()

    ################### Combinaison Linéaire avec deux coefficients
    # On fait une combinaison linéaire avec les coefficients du coin en haut à
    # gauche qui ont l'air les plux forts
    comb_lin = np.zeros((blocks.N,blocks.N))
    comb_lin += dct.dct_basis_element(0,1) * encoded_block[0,1] / 255
    comb_lin += dct.dct_basis_element(1,0) * encoded_block[1,0] / 255
    plt.imshow(comb_lin, cmap=plt.get_cmap('gray_r'))
    plt.title("Bloc > dct > quantification AGRESSIVE > idct")
    plt.show()

    ################### Combinaison Linéaire avec 16 coefficients
    fig_w = 8
    fig = plt.figure(figsize=(blocks.N,blocks.N))
    plt.title(f"{fig_w**2} vecteurs de la base DCT")
    for k in range(fig_w * fig_w):
        i = k // fig_w;
        j = k % fig_w;
        fig.add_subplot(fig_w,fig_w,k+1)
        plt.imshow(dct.dct_basis_element(i,j), cmap=plt.get_cmap('gray'))
    plt.show()
    i = 0
    j = 0
    print(f'bloc_dct({i},{j}) : \n{dct.dct_basis_element(i,j)}')
    i = 1
    j = 0
    print(f'bloc_dct({i},{j}) : \n{dct.dct_basis_element(i,j)}')

    comb_lin = np.zeros((blocks.N,blocks.N))
    for i in range(4):
        for j in range(4):
            comb_lin += dct.dct_basis_element(i,j) * encoded_block[i,j] / 255
    plt.imshow(comb_lin, cmap=plt.get_cmap('gray_r'))
    plt.title("Bloc > dct > quantification moins agressive > idct")
    plt.show()

    idct_block = dct.decode_dct(quantized_block)
    plt.imshow(idct_block, cmap=plt.get_cmap('gray_r'))
    plt.title("Bloc > dct > la vraie quantification > idct")
    plt.show()

    idct_block = dct.decode_dct(encoded_block)
    plt.imshow(idct_block, cmap=plt.get_cmap('gray_r'))
    plt.title("Bloc > dct > PAS de quantification > idct")
    plt.show()

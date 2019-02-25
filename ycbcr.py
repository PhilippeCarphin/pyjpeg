import image
import blocks
import matplotlib.pyplot as plt
import numpy as np
import dct_experiment as mdct


if __name__ == "__main__":
    img = image.get_test_image()

    h = img.shape[0]
    w = img.shape[1]

    # Make the dimensions a multiple of 16 so
    # that our chroma subsampling whatever can work easily
    img = img[:h - h%16, :w-w%16,:]

    green_channel = img[:,:,1]

    # Our friendly block (30,40)
    green_8x8_blocks = blocks.split_8x8(green_channel)
    that_block_green = green_8x8_blocks[30,40]
    # plt.imshow(that_block_green, cmap=plt.get_cmap('Greens'))
    # plt.show()

    ########### THIS IS THE SUBSAMPLING PART
    # NOTE We are not going to subsample the green, we're going to subsample the
    # from the YCbCr
    green_channel = green_channel[::2,::2]
    green_8x8_blocks = blocks.split_8x8(green_channel)

    # Print the sub sampled block, just the part corresponding
    # to the block above
    that_block_green = green_8x8_blocks[15,20]
    that_block_sub_part = that_block_green[:4,:4]
    # plt.imshow(that_block_sub_part, cmap=plt.get_cmap('Greens'))
    # plt.show()

    # Encoding and decoding foud on stack overflow
    # Ref : https://stackoverflow.com/a/34913974/5795941
    # What a numpy boss that guy.
    def rgb2ycbcr(im):
        """ Each rgb which is a im[i,j,:] so a vector that goes in the page,
        That vector gets multiplied by the matrix, M given below, and the result
        becomes ycbcr[i,j,:] he does the same thing for the other way around

        so this is like doing

        for i,j:
            rgb = im[i,j,:]
            ycc = M * rgb + b (multiplication d'un vecteur par une matrice
            ycbcr_img[i,j,:] = ycc

        et b = [0, 128, 128]."""
        xform = np.array(
            [[.299,    .587,   .114],
             [-.1687, -.3313,  .5],
             [.5,     -.4187, -.0813]])
        ycbcr = im.dot(xform.T)
        ycbcr[:, :, [1, 2]] += 128
        return np.uint8(ycbcr)

    def ycbcr2rgb(im):
        xform = np.array([
            [1, 0, 1.402],
            [1, -0.34414, -.71414],
            [1, 1.772, 0]])
        rgb = im.astype(np.float)
        rgb[:, :, [1, 2]] -= 128
        rgb = rgb.dot(xform.T)
        np.putmask(rgb, rgb > 255, 255)
        np.putmask(rgb, rgb < 0, 0)
        return np.uint8(rgb)

    img = image.get_test_image()
    img = img[:h - h%16, :w-w%16,:]
    ycbcr_img = rgb2ycbcr(img)
    img_back = ycbcr2rgb(ycbcr_img)

    # plt.imshow(img_back)
    # plt.show()

    YCBCR_IMG = ycbcr_img # juste pour flasher

    Y = YCBCR_IMG[:,:,0]
    CB = YCBCR_IMG[:,:,1]
    CR = YCBCR_IMG[:,:,2]
    Y_subsample = Y[::2,::2]

    # plt.imshow(Y_subsample, cmap=plt.get_cmap('gray'))
    # plt.show()
    # plt.imshow(CB, cmap=plt.get_cmap('gray'))
    # plt.show()
    # plt.imshow(CR, cmap=plt.get_cmap('gray'))
    # plt.show()

    Y_subsample_blocks = blocks.split_8x8(Y_subsample)
    CB_blocks = blocks.split_8x8(CB)
    CR_blocks = blocks.split_8x8(CR)

    CB_thresh = 3

    zz = [[0,  1,  5,  6,  14, 15, 27, 28],
          [2,  4,  7,  13, 16, 26, 29, 42],
          [3,  8,  12, 17, 25, 30, 41, 43],
          [9,  11, 18, 24, 31, 40, 44, 53],
          [10, 19, 23, 32, 39, 45, 52, 54],
          [20, 22, 33, 38, 46, 51, 55, 60],
          [21, 34, 37, 47, 50, 56, 59, 61],
          [35, 36, 48, 49, 57, 58, 62, 63]]

    my_block = CB_blocks[30, 40]

    # plt.imshow(my_block, cmap=plt.get_cmap('gray'))
    # plt.show()
    print(my_block)

    encoded_block = mdct.encode_dct(my_block)
    # plt.imshow(my_block, cmap=plt.get_cmap('gray'))
    # plt.show()
    print(encoded_block)

    Quant1= np.matrix('16 11 10 16 24 40 51 61;\
        12 12 14 19 26 58 60 55;\
        14 13 16 24 40 57 69 56;\
        14 17 22 29 51 87 80 62;\
        18 22 37 56 68 109 103 77;\
        24 35 55 64 81 104 103 92;\
        49 64 78 77 103 121 120 101;\
        72 92 95 98 112 100 103 99').astype('float')

    quantified_block = np.round(np.divide(encoded_block, Quant1))
    print(quantified_block.astype('int8'))

    decoded_block = mdct.decode_dct(quantified_block)
    plt.imshow(my_block, cmap=plt.get_cmap('gray'))
    plt.show()
    plt.imshow(decoded_block, cmap=plt.get_cmap('gray'))
    plt.show()

import image
import ycbcr
import subsampling
import blocks
import dct
import quantize
import zigzag
import huffman_8770 as huffman
import numpy as np

import matplotlib.pyplot as plt
subsample = True
def encode(filename):
    rgb_img = image.open_image_as_ndarray(filename)

    ycbcr_img = ycbcr.rgb2ycbcr(rgb_img)

    subsampled = subsampling.scheme_subsample(ycbcr_img, (4,4,4))
    blocks_Y_ss = blocks.split_NxN(subsampled['Y'])
    blocks_Cb_ss = blocks.split_NxN(subsampled['Cb'])
    blocks_Cr_ss = blocks.split_NxN(subsampled['Cr'])
    ycbcr_blocks = blocks.split_NxN(ycbcr_img)
    blocks_Y = ycbcr_blocks[:,:,:,:,0]
    blocks_Cb = ycbcr_blocks[:,:,:,:,1]
    blocks_Cr = ycbcr_blocks[:,:,:,:,2]

    assert blocks_Y_ss.shape == blocks_Y.shape
    if subsample:
        blocks_Y = blocks_Y_ss
        blocks_Cb = blocks_Cb_ss
        blocks_Cr = blocks_Cr_ss



    return (blocks_Y, blocks_Cb, blocks_Cr)

def decode(encoded):
    blocks_Y, blocks_Cb, blocks_Cr = encoded

    Y_channel = blocks.combine_NxN_channel(blocks_Y)
    Cb_channel = blocks.combine_NxN_channel(blocks_Cb)
    Cr_channel = blocks.combine_NxN_channel(blocks_Cr)


    if subsample:
        ycbcr_img = subsampling.upsample_and_assemble({
            'Y': Y_channel,
            'Cb': Cb_channel,
            'Cr': Cr_channel,
            'scheme': (4,4,4)
        })
    else:
        ycbcr_img = np.zeros(Y_channel.shape + (3,))
        ycbcr_img[:,:,0] = Y_channel
        ycbcr_img[:,:,1] = Cb_channel
        ycbcr_img[:,:,2] = Cr_channel

    rgb_img = ycbcr.ycbcr2rgb(ycbcr_img)

    return rgb_img




if __name__ == "__main__":
    encoded = encode('input_image.png')

    rgb_img = decode(encoded)

    print(rgb_img.dtype)

    plt.imshow(rgb_img)
    plt.show()













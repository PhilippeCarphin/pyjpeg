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
use_dct = True
use_quantize = True
use_zigzag = True
use_huffman = False

image_shape = None
blocks_shape = None
subsampled_shape = None

def encode(filename):

    global subsampled_shape
    global blocks_shape

    rgb_img = image.open_image_as_ndarray(filename)
    ycbcr_img = ycbcr.rgb2ycbcr(rgb_img)

    ycbcr_blocks = blocks.split_NxN(ycbcr_img)
    blocks_Y = ycbcr_blocks[:,:,:,:,0]
    blocks_Cb = ycbcr_blocks[:,:,:,:,1]
    blocks_Cr = ycbcr_blocks[:,:,:,:,2]

    if subsample:

        subsampled = subsampling.scheme_subsample(ycbcr_img, (4,2,0))

        blocks_Y_ss = blocks.split_NxN(subsampled['Y'])
        blocks_Cb_ss = blocks.split_NxN(subsampled['Cb'])
        blocks_Cr_ss = blocks.split_NxN(subsampled['Cr'])

        blocks_Y = blocks_Y_ss
        blocks_Cb = blocks_Cb_ss
        blocks_Cr = blocks_Cr_ss

        assert blocks_Y_ss.shape == blocks_Y.shape

    subsampled_shape = blocks_Cb.shape
    blocks_shape = blocks_Y_ss.shape

    if use_dct:
        blocks_Y = dct.dct_encode_blocks(blocks_Y)
        blocks_Cb = dct.dct_encode_blocks(blocks_Cb)
        blocks_Cr = dct.dct_encode_blocks(blocks_Cr)

        if use_quantize:
            blocks_Y = quantize.quantize_blocks(blocks_Y)
            blocks_Cb = quantize.quantize_blocks(blocks_Cb)
            blocks_Cr = quantize.quantize_blocks(blocks_Cr)

            if use_zigzag:
                zigzag_Y = zigzag.zig_zag_blocks(blocks_Y)
                zigzag_Cb = zigzag.zig_zag_blocks(blocks_Cb)
                zigzag_Cr = zigzag.zig_zag_blocks(blocks_Cr)
                if use_huffman:
                    huffed_Y = huffman.huffman_encode(zigzag_Y.astype('uint8'))
                    huffed_Cb = huffman.huffman_encode(zigzag_Cb.astype('uint8'))
                    huffed_Cr = huffman.huffman_encode(zigzag_Cr.astype('uint8'))
                    return {
                        'data': (huffed_Y, huffed_Cb, huffed_Cr),
                        'subsampling_scheme': (4,2,0),
                        'img_shape': ycbcr_img.shape,
                        'Y_shape': blocks_Y.shape,
                        'Cr_shape': blocks_Cr.shape}
                else:
                    return (zigzag_Y, zigzag_Cb, zigzag_Cr)

        return (blocks_Y, blocks_Cb, blocks_Cr)

def decode(encoded):
    if not use_huffman:
        blocks_Y, blocks_Cb, blocks_Cr = encoded

    if use_zigzag:

        if use_huffman:
            huffed_Y, huffed_Cb, huffed_Cr = encoded['data']

            zigzag_Y = np.uint8(huffman.huffman_decode(huffed_Y['data'], huffed_Y['codebook'])).astype('float')
            zigzag_Cb = np.uint8(huffman.huffman_decode(huffed_Cb['data'], huffed_Cb['codebook'])).astype('float')
            zigzag_Cr = np.uint8(huffman.huffman_decode(huffed_Cr['data'], huffed_Cr['codebook'])).astype('float')
        else:
            zigzag_Y, zigzag_Cb, zigzag_Cr = encoded

        blocks_Y = zigzag.un_zig_zag_blocks(zigzag_Y, blocks_shape)
        blocks_Cb = zigzag.un_zig_zag_blocks(zigzag_Cb, subsampled_shape)
        blocks_Cr = zigzag.un_zig_zag_blocks(zigzag_Cr, subsampled_shape)

    if use_dct:
        blocks_Y = dct.dct_decode_blocks(blocks_Y)
        blocks_Cb = dct.dct_decode_blocks(blocks_Cb)
        blocks_Cr = dct.dct_decode_blocks(blocks_Cr)


    Y_channel = blocks.combine_NxN_channel(blocks_Y)
    Cb_channel = blocks.combine_NxN_channel(blocks_Cb)
    Cr_channel = blocks.combine_NxN_channel(blocks_Cr)


    if subsample:
        ycbcr_img = subsampling.upsample_and_assemble({
            'Y': Y_channel,
            'Cb': Cb_channel,
            'Cr': Cr_channel,
            'scheme': (4,2,0)
        })
    else:
        ycbcr_img = np.zeros(Y_channel.shape + (3,))
        ycbcr_img[:,:,0] = Y_channel
        ycbcr_img[:,:,1] = Cb_channel
        ycbcr_img[:,:,2] = Cr_channel

    rgb_img = ycbcr.ycbcr2rgb(ycbcr_img)

    return rgb_img




if __name__ == "__main__":
    img_input = image.get_test_image()
    encoded = encode('input_image.png')

    rgb_img = decode(encoded)

    print(rgb_img.dtype)

    plt.imshow(rgb_img)
    plt.show()













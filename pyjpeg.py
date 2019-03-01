import image
import ycbcr
import subsampling
import blocks
import dct
import quantize
import zigzag
import huffman_8770 as huffman
import numpy as np
from skimage import io

import matplotlib.pyplot as plt


class JpegObject():
    steps = {
        1: 'ycbcr',
        2: 'subsampling',
        3: 'blocks',
        4: 'dct',
        5: 'zigzag',
        6: 'huffman'
    }
    def __init__(self, **kwargs):
        self.use_zigzag=kwargs.get('use_zigzag', True)
        self.use_dct=kwargs.get('use_dct', True)
        self.use_subsampling = True
        self.use_dct = True
        self.use_quantize = True
        self.use_zigzag = True
        self.use_huffman = False
        self.image_shape = None
        self.blocks_shape = None
        self.subsampled_shape = None
        self.last_step = None


# TODO Remplacer les variables golbales par des paramêtres
    def encode_decode(self, filename):

        rgb_img = image.open_image_as_ndarray(filename)
        ycbcr_img = ycbcr.rgb2ycbcr(rgb_img)

        ycbcr_blocks = blocks.split_NxN(ycbcr_img)

        if self.use_subsampling:
            subsampled = subsampling.scheme_subsample(ycbcr_img, (4,2,0))
            blocks_Y = blocks.split_NxN(subsampled['Y'])
            blocks_Cb = blocks.split_NxN(subsampled['Cb'])
            blocks_Cr = blocks.split_NxN(subsampled['Cr'])
        else:
            blocks_Y = ycbcr_blocks[:,:,:,:,0]
            blocks_Cb = ycbcr_blocks[:,:,:,:,1]
            blocks_Cr = ycbcr_blocks[:,:,:,:,2]

        subsampled_shape = blocks_Cb.shape
        blocks_shape = blocks_Y.shape

        if self.use_dct:
            blocks_Y = dct.dct_encode_blocks(blocks_Y)
            blocks_Cb = dct.dct_encode_blocks(blocks_Cb)
            blocks_Cr = dct.dct_encode_blocks(blocks_Cr)

        if self.use_quantize:
            blocks_Y = quantize.quantize_blocks(blocks_Y)
            blocks_Cb = quantize.quantize_blocks(blocks_Cb)
            blocks_Cr = quantize.quantize_blocks(blocks_Cr)

        if self.use_zigzag:
            zigzag_Y = zigzag.zig_zag_blocks(blocks_Y)
            zigzag_Cb = zigzag.zig_zag_blocks(blocks_Cb)
            zigzag_Cr = zigzag.zig_zag_blocks(blocks_Cr)
            zigzag_object = {
                'data': (zigzag_Y, zigzag_Cb, zigzag_Cr),
                'full_shape': blocks_Y.shape,
                'CbCr_shape': blocks_Cb.shape
            }
            if self.use_huffman:
                huffed_Y = huffman.huffman_encode(zigzag_Y.astype('uint8'))
                huffed_Cb = huffman.huffman_encode(zigzag_Cb.astype('uint8'))
                huffed_Cr = huffman.huffman_encode(zigzag_Cr.astype('uint8'))
                huffman_object = {
                    'data': (huffed_Y, huffed_Cb, huffed_Cr),
                    'subsampling_scheme': (4,2,0),
                    'img_shape': ycbcr_img.shape,
                    'Y_shape': blocks_Y.shape,
                    'Cr_shape': blocks_Cr.shape}
    ################# BEGIN DECODE #################################################
                # Undo Huffman
                huffed_Y, huffed_Cb, huffed_Cr = huffman_object['data']
                zigzag_Y = np.uint8(huffman.huffman_decode(huffed_Y['data'], huffed_Y['codebook'])).astype('float')
                zigzag_Cb = np.uint8(huffman.huffman_decode(huffed_Cb['data'], huffed_Cb['codebook'])).astype('float')
                zigzag_Cr = np.uint8(huffman.huffman_decode(huffed_Cr['data'], huffed_Cr['codebook'])).astype('float')

            zigzag_Y, zigzag_Cb, zigzag_Cr = zigzag_object['data']
            # Undo Zigzag
            blocks_Y = zigzag.un_zig_zag_blocks(zigzag_Y, zigzag_object['full_shape'])
            blocks_Cb = zigzag.un_zig_zag_blocks(zigzag_Cb, zigzag_object['CbCr_shape'])
            blocks_Cr = zigzag.un_zig_zag_blocks(zigzag_Cr, zigzag_object['CbCr_shape'])

        if self.use_dct:
            blocks_Y = dct.dct_decode_blocks(blocks_Y)
            blocks_Cb = dct.dct_decode_blocks(blocks_Cb)
            blocks_Cr = dct.dct_decode_blocks(blocks_Cr)

        Y_channel = blocks.combine_NxN_channel(blocks_Y)
        Cb_channel = blocks.combine_NxN_channel(blocks_Cb)
        Cr_channel = blocks.combine_NxN_channel(blocks_Cr)

        if self.use_subsampling:
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

        io.imsave('/Users/pcarphin/Desktop/output.png', rgb_img)

        return rgb_img

if __name__ == "__main__":
    jpegobj = JpegObject()
    encoded_decoded = jpegobj.encode_decode('input_image.png')













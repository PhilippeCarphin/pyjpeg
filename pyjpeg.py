import os
import numpy as np
from skimage import io
import itertools

import image
import ycbcr
import subsampling
import blocks
import dct
import quantize
import zigzag
import huffman_8770 as huffman

import matplotlib.pyplot as plt


class JpegObject():
    def __init__(self, **kwargs):
        self.use_zigzag = kwargs.get('use_zigzag', True)
        self.use_dct = kwargs.get('use_dct', True)
        self.use_subsampling = kwargs.get('use_subsampling', True)
        self.subsample_scheme = kwargs.get('subsample_scheme', (4, 2, 0))
        self.use_dct = True
        self.use_quantize = kwargs.get('use_quantize', True)
        self.quant = kwargs.get('quant', quantize.Quant1)
        self.use_zigzag = True
        self.use_huffman = kwargs.get('use_huffman', True)
        self.image_shape = None
        self.blocks_shape = None
        self.cbcr_shape = None
        self.last_step = None
        self.huff_code_Y = None
        self.huff_code_Cb = None
        self.huff_code_Cr = None

        if self.use_huffman and not self.use_zigzag:
            raise Exception("Zigzag must be done to do huffman encoding")

    def get_subsampled_blocks(self, ycbcr_img):
        subsampled = subsampling.scheme_subsample(ycbcr_img, self.subsample_scheme)
        blocks_Y = blocks.split_NxN(subsampled['Y'])
        blocks_Cb = blocks.split_NxN(subsampled['Cb'])
        blocks_Cr = blocks.split_NxN(subsampled['Cr'])
        self.blocks_shape = blocks_Y.shape
        self.CbCr_shape = blocks_Cb.shape
        channels_data = (blocks_Y, blocks_Cb, blocks_Cr)
        return channels_data

    def undo_subsampling(self, channels_data):
        Y_channel, Cb_channel, Cr_channel = channels_data
        ycbcr_img = subsampling.upsample_and_assemble({
            'Y': Y_channel,
            'Cb': Cb_channel,
            'Cr': Cr_channel,
            'scheme': self.subsample_scheme
        })
        return ycbcr_img

    def split_channels(self, ycbcr_blocks):
        blocks_Y = ycbcr_blocks[:, :, :, :, 0]
        blocks_Cb = ycbcr_blocks[:, :, :, :, 1]
        blocks_Cr = ycbcr_blocks[:, :, :, :, 2]
        self.blocks_shape = blocks_Y.shape
        self.CbCr_shape = blocks_Cb.shape
        return (blocks_Y, blocks_Cb, blocks_Cr)

    def combine_channels(self, channels_data):
        # NOTE, combine channels takes a tuple of channels
        # while split channels takes things in block form
        Y_channel, Cb_channel, Cr_channel = channels_data
        ycbcr_img = np.zeros(Y_channel.shape + (3,))
        ycbcr_img[:, :, 0] = Y_channel
        ycbcr_img[:, :, 1] = Cb_channel
        ycbcr_img[:, :, 2] = Cr_channel
        return ycbcr_img

    def do_combine_blocks(self, blocks_data):
        blocks_Y, blocks_Cb, blocks_Cr = blocks_data
        # Combinaison de blocs
        Y_channel = blocks.combine_NxN_channel(blocks_Y)
        Cb_channel = blocks.combine_NxN_channel(blocks_Cb)
        Cr_channel = blocks.combine_NxN_channel(blocks_Cr)
        ycbcr_data = (Y_channel, Cb_channel, Cr_channel)
        return ycbcr_data

    def do_dct(self, ycbcr_data):
        blocks_Y, blocks_Cb, blocks_Cr = ycbcr_data
        dct_blocks_Y = dct.dct_encode_blocks(blocks_Y)
        dct_blocks_Cb = dct.dct_encode_blocks(blocks_Cb)
        dct_blocks_Cr = dct.dct_encode_blocks(blocks_Cr)
        dct_blocks_data = (dct_blocks_Y, dct_blocks_Cb, dct_blocks_Cr)
        return dct_blocks_data

    def undo_dct(self, dct_blocks_data):
        dct_blocks_Y, dct_blocks_Cb, dct_blocks_Cr = dct_blocks_data
        blocks_Y = dct.dct_decode_blocks(dct_blocks_Y)
        blocks_Cb = dct.dct_decode_blocks(dct_blocks_Cb)
        blocks_Cr = dct.dct_decode_blocks(dct_blocks_Cr)
        blocks_data = (blocks_Y, blocks_Cb, blocks_Cr)
        return blocks_data

    def do_quantize(self, blocks_data):
        blocks_Y, blocks_Cb, blocks_Cr = blocks_data
        quantized_blocks_Y = quantize.quantize_blocks(blocks_Y)
        quantized_blocks_Cb = quantize.quantize_blocks(blocks_Cb)
        quantized_blocks_Cr = quantize.quantize_blocks(blocks_Cr)
        quantize_data = (quantized_blocks_Y, quantized_blocks_Cb, quantized_blocks_Cr)
        return quantize_data

    def do_zigzag(self, blocks_data):
        blocks_Y, blocks_Cb, blocks_Cr = blocks_data
        zigzag_Y = zigzag.zig_zag_blocks(blocks_Y)
        zigzag_Cb = zigzag.zig_zag_blocks(blocks_Cb)
        zigzag_Cr = zigzag.zig_zag_blocks(blocks_Cr)
        self.zigzag_length = len(zigzag_Y) + len(zigzag_Cb) + len(zigzag_Cr)
        zigzag_data = (zigzag_Y, zigzag_Cb, zigzag_Cr)
        return zigzag_data

    def undo_zigzag(self, zigzag_data):
        zigzag_Y, zigzag_Cb, zigzag_Cr = zigzag_data
        blocks_Y = zigzag.un_zig_zag_blocks(zigzag_Y, self.blocks_shape)
        blocks_Cb = zigzag.un_zig_zag_blocks(zigzag_Cb, self.CbCr_shape)
        blocks_Cr = zigzag.un_zig_zag_blocks(zigzag_Cr, self.CbCr_shape)
        blocks_data = (blocks_Y, blocks_Cb, blocks_Cr)
        return blocks_data

    def do_huffman(self, zigzag_data):
        zigzag_Y, zigzag_Cb, zigzag_Cr = zigzag_data
        huffed_Y = huffman.huffman_encode(zigzag_Y.astype('uint8'))
        huffed_Cb = huffman.huffman_encode(zigzag_Cb.astype('uint8'))
        huffed_Cr = huffman.huffman_encode(zigzag_Cr.astype('uint8'))
        huffman_data = (huffed_Y, huffed_Cb, huffed_Cr)
        return huffman_data

    def undo_huffman(self, huffman_data):
        huffed_Y, huffed_Cb, huffed_Cr = huffman_data
        zigzag_Y = np.uint8(huffman.huffman_decode(huffed_Y['data'], huffed_Y['codebook']))
        zigzag_Cb = np.uint8(huffman.huffman_decode(huffed_Cb['data'], huffed_Cb['codebook']))
        zigzag_Cr = np.uint8(huffman.huffman_decode(huffed_Cr['data'], huffed_Cr['codebook']))
        zigzag_data = (zigzag_Y, zigzag_Cb, zigzag_Cr)
        return zigzag_data

    def encode_decode_file(self, filename):
        rgb_img = image.open_image_as_ndarray(filename)
        return self.encode_decode(rgb_img)

    def encode_decode(self, rgb_img):

        ycbcr_img = ycbcr.rgb2ycbcr(rgb_img)
        ycbcr_blocks = blocks.split_NxN(ycbcr_img)
        self.blocks_shape = ycbcr_blocks[:, :, :, :, 0].shape

        if self.use_subsampling:
            ycbcr_data = self.get_subsampled_blocks(ycbcr_img)
        else:
            ycbcr_data = self.split_channels(ycbcr_blocks)

        if self.use_dct:
            ycbcr_data = self.do_dct(ycbcr_data)
        if self.use_quantize:
            ycbcr_data = self.do_quantize(ycbcr_data)
        if self.use_zigzag:
            zigzag_data = self.do_zigzag(ycbcr_data)
        if self.use_huffman:
            huffman_data = self.do_huffman(zigzag_data)

        ################# BEGIN DECODE #################################################

        if self.use_huffman:
            zigzag_data = self.undo_huffman(huffman_data)

        if not self.use_huffman:
            huffman_data = self.do_huffman(zigzag_data)
            huffed_Y, huffed_Cb, huffed_Cr = huffman_data
            self.huffman_size = len(huffed_Y['data']) + len(huffed_Cb['data']) + len(huffed_Cr['data'])

        if self.use_zigzag:
            blocks_data = self.undo_zigzag(zigzag_data)
        if self.use_dct:
            # takes blocks_data and spits out blocks_data
            blocks_data = self.undo_dct(blocks_data)

        channels_data = self.do_combine_blocks(blocks_data)

        if self.use_subsampling:
            # Undo subsampling combine les channels en même temps, peut-être je
            # le changerai plus tard
            ycbcr_img = self.undo_subsampling(channels_data)
        else:
            ycbcr_img = self.combine_channels(channels_data)

        rgb_img = ycbcr.ycbcr2rgb(ycbcr_img)

        io.imsave(os.path.expanduser('~/Desktop/output.png'), rgb_img)

        return rgb_img

def quant_str(q):
    if q is quantize.Quant1:
        quant_str = 'quant-std'
    elif q is quantize.Quant2:
        quant_str = 'quant-std-modif'
    elif q is quantize.QuantAgressive:
        quant_str = 'quant-aggres'
    elif q is quantize.QuantSuperAgressive:
        quant_str = 'quant-super-aggres'
    else:
        quant_str = "None"
    return quant_str

def generate_images():
    images = [
        'input_image.png',
        # 'images/asdf', #ADAM ICI!!
    ]
    quants = [
        quantize.Quant1,
        quantize.Quant2,
        quantize.QuantAgressive,
        quantize.QuantSuperAgressive
    ]
    subsampling_schemes = [
        (4,4,4),
        (4,2,0),
        (4,1,1)
    ]
    for quant, subsampling_scheme, img_file in itertools.product(quants, subsampling_schemes, images):
        jpegobj = JpegObject(
            use_huffman=False,
            use_subsampling=True,
            use_dct=True,
            use_quantize=True,
            quant=quant,
            subsample_scheme=subsampling_scheme
        )
        encoded_decoded = jpegobj.encode_decode_file(img_file)
        print(jpegobj.zigzag_length)
        print(jpegobj.huffman_size // 8) # parce que huffman_size est la longeur d'une liste de bits


        filename = img_file[:-4] # remove the '.png'
        filename += '_' + str(subsampling_scheme) + '_' + quant_str(quant) + '.png'

        filepath = os.path.join(os.getcwd(),'images_traitees', filename)
        io.imsave(filepath, encoded_decoded)
        print("saved file {}".format(filepath))



if __name__ == "__main__":
    pass
    jpegobj = JpegObject(
        use_huffman=False,
        use_subsampling=True,
        use_dct=True,
        use_quantize=True,
        # quant=quantize.Quant1,
        subsample_scheme=(4,2,0) # or (4,1,1) or (4,4,4) ((4,4,4) is equivalent to setting use_quantize to false)
    )
    encoded_decoded = jpegobj.encode_decode_file('input_image.png')
    print(jpegobj.zigzag_length)
    print(jpegobj.huffman_size // 8)  # parce que huffman_size est la longeur d'une liste de bits
    plt.imshow(encoded_decoded)
    plt.show()

    generate_images()

    jpeg1 = JpegObject(
        use_huffman=False,
        use_subsampling=True,
        use_dct=True,
        use_quantize=True,
        quant=quantize.Quant1,
        subsample_scheme=(4,2,0)
    )
    jpeg2 = JpegObject(
        use_huffman=False,
        use_subsampling=True,
        use_dct=True,
        use_quantize=True,
        quant=quantize.Quant2,
        subsample_scheme=(4,2,0)
    )


    compressed_once = jpeg1.encode_decode_file('input_image.png')
    compressed_twice = jpeg1.encode_decode(compressed_once)
    compressed_thrice = jpeg1.encode_decode(compressed_twice)

    plt.imshow(compressed_thrice)
    plt.show()

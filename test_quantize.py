import unittest
import quantize
import blocks


class TestQuantize(unittest.TestCase):

    def test_quantize(self):
        my_block = blocks.get_one_test_NxN_block()
        quantized_block = quantize.quantize_one_block(my_block)

        img = blocks.get_test_image_as_NxN_blocks()
        img = blocks.open_image_as_NxN_blocks('input_image.png')
        quantize.quantize_blocks(img[:, :, :, :, 0])
    # print(f"quantized_block : {quantized_block}")

    # ################################### APPLY TO YCBCR
    # ycbcr_img_blocks = blocks.split_NxN(ycbcr.get_ycbcr_test_image())

    # Y_blocks = ycbcr_img_blocks[:,:,:,:,0]
    # CB_blocks = ycbcr_img_blocks[:,:,:,:,1]
    # CR_blocks = ycbcr_img_blocks[:,:,:,:,2]

    # my_block = CB_blocks[30, 40]
    # plt.imshow(my_block, cmap=plt.get_cmap('gray_r'))
    # plt.title("Un blocCB (le bloc (30,40))")
    # plt.show()

    # encoded_block = dct.encode_dct(my_block)
    # print(f"dct block\n{encoded_block}")
    # plt.imshow(encoded_block, cmap=plt.get_cmap('gray_r'))
    # plt.title("BlocCB > DCT")
    # plt.show()

    # CB_blocks_dct = dct.dct_encode_blocks(CB_blocks)

    # # Quantizing a whole bunch of blocks with slices
    # quantized_blocks = quantize_blocks(CB_blocks_dct)
    # quantified_block = quantized_blocks[30,40]
    # plt.imshow(quantified_block, cmap=plt.get_cmap('gray_r'))
    # plt.title("BlocCB > DCT > Quantification ")
    # plt.show()
    # print(f"quantized dct block\n{quantified_block}")

    # decoded_block = dct.decode_dct(quantified_block)
    # plt.imshow(decoded_block, cmap=plt.get_cmap('gray_r'))
    # plt.title("BlocCB > DCT > Quantification > IDCT")
    # plt.show()

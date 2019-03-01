import numpy as np
import dct

Quant1= np.matrix( ' 1  1 10 16  24  40  51  61;\
                     1 12 14 19  26  58  60  55;\
                    14 13 16 24  40  57  69  56;\
                    14 17 22 29  51  87  80  62;\
                    18 22 37 56  68 109 103  77;\
                    24 35 55 64  81 104 103  92;\
                    49 64 78 77 103 121 120 101;\
                    72 92 95 98 112 100 103  99').astype('uint8')
Quant2= np.matrix( ' 1  1 255 255  255  255  255  255;\
                    1 255 255 255  255  255  60  55;\
                    2551 255 255 255  255  57  69  56;\
                    255 255 255 255  51  87  80  62;\
                    255 255 255 56  68 109 103  77;\
                    255 255 255 64  81 104 103  92;\
                    255 255 255 77 103 121 120 101;\
                    255 255 255 98 112 100 103  99').astype('uint8')

def quantize_one_block(block, quant=Quant1):
    assert block.shape == quant.shape, f"blocks must have same shape as quant"
    qb = np.empty_like(block)
    qb[:,:] = np.round(np.divide(block[:,:], quant))
    return qb.astype('float')

def quantize_blocks(blocks, quant=Quant2):
    assert blocks.shape[2:] == quant.shape, f"blocks must have same shape as quant"
    """ Takes an array of shape (n_blocks_h, n_blocks_w, N, N) and
    goes through all the blocks to quantize them"""
    # (n_blocks_h, n_blocks_w, N,N)
    qb = np.empty_like(blocks)
    assert len(blocks.shape) == 4
    n_blocks_h = blocks.shape[0]
    n_blocks_w = blocks.shape[1]
    for i in range(n_blocks_h):
        for j in range(n_blocks_w):
            qb[i,j,:,:] = np.round(np.divide(blocks[i,j,:,:], quant))
    return qb.astype('float')

if __name__ == "__main__":
    pass
    # import blocks
    # import image
    # import matplotlib.pyplot as plt
    # import ycbcr

    # my_block = blocks.get_one_test_NxN_block()
    # print(f"my_block : {my_block}")

    # quantized_block = quantize_one_block(my_block)
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

CB_thresh = 3

zz = [[0,  1,  5,  6,  14, 15, 27, 28],
      [2,  4,  7,  13, 16, 26, 29, 42],
      [3,  8,  12, 17, 25, 30, 41, 43],
      [9,  11, 18, 24, 31, 40, 44, 53],
      [10, 19, 23, 32, 39, 45, 52, 54],
      [20, 22, 33, 38, 46, 51, 55, 60],
      [21, 34, 37, 47, 50, 56, 59, 61],
      [35, 36, 48, 49, 57, 58, 62, 63]]
"""
Y_subsample_blocks = blocks.split_8x8(Y_subsample)
CB_blocks = blocks.split_8x8(CB)
CR_blocks = blocks.split_8x8(CR)

n_blocks_h = CB_blocks.shape[0]
n_blocks_w = CB_blocks.shape[1]


my_block = CB_blocks[30, 40]
plt.imshow(my_block, cmap=plt.get_cmap('gray_r'))
plt.title("")
plt.show()
# print(my_block)

encoded_block = mdct.encode_dct(my_block)
# plt.imshow(my_block, cmap=plt.get_cmap('gray_r'))
# plt.show()
# print(encoded_block)

# Le astype('float') est ben important ici, sinon les
# tests ont l'air de pas marcher
CB_blocks_dct = np.empty_like(CB_blocks).astype('float')
for i in range(n_blocks_h):
    for j in range(n_blocks_w):
        CB_blocks_dct[i,j,:,:] = mdct.encode_dct(CB_blocks[i,j,:,:])

# Quantizing a whole bunch of blocks with slices
quantized_blocks = quantize.quantize_blocks(CB_blocks_dct)
quantified_block = quantized_blocks[30,40]
print(quantified_block.astype('int8'))

decoded_block = mdct.decode_dct(quantified_block)
plt.imshow(my_block, cmap=plt.get_cmap('gray_r'))
plt.show()
plt.imshow(decoded_block, cmap=plt.get_cmap('gray_r'))
plt.show()
"""

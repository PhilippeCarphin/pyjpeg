import image
import blocks
import matplotlib.pyplot as plt


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
    plt.imshow(that_block_green, cmap=plt.get_cmap('Greens'))
    plt.show()

    ########### THIS IS THE SUBSAMPLING PART
    # NOTE We are not going to subsample the green, we're going to subsample the
    # from the YCbCr
    green_channel = green_channel[::2,::2]
    green_8x8_blocks = blocks.split_8x8(green_channel)

    # Print the sub sampled block, just the part corresponding
    # to the block above
    that_block_green = green_8x8_blocks[15,20]
    that_block_sub_part = that_block_green[:4,:4]
    plt.imshow(that_block_sub_part, cmap=plt.get_cmap('Greens'))
    plt.show()

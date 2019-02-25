import image
import blocks


if __name__ == "__main__":
    img = image.get_test_image()

    h = img.shape[0]
    w = img.shape[1]

    # Make the dimensions a multiple of 16 so
    # that our chroma subsampling whatever can work easily
    img = img[:h - h%16, :w-w%16,:]
    print(img.shape)

    green_channel = img[:,:,1]
    print(green_channel.shape)

    green_channel = green_channel[::2,::2]
    print(green_channel.shape)

    green_8x8_blocks = blocks.split_8x8(green_channel)
    print(green_8x8_blocks.shape)

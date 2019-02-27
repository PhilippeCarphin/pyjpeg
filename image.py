from skimage import io
import matplotlib.pyplot as plt
import numpy as np

np.set_printoptions(precision=2)
np.set_printoptions(suppress=True)


def open_image_as_ndarray(filename):
    img = io.imread(filename)
    h = img.shape[0]
    w = img.shape[1]
    assert h % 16 == 0 and w % 16 == 0, "We're only doing multiples of 16 for width and height"
    return img


def get_test_image():
    # RECAP : Opening the image and splitting it into blocks
    img = io.imread('input_image.png')
    h = img.shape[0]
    w = img.shape[1]
    img = img[:h - h % 16, :w - w % 16, :]
    return img


if __name__ == "__main__":
    my_img = get_test_image()
    red = my_img[:, :, 0]
    green = my_img[:, :, 1]
    blue = my_img[:, :, 2]
    plt.imshow(my_img)
    plt.show()
    plt.imshow(red, cmap=plt.get_cmap('Reds'))
    plt.show()
    plt.imshow(green, cmap=plt.get_cmap('Greens'))
    plt.show()
    plt.imshow(blue, cmap=plt.get_cmap('Blues'))
    plt.show()

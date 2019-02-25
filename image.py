from skimage import io
import matplotlib.pyplot as plt
import numpy as np
np.set_printoptions(precision=2)
np.set_printoptions(suppress=True)
def open_image_as_ndarray(filename):
    img = io.imread(filename)
    return img.astype('float')

def get_test_image():
    # RECAP : Opening the image and splitting it into blocks
    img = io.imread('input_image.png')
    return img

if __name__ == "__main__":
    img = get_test_image()
    red = img[:,:,0]
    green = img[:,:,1]
    blue = img[:,:,2]
    plt.imshow(img)
    plt.show()
    plt.imshow(red, cmap=plt.get_cmap('Reds'))
    plt.show()
    plt.imshow(green, cmap=plt.get_cmap('Greens'))
    plt.show()
    plt.imshow(blue, cmap=plt.get_cmap('Blues'))
    plt.show()



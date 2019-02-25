from skimage import io

def open_image_as_ndarray(filename):
    img = io.imread(filename)
    return img.astype('float')

def get_test_image():
    # RECAP : Opening the image and splitting it into blocks
    img = io.imread('input_image.png').astype('float')
    return img



import image
import ycbcr
import blocks

def encode(filename):
    rgb_img = image.open_image_as_ndarray(filename)

    ycbcr_img = ycbcr.rgb2ycbcr(rgb_img)













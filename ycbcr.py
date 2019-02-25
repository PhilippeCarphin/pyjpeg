import image
import blocks
import matplotlib.pyplot as plt
import numpy as np
import dct_experiment as mdct
import quantize


# Encoding and decoding foud on stack overflow
# Ref : https://stackoverflow.com/a/34913974/5795941
def rgb2ycbcr(im):
    """ Phil notes: Each rgb which is a im[i,j,:] so a vector that goes
    inside the page if your rgb block is a stack of three 2D arrays
    That vector gets multiplied by the matrix, M given below, and the result
    becomes ycbcr[i,j,:] he does the same thing for the inverse function

    so this is like doing

    for i,j:
        rgb = im[i,j,:]
        ycc = M * rgb + b (multiplication d'un vecteur par une matrice)
        ycbcr_img[i,j,:] = ycc

    et b = [0, 128, 128].
    """
    xform = np.array(
        [[.299,    .587,   .114],
         [-.1687, -.3313,  .5],
         [.5,     -.4187, -.0813]])
    ycbcr = im.dot(xform.T)
    ycbcr[:, :, [1, 2]] += 128
    return np.uint8(ycbcr)

def ycbcr2rgb(im):
    xform = np.array([
        [1, 0, 1.402],
        [1, -0.34414, -.71414],
        [1, 1.772, 0]])
    # The as float is important
    rgb = im.astype(np.float)
    rgb[:, :, [1, 2]] -= 128
    rgb = rgb.dot(xform.T)
    np.putmask(rgb, rgb > 255, 255)
    np.putmask(rgb, rgb < 0, 0)
    return np.uint8(rgb)


def get_ycbcr_test_image():
    img = image.get_test_image()
    return rgb2ycbcr(img)

if __name__ == "__main__":
    img = image.get_test_image()
    plt.imshow(img)
    plt.title("Image originale")
    plt.show()

    ycbcr_img = rgb2ycbcr(img)
    img_back = ycbcr2rgb(ycbcr_img)
    plt.imshow(abs(img_back.astype('float') - img.astype('float')))
    plt.title("Image encodée et décodée")
    plt.show()


    Y = ycbcr_img[:,:,0]
    CB = ycbcr_img[:,:,1]
    CR = ycbcr_img[:,:,2]

    Y_subsample = Y[::2,::2]

    plt.imshow(Y_subsample, cmap=plt.get_cmap('gray'))
    plt.title("Channel Y subsamplé")
    plt.show()

    plt.imshow(CB, cmap=plt.get_cmap('gray'))
    plt.title("Channel CB")
    plt.show()

    plt.imshow(CR, cmap=plt.get_cmap('gray'))
    plt.title("Channel CR")
    plt.show()

import unittest
import image
import ycbcr

class TestYCbCr(unittest.TestCase):

    def test_stuff(self):
        test_img = image.get_test_image()
        # plt.imshow(test_img)
        # plt.title("Image originale")
        # plt.show()

        ycbcr_img = ycbcr.rgb2ycbcr(test_img)
        img_back = ycbcr.ycbcr2rgb(ycbcr_img.astype('float'))
        # plt.imshow(img_back)
        # plt.title("Image encodée et décodée")
        # plt.show()


        # Y = ycbcr_img[:, :, 0]
        # CB = ycbcr_img[:, :, 1]
        # CR = ycbcr_img[:, :, 2]

        # Y_subsample = Y[::2, ::2]

        # plt.imshow(Y_subsample, cmap=plt.get_cmap('gray_r'))
        # plt.title("Channel Y subsamplé")
        # plt.show()

        # plt.imshow(CB, cmap=plt.get_cmap('gray_r'))
        # plt.title("Channel CB")
        # plt.show()

        # plt.imshow(CR, cmap=plt.get_cmap('gray_r'))
        # plt.title("Channel CR")
        # plt.show()

        self.assertEqual(img_back.all(), test_img.all())

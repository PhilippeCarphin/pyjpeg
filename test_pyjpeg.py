import unittest
import pyjpeg
import image
import matplotlib.pyplot as plt
import numpy as np

class TestPyjpeg(unittest.TestCase):

    def test_encode_decode(self):
        img_input = image.get_test_image()

        encoded = pyjpeg.encode('input_image.png')
        rgb_img = pyjpeg.decode(encoded)

        self.assertEqual(img_input.shape, rgb_img.shape)

        # I don't know what assert equals does but it sure doesn't return flase
        # and that is weird
        # self.assertEqual(img_input.all(), rgb_img.all())

        # print(rgb_img.dtype)

        # plt.imshow(rgb_img)
        # plt.title('compressed and uncompressed')
        # plt.show()
        # plt.imshow(img_input)
        # plt.title('Original image')
        # plt.show()

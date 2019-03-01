import unittest
import image

class TestImage(unittest.TestCase):

    def test_image(self):
        image.open_image_as_ndarray('input_image.png')

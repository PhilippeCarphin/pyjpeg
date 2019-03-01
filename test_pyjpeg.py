import unittest
import pyjpeg
import image
import matplotlib.pyplot as plt
import numpy as np


class TestPyjpeg(unittest.TestCase):

    def test_encode_decode(self):
        jpegobj = pyjpeg.JpegObject()
        encoded_decoded = jpegobj.encode_decode_file('input_image.png')

    def test_encode_decode_without_subsampling(self):
        jpegobj = pyjpeg.JpegObject(use_subasmpling=False)
        encoded_decoded = jpegobj.encode_decode_file('input_image.png')

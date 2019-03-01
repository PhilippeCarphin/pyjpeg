import unittest
import numpy as np
import subsampling
from itertools import product

class TestSubsampling(unittest.TestCase):

    def setUp(self):
        self.h = 18
        self.w = 20
        self.input_array = np.random.rand(self.h,self.w)
        self.input_img = np.random.rand(self.h, self.w, 3)

    def test_subsample(self):

        SSH = 2
        SSW = 4

        res = subsampling.subsample(self.input_array, SSH, SSW )

    def test_scheme_subsample(self):

        scheme = (4,2,0)

        res = subsampling.scheme_subsample(self.input_img, scheme)

        self.assertTrue(res['Y'].shape == (self.h,self.w))
        self.assertTrue(res['Cb'].shape == (self.h//2,self.w//2))
        self.assertTrue(res['Cr'].shape == (self.h//2,self.w//2))

        scheme = (4,2,0)

        res = subsampling.scheme_subsample(self.input_img, scheme)

        self.assertTrue(res['Y'].shape == (self.h,self.w))
        self.assertTrue(res['Cb'].shape == (self.h//2,self.w//2))
        self.assertTrue(res['Cr'].shape == (self.h//2,self.w//2))

    def test_upsample(self):

        up_h, up_w = 2, 4

        output_array = np.array(subsampling.upsample(self.input_array, up_h, up_w))

        self.assertTrue(output_array.shape == (up_h * self.h, up_w * self.w))

        def verify_ij(i,j):
            expected = self.input_array[i,j]
            for di, dj in product(range(up_h), range(up_w)):
                output = output_array[i*up_h + di, j*up_w + dj]
                self.assertTrue(
                    output == expected,
                    f"Failed at (i,j) == {(i,j)}, (di, dj) == {(di,dj)}")

        for i,j in product(range(self.h), range(self.w)):
            verify_ij(i,j)

    def test_one_way_invertible(self):
        up_h, up_w = 2, 4
        upsampled = np.array(subsampling.upsample(self.input_array, up_h, up_w))
        upsampled_subsampled = subsampling.subsample(upsampled, up_h, up_w)

        self.assertTrue(np.array_equal(upsampled_subsampled, self.input_array))








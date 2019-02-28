import unittest
import image
import numpy as np
import zigzag

zigzag_order = np.array(
    [[ 0,  1,  5,  6, 14, 15, 27, 28],
     [ 2,  4,  7, 13, 16, 26, 29, 42],
     [ 3,  8, 12, 17, 25, 30, 41, 43],
     [ 9, 11, 18, 24, 31, 40, 44, 53],
     [10, 19, 23, 32, 39, 45, 52, 54],
     [20, 22, 33, 38, 46, 51, 55, 60],
     [21, 34, 37, 47, 50, 56, 59, 61],
     [35, 36, 48, 49, 57, 58, 62, 63]])

class TestZigzag(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_zig_zag_indices(self):

        order = np.zeros((8,8))
        for ind, i in zip(zigzag.zig_zag_indices(8,8), range(8*8)):
            order[ind] = i

        self.assertTrue(np.array_equal(zigzag_order, order))


    def test_zig_zag_blocks(self):
        expected = np.array([0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10.,
               11., 12., 13., 14., 15., 16., 17., 18., 19., 20., 21.,
               22., 23., 24., 25., 26., 27., 28., 29., 30., 31., 32.,
               33., 34., 35., 36., 37., 38., 39., 40., 41., 42., 43.,
               44., 45., 46., 47., 48., 49., 50., 51., 52., 53., 54.,
               55., 56., 57., 58., 59., 60., 61., 62., 63., -0., -1.,
               -2., -3., -4., -5., -6., -7., -8., -9., -10., -11., -12.,
               -13., -14., -15., -16., -17., -18., -19., -20., -21., -22., -23.,
               -24., -25., -26., -27., -28., -29., -30., -31., -32., -33., -34.,
               -35., -36., -37., -38., -39., -40., -41., -42., -43., -44., -45.,
               -46., -47., -48., -49., -50., -51., -52., -53., -54., -55., -56.,
               -57., -58., -59., -60., -61., -62., -63.])
        blocks = np.zeros((2, 1, 8, 8))
        blocks[0, 0, :, :] = zigzag_order[:, :]
        blocks[1, 0, :, :] = -zigzag_order

        zz = zigzag.zig_zag_blocks(blocks)

        self.assertTrue(np.array_equal(zz, expected))

    def test_whole_thing_1(self):
        blocks = np.random.rand(2,1,8,8)

        zz = zigzag.zig_zag_blocks(blocks)
        un_zz = zigzag.un_zig_zag_blocks(zz, blocks.shape)

        self.assertTrue(np.array_equal(blocks, un_zz))

    def test_whole_thing_2(self):
        blocks = np.random.rand(2,3,8,8)

        zz = zigzag.zig_zag_blocks(blocks)
        un_zz = zigzag.un_zig_zag_blocks(zz, blocks.shape)

        self.assertTrue(np.array_equal(blocks, un_zz))

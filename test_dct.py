import unittest
import blocks
import dct


class TestDct(unittest.TestCase):

    def test_stuff(self):
        the_blocks = blocks.get_test_image_as_NxN_blocks()
        red_blocks, green_blocks, blue_blocks = blocks.split_channels(the_blocks)

        my_block = red_blocks[30, 40]
        encoded_block = dct.encode_dct(my_block)
        idct_block = dct.decode_dct(encoded_block)

        self.assertEqual(idct_block.all(), my_block.all())

        dct.dct_encode_blocks(red_blocks)
        dct.dct_basis_element(1, 2)

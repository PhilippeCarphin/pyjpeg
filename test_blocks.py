import unittest
import blocks
import zigzag
import numpy as np


class TestBlocks(unittest.TestCase):

    def test_blocks(self):
        arr = np.zeros((16, 16, 3))

        for ind, i in zip(zigzag.zig_zag_indices(16, 16), range(16 * 16)):
            arr[(ind) + (0,)] = i

        first_channel = arr[:, :, 0]
        three_channel_blks = blocks.split_NxN(arr)
        first_channel_blocks = three_channel_blks[:, :, :, :, 0]

        first_channel_combined = blocks.combine_NxN_channel(first_channel_blocks)

        print(first_channel_combined.shape)
        print(first_channel.shape)
        print(first_channel_combined[:8, :8])
        print(first_channel)

        self.assertEqual(arr[:, :, 0].all(), first_channel_combined.all())

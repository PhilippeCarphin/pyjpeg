import unittest
import huffman_8770 as huffman
import numpy as np
import blocks
import zigzag


class TestHuffman(unittest.TestCase):

    def setUp(self):
        self.input = np.random.randint(30, size=(20))

    def test_huffman_encode(self):
        encoded = huffman.huffman_encode(self.input)
        self.assertIn('data', encoded)
        self.assertIn('codebook', encoded)

    def test_huffman_codebook(self):
        block = blocks.get_one_test_NxN_block()
        line = zigzag.zig_zag_block(block)
        huff_code = huffman.get_huffman_codebook(line)
        h_line = []
        for s in line:
            h_line += [int(b) for b in huff_code[s]]

    def test_huffman_decode(self):
        encoded = huffman.huffman_encode(self.input)
        decoded = huffman.huffman_decode(encoded['data'], encoded['codebook'])
        decoded = np.array(list(decoded))

        self.assertTrue(np.array_equal(self.input, decoded))

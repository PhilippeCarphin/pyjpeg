import huffman
import zigzag
import numpy as np

def get_frequencies(big_line):
    frequencies = {}
    for symbol in big_line.astype('uint8'):
        if symbol not in frequencies:
            frequencies[symbol] = 1
        frequencies[symbol] += 1
    return frequencies.items()


if __name__ == '__main__':
    import blocks
    block = blocks.get_one_test_NxN_block()

    print(block.shape)

    line = zigzag.zig_zag_block(block)
    print(line)

    freqs = get_frequencies(line)
    print(freqs)

    huff_code = huffman.codebook(freqs)
    print(huff_code)

    h_line = []
    for s in line:
        h_line += [int(b) for b in huff_code[s]]

    print(h_line)

    print(f'len(line) = {len(line)}, h_line = {len(h_line)/8}')
    print(f'packbits result: {np.packbits(h_line)}')

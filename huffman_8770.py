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

def get_huffman_codebook(big_line):
    freqs = get_frequencies(big_line)
    huff_code = huffman.codebook(freqs)
    return huff_code


def huffman_encode(big_line):
    huff_code = get_huffman_codebook(big_line)
    h_line = []
    for s in big_line:
        h_line += [int(b) for b in huff_code[s]]
    return { 'data': h_line, 'codebook': huff_code}


def huffman_encode_packed(big_line):
    encoded = huffman_encode(big_line)
    encoded_packed = np.packbits(encoded['data'])

    return {
        'data': encoded_packed,
        'original_length': len(encoded['data']),
        'codebook': encoded['codebook']
    }


def huffman_decode_packed(packed_line):

    unpacked = np.unpackbits(packed_line['data'])
    unpacked_bits_list = list(unpacked[:packed_line['original_length']])
    return huffman_decode(unpacked_bits_list, packed_line['codebook'])


def huffman_decode(bit_list, codebook):
    reverse_code = {codebook[symbol]: symbol for symbol in codebook}

    chunk_list = []
    try:
        i = iter(bit_list)
        while True:
            bits = ''
            while bits not in reverse_code:
                bits += str(next(i))
            chunk_list.append(bits)
    except StopIteration:
        pass

    decoded = map(lambda bits : reverse_code[bits], chunk_list)

    return decoded


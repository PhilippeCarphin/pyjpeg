import numpy as np
import matplotlib.pyplot as plt


def zig_zag_block(block):
    line = []
    for ind in zig_zag_indices(block.shape[0], block.shape[1]):
        line.append(block[ind])

    return np.array(line)


def zig_zag_blocks(blocks):
    n_blocks_h, n_blocks_w, block_h, block_w = blocks.shape
    zig_zagged = np.zeros(n_blocks_h * n_blocks_w * block_w * block_h)
    block_wh = block_w * block_h
    for i in range(n_blocks_h):
        i_nhh = i * block_wh * n_blocks_w
        for j in range(n_blocks_w):
            start = i_nhh + j * block_h
            zig_zagged[start: start + block_wh] = zig_zag_block(blocks[i, j, :, :])
            pass

    return zig_zagged


def un_zig_zag_block(line, h, w):
    block = np.zeros((h, w))
    for ind, k in zip(zig_zag_indices(h, w), line):
        block[ind] = k

    return block


def un_zig_zag_blocks(line, n_blocks_h, n_blocks_w, block_h, block_w):
    blocks = np.zeros((n_blocks_h, n_blocks_w, block_h, block_w))
    block_wh = block_w * block_h
    for i in range(n_blocks_h):
        i_nhh = i * block_wh * n_blocks_w
        for j in range(n_blocks_w):
            start = i_nhh + j * block_h
            blocks[i, j, :, :] = un_zig_zag_block(line[start: start + block_wh], block_h, block_w)

    return blocks


def zig_zag_indices(h, w):
    assert h % 2 == 0 and w % 2 == 0

    i, j = 0, 0

    while i + 1 < h and j + 1 < w:

        yield (i, j)

        if i == 0 and j % 2 == 0:
            j += 1
        elif j == 0 and i % 2 != 0:
            i += 1
        else:
            if (i + j) % 2 == 0:
                i -= 1
                j += 1
            else:
                i += 1
                j -= 1

    while i < h and j < w:

        yield (i, j)

        if i + 1 == h and j % 2 == 0:
            j += 1
        elif j + 1 == w and i % 2 != 0:
            i += 1
        else:
            if (i + j) % 2 == 0:
                i -= 1
                j += 1
            else:
                i += 1
                j -= 1


if __name__ == "__main__":

    # from blocks import N

    N = 12

    a = un_zig_zag_block(np.array(range(N * N)), N, N)

    print(a.astype('int'))
    plt.imshow(a, cmap=plt.get_cmap('gray'))
    plt.show()

    big_a = np.zeros((2, 1, N, N))
    big_a[0, 0, :, :] = a[:, :]
    big_a[1, 0, :, :] = -a

    zz = zig_zag_block(a)
    un_zzd = un_zig_zag_block(zz, N, N).astype('int')
    assert un_zzd.all() == a.astype('int').all(), "Didn't work"

    zz = zig_zag_blocks(big_a)
    print(zz)
    un_zzd = un_zig_zag_blocks(zz, big_a.shape[0], big_a.shape[1], N, N)
    assert un_zzd.all() == big_a.all(), "didn't work"
    print(un_zzd)

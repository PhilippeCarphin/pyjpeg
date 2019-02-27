
import numpy as np
import matplotlib.pyplot as plt
def zig_zag_block(block):


    return np.zeros(64)


def zig_zag_indices(w, h):

    i, j = 0, 0

    while i+1 < h and j+1 < w:

        yield (i,j)

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

        yield (i,j)

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

    from blocks import N

    def zig_zag_one_block(block):
        line = np.zeros(block.shape[0] * block.shape[1])
    a = np.zeros((N,N))
    i = 0
    for ind in zig_zag_indices(N, N):
        a[ind] = i
        i += 1

    print(a.astype('int'))


    plt.imshow(a, cmap=plt.get_cmap('gray'))
    plt.show()


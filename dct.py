from image import open_image_as_ndarray
import blocks
import numpy as np
import scipy.fftpack as dctpack
import matplotlib.pyplot as plt
import quantize

def encode_dct(block_NxN):
    first_pass_dct = dctpack.dct(block_NxN, axis=0, norm='ortho')
    dct_block_NxN = dctpack.dct(first_pass_dct, axis=1, norm='ortho')
    return dct_block_NxN

def decode_dct(dct_block_NxN):
    first_pass_idct = dctpack.idct(dct_block_NxN, axis=0, norm='ortho')
    idct_block_NxN = dctpack.idct(first_pass_idct, axis=1, norm='ortho')
    return idct_block_NxN

def dct_basis_element(i,j):
    """ Returns the inverse discrete cosine transform of a canonical basis
    element (i.e. a vector with a '1' in a single position and zeroes everywhere
    else."""
    block = np.zeros((blocks.N,blocks.N))
    block[i,j] = 1
    first_pass_idct = dctpack.idct(block, axis=0, norm='ortho')
    idct_block = dctpack.idct(first_pass_idct, axis=1, norm='ortho')
    return idct_block

def dct_encode_blocks(blocks):
    # Le astype('float') est ben important ici, sinon les
    # tests ont l'air de pas marcher
    n_blocks_h = blocks.shape[0]
    n_blocks_w = blocks.shape[1]
    dct_blocks = np.empty_like(blocks).astype('float')
    for i in range(n_blocks_h):
        for j in range(n_blocks_w):
            dct_blocks[i,j,:,:] = encode_dct(blocks[i,j,:,:])
    return dct_blocks


if __name__ == "__main__":
    pass


# Accent, Accent_r, Blues, Blues_r, BrBG, BrBG_r, BugqGn, BuGn_r, BuPu, BuPu_r,\
# CMRmap, CMRmap_r, Dark2, Dark2_r, GnBu, GnBu_r, Greens, Greens_r, Greys,\
# Greys_r, OrRd, OrRd_r, Oranges, Oranges_r, PRGn, PRGn_r, Paired, Paired_r,\
# Pastel1, Pastel1_r, Pastel2, Pastel2_r, PiYG, PiYG_r, PuBu, PuBuGn, PuBuGn_r,\
# PuBu_r, PuOr, PuOr_r, PuRd, PuRd_r, Purples, Purples_r, RdBu, RdBu_r, RdGy,\
# RdGy_r, RdPu, RdPu_r, RdYlBu, RdYlBu_r, RdYlGn, RdYlGn_r, Reds, Reds_r, Set1,\
# Set1_r, Set2, Set2_r, Set3, Set3_r, Spectral, Spectral_r, Wistia, Wistia_r, \
# YlGn, YlGnBu, YlGnBu_r, YlGn_r, YlOrBr, YlOrBr_r, YlOrRd, YlOrRd_r, afmhot, \
# afmhot_r, autumn, autumn_r, binary, binary_r, bone, bone_r, brg, brg_r, bwr, \
# bwr_r, cividis, cividis_r, cool, cool_r, coolwarm, coolwarm_r, copper, copper_r,\
# cubehelix, cubehelix_r, flag, flag_r, gist_earth, gist_earth_r, gist_gray, \
# gist_gray_r, gist_heat, gist_heat_r, gist_ncar, gist_ncar_r, gist_rainbow, \
# gist_rainbow_r, gist_stern, gist_stern_r, gist_yarg, gist_yarg_r, gnuplot, \
# gnuplot2, gnuplot2_r, gnuplot_r, gray, gray_r, hot, hot_r, hsv, hsv_r, inferno, \
# inferno_r, jet, jet_r, magma, magma_r, nipy_spectral, nipy_spectral_r, ocean, \
# ocean_r, pink, pink_r, plasma, plasma_r, prism, prism_r, rainbow, rainbow_r, \
# seismic, seismic_r, spring, spring_r, summer, summer_r, tab10, tab10_r, tab20, \
# tab20_r, tab20b, tab20b_r, tab20c, tab20c_r, terrain, terrain_r, twilight, \
# twilight_r, twilight_shifted, twilight_shifted_r, viridis, viridis_r, winter, winter_r


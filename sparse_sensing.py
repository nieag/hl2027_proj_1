import pywt
import numpy as np
from scipy.ndimage.filters import convolve
import SimpleITK as sitk
import os
import matplotlib.pyplot as plt
from skimage.transform import resize
from scipy.stats.distributions import norm
from mpl_toolkits.mplot3d import Axes3D


def generate_2d_image(sz):
    X = np.linspace(-10, 10, sz)
    Y = X
    X,Y = np.meshgrid(X, Y)
    _im = np.zeros((sz, sz))
    _im[int(sz/2), int(sz/2)] = 255
    return X, Y, _im

def generate_3d_vol(sz, x_loc, y_loc, z_loc):
    X = np.linspace(-10, 10, sz)
    Y = X
    X,Y = np.meshgrid(X, Y)
    _vol = np.zeros((sz, sz, sz))
    _vol[x_loc, y_loc, z_loc] = 255

    return X, Y, _vol

def basic_psf(im):
    _im = np.copy(im) # local copy for transforms
    _im_fft = np.fft.fft2(_im)
    _rd_us = np.zeros(_im.shape)
    _im_fft_rd_us = np.copy(_im_fft)#
    mask = np.random.randint(0,2,size=_im.shape).astype(np.bool)

    # use your mask to replace values in your input array
    _im_fft_rd_us[mask] = _rd_us[mask]
    _im_rd_us = np.fft.ifft2(_im_fft_rd_us)
    return _im_fft, _im_fft_rd_us, _im_rd_us

def TPSF_2DFT(vol, fft_axes):
    """ Undersamples every slice (z-axis) using the same pattern"""
    _vol = np.copy(vol)
    W, H, D = _vol.shape
    _rd_us = np.zeros((H, D))
    mask = np.random.randint(0,2,size=(H, D)).astype(np.bool)
    _vol_fft = np.fft.fftn(_vol, axes=fft_axes)
    _vol_fft_rd_us = np.copy(_vol_fft)
    _vol_fft_rd_us[:, mask] = _rd_us[mask]
    _vol_rd_us = np.fft.ifftn(_vol_fft_rd_us, axes=fft_axes)

    return _vol_rd_us

def TPSF_multi_slice_2DFT(vol, fft_axes):
    """Undersamples every slice (z-axis) using a different pattern"""
    _vol = np.copy(vol)
    W, H, D = _vol.shape
    _rd_us = np.zeros((H, D))
    _vol_fft = np.fft.fftn(_vol, axes=fft_axes)
    _vol_fft_rd_us = np.copy(_vol_fft)
    for i in range(D):
        mask = np.random.randint(0,2,size=(H, D)).astype(np.bool)
        _vol_fft_rd_us[i, mask] = _rd_us[mask]

    _vol_rd_us = np.fft.ifftn(_vol_fft_rd_us, axes=fft_axes)


    return _vol_rd_us

def TPSF_3DFT(vol):
    _vol = np.copy(vol)
    W, H, D = _vol.shape
    _rd_us = np.zeros((H, D))
    _vol_fft = np.fft.fftn(_vol, axes=(0, 1, 2))
    _vol_fft_rd_us = np.copy(_vol_fft)
    for i in range(D):
        mask = np.random.randint(0,2,size=(H, D)).astype(np.bool)
        _vol_fft_rd_us[i, mask] = _rd_us[mask]

    _vol_rd_us = np.fft.ifftn(_vol_fft_rd_us, axes=(0, 1, 2))

    return _vol_rd_us

def plot_surface(X, Y, images):
    for i, im in enumerate(images):
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        surf = ax.plot_surface(X, Y, im, color='r')

def dwt2(_im, _lp, _hp, levels):
    """
    Perform 2D discrete wavelet transform on given image using given low pass and high pass filters.

    :param _im: 2D input image
    :type _im: np.ndarray
    :param _lp: 1D filter kernel; lowpass
    :type _lp: np.ndarray
    :param _hp: 1D filter kernel; highpass
    :type _hp: np.ndarray
    :param levels: number of decomposition levels
    :type levels: int
    :return: DWT coefficients
    :rtype: np.ndarray
    """

    # ensure lowpass and highpass kernels have the expected shape #
    _lp = _lp.reshape(-1, 1)
    _hp = _hp.reshape(-1, 1)

    # extract shape of input image #
    (M, N) = _im.shape
    (M2, N2) = map(lambda x: int(np.ceil(x/2)), (M, N))

    # initialise output image (same size as input) #
    _dwt2d = np.zeros_like(_im)

    # 1st STAGE: convolution along 1st AXIS #
    _tmp_l = convolve(_im, _lp)[::2, :]  # TODO: perform low pass filtering (1st axis) #
    _tmp_h = convolve(_im, _hp)[::2, :]  # perform high pass filtering (1st axis) #

    # 2nd STAGE: convolution along 2nd AXIS #
    _dwt2d[:M2, :N2] = convolve(_tmp_l, _lp.transpose())[:, ::2]  # LOW pass   ->  LL (top left) #
    _dwt2d[:M2, N2:] = convolve(_tmp_l, _hp.transpose())[:, ::2]  # HIGH pass  ->  LH (top right) #  # TODO: high pass filtering (2nd axis) #
    _dwt2d[M2:, :N2] = convolve(_tmp_h, _lp.transpose())[:, ::2]  # LOW pass   ->  HL (bottom left) #  # TODO: low pass filtering (2nd axis) #
    _dwt2d[M2:, N2:] = convolve(_tmp_h, _hp.transpose())[:, ::2]  # HIGH pass  ->  HH (bottom right) #  # TODO: high pass filtering (2nd axis) #

    # compute further decomposition levels for approximation part (i.e. LL aka 'top left') #
    if levels > 1:
        _dwt2d[:M2, :N2] = dwt2(_dwt2d[:M2, :N2], _lp, _hp, levels=levels-1)  # recursive call for LL #

    return _dwt2d

def idwt2(_dwt2d, _lp, _hp, levels):
    """
    Reconstruct a 2D image from given `levels`-level discrete wavelet transform using given filter kernels.

    :param _dwt2d: 2D discrete wavelet transform
    :type _dwt2d: np.ndarray
    :param _lp: 1D filter kernel; lowpass
    :type _lp: np.ndarray
    :param _hp: 1D filter kernel; highpass
    :type _hp: np.ndarray
    :param levels: number of decomposition levels
    :type levels: int
    :return: reconstructed image
    :rtype: np.ndarray
    """

    # obtain shape of input data, i.e. wavelet coefficients #
    (M, N) = _dwt2d.shape
    (M2, N2) = map(lambda x: int(np.ceil(x/2)), (M, N))

    # copy input data to be able to modify the array (for recursion) #
    _dwt2d = _dwt2d.copy()

    # reconstruct the approximation part first (i.e. LL aka 'top left') #
    if levels > 1:
        _dwt2d[:M2, :N2] = idwt2(_dwt2d[:M2, :N2], _lp, _hp, levels=levels-1)  # recursive call for LL #

    # ensure filter kernels have expected shape #
    _lp = _lp.reshape(-1, 1)
    _hp = _hp.reshape(-1, 1)

    # initialise output image #
    _im = np.zeros_like(_dwt2d)



    # ###
    # 1st STAGE: convolution along 2nd AXIS (revert steps of forward transform) #
    # ###

    # initialise temporary reconstruction arrays #
    _tmp_l = np.zeros_like(_dwt2d)
    _tmp_ll, _tmp_lh = np.zeros((M2, N)), np.zeros((M//2, N))

    # ZERO FILLING of LL and LH #
    _tmp_ll[:, 1::2] = _dwt2d[:M2, :N2]  # LL aka 'top left' #
    _tmp_lh[:, 1::2] = _dwt2d[:M2, N2:]  # LH aka 'top right' #

    # CONVOLUTION and summation #
    _tmp_l[1::2, :] = convolve(_tmp_ll, _lp.transpose())  # revert LOW pass #
    _tmp_l[1::2, :] += convolve(_tmp_lh, _hp.transpose())  # revert HIGH pass #  # TODO: perform high pass filtering (2nd axis) #

    # initialise temporary reconstruction arrays #
    _tmp_h = np.zeros_like(_dwt2d)
    _tmp_hl, _tmp_hh = np.zeros((M2, N)), np.zeros((M2, N))

    # ZERO FILLING of HL and HH #
    _tmp_hl[:, 1::2] = _dwt2d[M2:, :N2]  # HL aka 'bottom left' #
    _tmp_hh[:, 1::2] = _dwt2d[M2:, N2:]  # HH aka 'bottom right' #

    # CONVOLUTION and summation #
    _tmp_h[1::2, :] = convolve(_tmp_hl, _lp.transpose())  # revert LOW pass #  # TODO: perform low pass filtering (2nd axis) #
    _tmp_h[1::2, :] += convolve(_tmp_hh, _hp.transpose())  # revert HIGH pass #  # TODO: perform high pass filtering (2nd axis) #

    # ###
    # 2nd STAGE: convolution along 1st AXIS (revert steps of forward transform) #
    # ###
    _im += convolve(_tmp_l, _lp)  # revert LOW pass #  # TODO: perform low pass filtering (1st axis) #
    _im += convolve(_tmp_h, _hp)  # revert HIGH pass #  # TODO: perform high pass filtering (1st axis) #

    return _im

if __name__ == '__main__':

    """2D undersampling"""
    # X, Y, im = generate_2d_image(64)
    # wavelet_im = np.zeros(im.shape)
    # plot_surface(X, Y, [im])
    #
    # wl_name = 'haar'  # TODO: vary type of wavelet #vol_img = sitk.Image(sz,sz,sz,sitk.sitkUInt8)
    # lp_d, hp_d, lp_r, hp_r = map(np.array, pywt.Wavelet(wl_name).filter_bank)
    # # number of decomposition levels #mask = np.random.randint(0,2,size=(H, D)).astype(np.bool)
    # num_levels = 2  # TODO: vary number of decomposition levels #
    #
    # # apply discrete wavelet tran_fftsform (DWT) #
    # dwt = dwt2(im, lp_d, hp_d, levels=num_levels)  # TODO: call wavelet transform #
    # (M, N) = dwt.shape
    # (M2, N2) = map(lambda x: int(np.ceil(x/2)), (M, N))
    # wavelet_im[M2:, N2:] = dwt[M2:, N2:] # partial wavelet image
    #
    # idwt_im = idwt2(wavelet_im, lp_r, hp_r, levels=num_levels)
    #
    # wl_im_fft, wl_im_fft_rd_us, wl_im_rd_us = basic_psf(idwt_im)
    #
    # wavelet_im_rd_us = dwt2(np.real(wl_im_rd_us), lp_d, hp_d, levels=num_levels)
    # plot_surface(X, Y, [wavelet_im_rd_us])
    # plt.show()

    """Volumetric undersampling"""
    X, Y, vol = generate_3d_vol(32, 16, 16, 16)
    plot_surface(X, Y, [vol[16, :, :]])
    N = len(vol)

    (M, N, D) = vol.shape

    wavelet_slice = np.zeros((M, N))
    (M2, N2) = map(lambda x: int(np.ceil(x/2)), (M, N))
    wl_name = 'haar'  # TODO: vary type of wavelet #vol_img = sitk.Image(sz,sz,sz,sitk.sitkUInt8)
    lp_d, hp_d, lp_r, hp_r = map(np.array, pywt.Wavelet(wl_name).filter_bank)
    #
    # number of decomposition levels #mask = np.random.randint(0,2,size=(H, D)).astype(np.bool)
    num_levels = 2  # TODO: vary number of decomposition levels #
    dwt = dwt2(vol[16, :, :], lp_d, hp_d, levels=num_levels)
    wavelet_slice[M2:, N2:] = dwt[M2:, N2:]
    im_slice = idwt2(wavelet_slice, lp_r, hp_r, levels=num_levels)
    vol[16, :, :] = im_slice
    plot_surface(X, Y, [vol[16, :, :]])

    """Single-slice undersampling 2DFT"""
    # vol_rd_us_1D_im = TPSF_2DFT(vol, (0,1))
    # plot_surface(X, Y, [vol_rd_us_1D_im[16, :, :]])
    # vol_rd_us_1d_wave = np.zeros(vol_rd_us_1D_im.shape)
    # for i in range(M):
    #     vol_rd_us_1d_wave[i, :, :] = dwt2(np.real(vol_rd_us_1D_im[i, :, :]), lp_r, hp_r, levels=num_levels)
    # plot_surface(X, Y, [vol_rd_us_1d_wave[16, :, :]])

    """Multi-slice undersampling 2DFT"""
    # vol_rd_us_multi_im = TPSF_multi_slice_2DFT(vol, (0, 1))
    # plot_surface(X, Y, [vol_rd_us_multi_im[16, :, :]])
    # vol_rd_us_multi_wave = np.zeros(vol_rd_us_multi_im.shape)
    # for i in range(M):
    #     vol_rd_us_multi_wave[i, :, :] = dwt2(np.real(vol_rd_us_multi_im[i, :, :]), lp_r, hp_r, levels=num_levels)
    # plot_surface(X, Y, [vol_rd_us_multi_wave[16, :, :]])

    """3DFT"""
    vol_rd_us_3D_im = TPSF_3DFT(vol)
    plot_surface(X, Y, [vol_rd_us_3D_im[16, :, :]])
    vol_rd_us_3D_wave = np.zeros(vol_rd_us_3D_im.shape)
    for i in range(M):
        vol_rd_us_3D_wave[i, :, :] = dwt2(np.real(vol_rd_us_3D_im[i, :, :]), lp_r, hp_r, levels=num_levels)
    plot_surface(X, Y, [vol_rd_us_3D_wave[16, :, :]])

    plt.show()

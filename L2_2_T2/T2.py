"""
This module contains all contents for *Task 2* of lab *L2.2*.

**author**: daniel.jorgens@sth.kth.se

**intended time**: 1.5 hrs

**additional requirements** (to install with pip): *PyWavelets*

In this task you will investigate the discrete wavelet transform for 2D image data. **Briefly** familiarise
yourself with how the functions `dwt2` and `idwt2` use the low pass and high pass filter kernels in a cascading
way in order to arrive at the four `subimages` (referred to as *LL*, *LH*, *HL*, *HH*) after a single level of decomposition.

**a**. Do you understand how these four `subimages` are obtained? If not, we are happy to explain! :)

**b**. Vary the type of wavelet kernels. Which do you find (http://pywavelets.readthedocs.io/en/latest/ref/index.html)?
Are there differences in the result?

**c**. Vary the number of decomposition levels. What happens when increasing the decomposition level by one?

**d**. Add noise to the original image and have a look at the wavelet coefficients. Create a mask for these with
a chosen threshold. Compare the masks for the wavelet coefficients in the noisy and unnoisy case. What do you see?

**e**. Implement two thresholding approaches: *hard thresholding* and *soft thresholding*. Compare the results after
applying these functions on the wavelet coefficients of the noisy image data for different thresholds and optimise.
Which approach works best?

**f**. Use hard thresholding on the original image without noise and vary the threshold. How many wavelet coefficients
are enough to maintain a good image representation? (This is the basis for compression!)

"""


import pywt
import numpy as np
from scipy.ndimage.filters import convolve
import SimpleITK as sitk
import os
import matplotlib.pyplot as plt
from skimage.transform import resize
from scipy.stats.distributions import norm


# define global settings for all plots #
plot_ops = dict(cmap="gray")


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

def soft_thresh(_im, _thresh):
    """
    Apply soft (lower) threshold mask to given image data.

    :param _im: 2D image data
    :type _im: np.ndarray
    :param _thresh: scalar threshold parameter
    :type _thresh: float
    :return: thresholded image
    :rtype: np.ndarray
    """

    # mask image #
    _mask = _im>_thresh  # TODO: mask image data #

    return _mask * (np.abs(_im) - _thresh) * np.sign(_im)


def hard_thresh(_im, _thresh):
    """
    Apply hard threshold mask to given image data.

    :param _im: 2D image data
    :type _im: np.ndarray
    :param _thresh: scalar threshold parameter
    :type _thresh: float
    :return: thresholded image
    :rtype: np.ndarray
    """

    # mask image #
    _mask = _im>_thresh  # TODO: mask image data #

    # compute ratio of non-zero coefficients #
    _ratio = np.sum(_mask) / _mask.size

    return _mask * _im, _ratio


if __name__ == "__main__":

    # define test image #
    im = sitk.ReadImage(os.path.abspath('image_head.dcm'))
    im = sitk.GetArrayViewFromImage(im).squeeze().astype(np.float)

    # add noise to input image #
    im += norm.rvs(scale=2, size=im.shape)  # TODO: add noise to image #
    print(pywt.wavelist())
    # define LOW and HIGH pass filter kernels (for decomposition and reconstruction) #
    wl_name = 'haar'  # TODO: vary type of wavelet #
    lp_d, hp_d, lp_r, hp_r = map(np.array, pywt.Wavelet(wl_name).filter_bank)

    # number of decomposition levels #
    num_levels = 2  # TODO: vary number of decomposition levels #

    # padding to avoid border effects #
    padding = num_levels * 2 * len(lp_d)

    # resize the image (convenience for testing the code) #
    sz = (1024, 1024)
    im = resize(im, output_shape=tuple(map(lambda x: x - 2 * padding, sz)))

    # apply padding to image #
    im_pad = np.pad(im, padding, mode='constant')
    M_pad, N_pad = im_pad.shape

    # apply discrete wavelet transform (DWT) #
    dwt = dwt2(im_pad, lp_d, hp_d, levels=num_levels)  # TODO: call wavelet transform #

    # apply inverse discrete wavelet transform (iDWT) #
    im_reco = idwt2(dwt, lp_r, hp_r,levels=num_levels)  # TODO: call inverse wavelet transform #
    im_reco = im_reco[padding: (M_pad - padding), padding: (N_pad - padding)]  # remove padding #

    # PLOTTING #
    f = plt.figure()

    ax1 = plt.subplot(2, 2, 1)
    p = plt.imshow(im, **plot_ops)
    ax1.set_title('Original')
    f.colorbar(p)

    ax2 = plt.subplot(2, 2, 2)
    p = plt.imshow(im-im_reco, **plot_ops)
    ax2.set_title('original - im_reco')
    f.colorbar(p)

    ax3 = plt.subplot(2, 2, 3)
    p = plt.imshow(np.log(dwt), **plot_ops)
    ax3.set_title('DWT ({}, {} levels)'.format(wl_name, num_levels))
    f.colorbar(p)

    ax4 = plt.subplot(2, 2, 4)
    p = plt.imshow(im_reco, **plot_ops)
    ax4.set_title('im_reco')
    f.colorbar(p)

    plt.show()

    # #######
    # Thresholding
    # ########

    thresh = np.mean(dwt) # TODO: vary threshold #

    # apply thresholding to wavelet coefficients #
    dwt_s = soft_thresh(dwt,thresh)# ... #(dwt, thresh)  # TODO: apply soft thresholding #
    dwt_h, ratio = hard_thresh(dwt,thresh)  # TODO: apply hard thresholding #

    # apply inverse discrete wavelet transform (iDWT) #
    im_reco_s = idwt2(dwt_s, lp_r, hp_r, levels=num_levels)  # TODO: call inverse wavelet transform (after soft thresholding)#
    im_reco_s = im_reco_s[padding: (M_pad - padding), padding: (N_pad - padding)]  # remove padding #
    im_reco_h = idwt2(dwt_h, lp_r, hp_r, levels=num_levels)  # TODO: call inverse wavelet transform (after hard thresholding) #
    im_reco_h = im_reco_h[padding: (M_pad - padding), padding: (N_pad - padding)]  # remove padding #
    # PLOTTING #
    f = plt.figure()

    ax1 = plt.subplot(2, 2, 1)
    p = plt.imshow((dwt_h > 0).astype(float), **plot_ops)
    ax1.set_title('hard threshold: {} (non-zero coefficients: {:2.2f}%)'.format(thresh, ratio*100))
    f.colorbar(p)

    ax2 = plt.subplot(2, 2, 2)
    p = plt.imshow(np.log(dwt_s.astype(float)), **plot_ops)
    ax2.set_title('soft threshold: {}'.format(thresh))
    f.colorbar(p)

    ax3 = plt.subplot(2, 2, 3)
    p = plt.imshow(im_reco_h, **plot_ops)
    ax3.set_title('im_reco_h')
    f.colorbar(p)

    ax4 = plt.subplot(2, 2, 4)
    p = plt.imshow(im_reco_s, **plot_ops)
    ax4.set_title('im_reco_s')
    f.colorbar(p)

    plt.show()

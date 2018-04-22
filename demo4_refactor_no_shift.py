import skimage
import numpy as np
from numpy.fft import *
import matplotlib.pyplot as plt
import os
import scipy.io as spio
import pywt
from scipy.ndimage.filters import convolve
from scipy.ndimage.interpolation import rotate
import SimpleITK as sitk
from skimage.transform import resize
from scipy.stats.distributions import norm

def read_data(path):
    mat = spio.loadmat(path, squeeze_me=True)

    _im = mat['im']
    _mask_unif = mat['mask_unif']
    _mask_vardens = mat['mask_vardens']
    _pdf_unif = mat['pdf_unif']
    _pdf_vardens = mat['pdf_vardens']

    return _im, _mask_unif, _mask_vardens, _pdf_unif, _pdf_vardens

def linear_recon(image, mask, pdf):
    _im = np.copy(image)
    _im_fft = fftshift(fft2(_im))
    _im_fft_us = np.divide(np.multiply(_im_fft, mask), pdf)
    _im_us = ifft2(_im_fft_us)

    return _im_us

def fft2c(image):
    """
    Apply 2d fourier transform to image

    """
    _im = np.copy(image)
    _im_freq = fftshift(fft2(ifftshift(_im)))

    return _im_freq

def ifft2c(image):
    """
    Apply inverse 2d fourier transform to image

    """
    _im = np.copy(image)
    _im_freq = fftshift(ifft2(ifftshift(_im)))

    return _im_freq

def generate_mask(im, type):
    '''
    Generate binary mask according to type
    # TODO: Add more patterns
    '''

    h, w = im.shape
    if type == 'uniform':
        return np.random.choice([0, 1], [h, w], p = [0.7, 0.3])
    elif type == 'ylines':
        _rows = np.random.randint(0, h, [1, 50])
        _mask = np.zeros(im.shape)
        _mask[_rows, :] = 1
        return _mask
    elif type == 'xlines':
        _columns = np.random.randint(0, w, [1, 50])
        _mask = np.zeros(im.shape)
        _mask[:, _columns] = 1
        return _mask
    elif type == 'fakeradials':
        _mask = np.zeros([h, w])
        _mask[int(h/2 - 3):int(h/2 + 3), 50:w-50] = 1
        _mask[50:h-50, int(w/2 - 3):int(w/2 + 3)] = 1
        for i in range(1, 6):
            _rot = rotate(_mask, 20 + i*10, reshape=False)
            _mask = _mask + _rot
        _mask[np.where(_mask > 0.5)] = 1
        return _mask

def POCS_algorithm(image, mask, pdf, thresh, filter_type="db1", n_iter=15):
    _im = np.copy(image)
    _DATA = np.multiply(fft2c(_im), mask)
    _DATA = np.fft.fftshift(_DATA) # Why is this line necessary?
    _im_cs = np.fft.ifft2(np.divide(_DATA, pdf))
    _im_cs = _im
    lp_d, hp_d, lp_r, hp_r = map(np.array, pywt.Wavelet(filter_type).filter_bank)

    plt.figure()
    for i in range(n_iter):
        _im_cs = dwt2(np.abs(_im_cs), lp_d, hp_d, levels=1) # Apply wavelet transform
        _im_cs = new_soft_thresh(_im_cs, thresh) # Soft thresholding on the wavelet transform to retrieve high frequency content
        _im_cs = idwt2(_im_cs, lp_r, hp_r, levels=1) # Inverse wavelet transform using compressed version
        _im_cs = ifft2c(np.multiply(fft2c(_im_cs), 1 - mask) + _DATA) # Undersample the fourier spectra of the compressed version of _im_cs, add base _DATA
        plt.imshow(np.abs(_im_cs), cmap='gray')
        plt.title("Iteration: {}".format(i))
        plt.pause(0.5)
        #plt.show()

    # Plot for comparison
    plt.figure(figsize = (10, 10))
    plt.subplot(1, 3, 1)
    plt.imshow(np.abs(image), cmap='gray')
    plt.title("image")
    plt.subplot(1, 3, 2)
    plt.imshow(np.abs(linear_recon(image, mask, pdf)), cmap='gray')
    plt.title("im_us")
    plt.subplot(1, 3, 3)
    plt.imshow(np.abs(_im_cs), cmap='gray')
    plt.title("im_cs")

def new_soft_thresh(_im, _thresh):
    # y = (abs(x) > lambda).*(x.*(abs(x)-lambda)./(abs(x)+eps));
    _left_side = np.abs(_im) > _thresh
    _right_side = np.divide(np.multiply(_im, np.abs(_im)-_thresh), np.abs(_im) + np.finfo(float).eps)
    _y = np.multiply(_left_side, _right_side)
    return _y

def hard_thresh(_im, _thresh):
    '''
    # TODO: Fix return type

    '''
    return np.abs(_im) > _thresh

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
    im, mask_unif, mask_vardens, pdf_unif, pdf_vardens = read_data('./workshop/brain.mat')
    im_us = linear_recon(im, mask_unif, pdf_vardens)

    # _DATA = np.multiply(np.fft.fft2(im), mask_vardens)
    # plt.imshow(np.abs(_DATA), cmap="gray")
    #POCS_algorithm(im, mask_unif, pdf_unif, 0.025)
    # test = new_soft_thresh(im, 0.025)
    # plt.imshow(np.abs(im_us), cmap='gray')

    mask = generate_mask(im, 'xlines')
    POCS_algorithm(im, mask, pdf_unif, 0.025)
    plt.figure()
    plt.imshow(np.abs(mask), cmap='gray')
    plt.show()

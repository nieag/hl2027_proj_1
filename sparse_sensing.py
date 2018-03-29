import pywt
import numpy as np
from scipy.ndimage.filters import convolve
import SimpleITK as sitk
import os
import matplotlib.pyplot as plt
from skimage.transform import resize
from scipy.stats.distributions import norm
from mpl_toolkits.mplot3d import Axes3D


def generate_image(size):
    X = np.linspace(-10, 10, size)
    Y = X
    X,Y = np.meshgrid(X, Y)
    _im = np.zeros((size, size))
    _im[int(size/2), int(size/2)] = 255
    return X, Y, _im

def basic_psf(im):
    _im = np.copy(im) # local copy for transforms
    _im_fft = np.fft.fft2(_im)
    _rd_us = np.zeros(_im.shape)
    _im_fft_rd_us = np.copy(_im_fft)
    mask = np.random.randint(0,2,size=_im.shape).astype(np.bool)

    # use your mask to replace values in your input array
    _im_fft_rd_us[mask] = _rd_us[mask]
    _im_rd_us = np.fft.ifft2(_im_fft_rd_us)
    print(_im_rd_us)
    return _im_fft, _im_fft_rd_us, _im_rd_us

def plot_surface(X, Y, im):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(X, Y, im, color='r')

if __name__ == '__main__':
    X, Y, image = generate_image(32)

    plot_surface(X, Y, image)


    im_fft, im_fft_rd_us, im_rd_us = basic_psf(image)


    plt.show()

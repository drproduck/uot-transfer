# -*- coding: utf-8 -*-
import torch
import numpy as np
import matplotlib.pylab as plt
import ot
from codebase.OT import sampling_sinkhorn_divergence
from codebase.UOT import sampling_sinkhorn_uot
from transfer import transfer_with_map

# from skimage.io import imread
# from gSLICrPy import __get_CUDA_gSLICr__, CUDA_gSLICr
import colorsys
from PIL import Image

r = np.random.RandomState(42)

method = 'uot'

def im2mat(img):
    """Converts an image to matrix (one pixel per line)"""
    img_flat = img.reshape((img.shape[0] * img.shape[1], img.shape[2]))
    return img_flat

def normalize_hsv(img):
    img[0] = img[0] / 179
    img[1] = img[1] / 255
    img[2] = img[2] / 255
    return img

def denormalize_hsv(imgarr):
    imgarr[0] = np.uint8(imgarr[0] * 179)
    imgarr[1] = np.uint8(imgarr[1] * 255)
    imgarr[2] = np.uint8(imgarr[2] * 255)
    return imgarr

def mat2im(X, shape):
    """Converts back a matrix to an image"""
    return X.reshape(shape)


def minmax(img):
    return np.clip(img, 0, 1)


##############################################################################
# Generate data
# -------------

# Loading images
# I1 = plt.imread('data_pot/ocean_day.jpg').astype(np.float64) / 256
# I2 = plt.imread('data_pot/ocean_sunset.jpg').astype(np.float64) / 256

# I1 = plt.imread('data_pot/city_night.jpg').astype(np.float64)
# I2 = plt.imread('data_pot/city_day.jpg').astype(np.float64)

I1 = np.array(Image.open('data_pot/city_night.jpg', mode='RGB').convert('HSV'), dtype=np.float32)
I2 = np.array(Image.open('data_pot/city_day.jpg', mode='RGB').convert('HSV'), dtype=np.float32)

I1 = normalize_hsv(I1) # should be in [0,1]
I2 = normalize_hsv(I2)

X1 = torch.from_numpy(im2mat(I1)).cuda()
X2 = torch.from_numpy(im2mat(I2)).cuda()

# training sampltes
nb = 10000
idx1 = r.randint(X1.shape[0], size=(nb,))
idx2 = r.randint(X2.shape[0], size=(nb,))

Xs = X1[idx1, :]
Xt = X2[idx2, :]



##############################################################################
# Instantiate the different transport algorithms and fit them
# -----------------------------------------------------------


# SinkhornTransport

if method == 'ot':
    _, P = sampling_sinkhorn_divergence(Xs, Xt, ret_plan=True)
elif method == 'uot':
    P = sampling_sinkhorn_uot(Xs, Xt, eta=0.1, t1=1., t2=1., n_iter=100)

# prediction between images (using out of sample prediction as in [6])
transp_Xs = transfer_with_map(Xt, P, X1, Xs)

transp_Xs = transp_Xs.detach().cpu().numpy()
I1te = minmax(mat2im(transp_Xs, I1.shape))

##############################################################################
# Plot new images
# ---------------

def torgbarray(img):
    return np.array(Image.fromarray(denormalize_hsv(img), mode='HSV').convert('RGB')) / 255

plt.figure(figsize=(16, 8))

plt.subplot(2, 3, 1)
plt.imshow(torgbarray(I1))
plt.axis('off')
plt.title('Image 1')

plt.subplot(2, 3, 2)
plt.imshow(torgbarray(I2))
plt.axis('off')
plt.title('Image 2')

plt.subplot(2, 3, 3)
plt.imshow(torgbarray(I1te))
plt.axis('off')
plt.title('Image 1 Adapt (reg)')

plt.tight_layout()

plt.savefig(f'hello_{method}.png')


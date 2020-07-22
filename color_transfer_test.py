# -*- coding: utf-8 -*-
import torch
import numpy as np
import matplotlib.pylab as plt
import ot
from codebase.OT import sampling_sinkhorn_divergence
from codebase.UOT import sampling_sinkhorn_uot
from transfer import transfer_with_map
import time
from skimage.exposure import equalize_adapthist

# from skimage.io import imread
# from gSLICrPy import __get_CUDA_gSLICr__, CUDA_gSLICr
import colorsys
from PIL import Image

from skimage.segmentation import slic 
r = np.random.RandomState(1000)

method = 'uot'

def im2mat(img):
    """Converts an image to matrix (one pixel per line)"""
    img_flat = img.reshape((img.shape[0] * img.shape[1], img.shape[2]))
    return img_flat

# def normalize_hsv(img):
#     img[0] = img[0] / 179
#     img[1] = img[1] / 255
#     img[2] = img[2] / 255
#     return img

# def denormalize_hsv(imgarr):
#     imgarr[0] = np.uint8(imgarr[0] * 179)
#     imgarr[1] = np.uint8(imgarr[1] * 255)
#     imgarr[2] = np.uint8(imgarr[2] * 255)
#     return imgarr

def denormalize_rgb(imgarr):
    return np.uint8(imgarr * 255)

def mat2im(X, shape):
    """Converts back a matrix to an image"""
    return X.reshape(shape)

def minmax(img):
    return np.clip(img, 0, 1)


##############################################################################
# Generate data
# -------------
img1 = 'data_pot/city_day.jpg'
img2 = 'data_pot/city_night.jpg'

time_s = time.time()
# Loading images
I1 = plt.imread(img1).astype(np.float32) / 255
I2 = plt.imread(img2).astype(np.float32) / 255

print(f'loading elapsed={time.time() - time_s:.3f}')
# time_s = time.time()
# slic(I1, n_segments=1000)
# print(f'segment elapsed={time.time() - time_s:.3f}')

time_s = time.time()
X1 = torch.from_numpy(im2mat(I1))
X2 = torch.from_numpy(im2mat(I2))
print(f'convert elapsed={time.time() - time_s:.3f}')

# training samples
nb = 10000
idx1 = r.randint(X1.shape[0], size=(nb,))
idx2 = r.randint(X2.shape[0], size=(nb,))

Xs = X1[idx1, :]
Xt = X2[idx2, :]
time_s = time.time()
Xs = Xs.to(0)
Xt = Xt.to(0)
print(f'sampling elapsed={time.time() - time_s:.3f}')

##############################################################################
# Instantiate the different transport algorithms and fit them
# -----------------------------------------------------------

# SinkhornTransport

time_s = time.time()
if method == 'ot':
    _, P = sampling_sinkhorn_divergence(Xs, Xt, eta=0.01, ret_plan=True)
elif method == 'uot':
    P = sampling_sinkhorn_uot(Xs, Xt, eta=0.01, t1=10., t2=1., n_iter=100)

print(f'sinkhorn elapsed={time.time() - time_s:.3f}')

time_s = time.time()
# prediction between images (using out of sample prediction as in [6])
transp_Xs = transfer_with_map(Xt, P, X1, Xs, batch_size=100000)
print(f'transfer elapsed={time.time() - time_s:.3f}')

transp_Xs = transp_Xs.detach().cpu().numpy()
I1te = minmax(mat2im(transp_Xs, I1.shape))

# I1te = denormalize_rgb(I1te)
# I1te = np.array(equalize_adapthist(Image.fromarray(I1te, mode='RGB'))) / 255
# I1te = equalize_adapthist(I1te)

##############################################################################
# Plot new images
# ---------------

plt.figure(figsize=(16, 8))

plt.subplot(2, 3, 1)
plt.imshow(I1)
plt.axis('off')
plt.title('Image 1')

plt.subplot(2, 3, 2)
plt.imshow(I2)
plt.axis('off')
plt.title('Image 2')

plt.subplot(2, 3, 3)
plt.imshow(I1te)
plt.axis('off')
plt.title('Image 1 Adapt (reg)')

plt.tight_layout()

import os

def path2name(path_img):
    name = os.path.splitext(path_img)[0]
    return name.split('/')[1]

plt.savefig(f'color_transfer_{method}_source={path2name(img1)}_target={path2name(img2)}.png')
plt.savefig(f'color_transfer_{method}_source={path2name(img1)}_target={path2name(img2)}.png')
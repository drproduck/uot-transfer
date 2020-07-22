# -*- coding: utf-8 -*-
import torch
import numpy as np
import matplotlib.pylab as plt
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

import cv2

method = 'uot'
nb = 10000

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

def img2tensor(img_raw):
    return torch.from_numpy(im2mat(img_raw.astype(np.float32) / 255))

def sample_pixel(img_tensor):
    idx = r.randint(img_tensor.shape[0], size=(nb,))
    return img_tensor[idx].to(0)

##############################################################################
# Generate data
# -------------
# img1 = 'data_pot/city_day.jpg'
vid1_path = 'oxford_scene_drone.mp4'
img2_path = 'data_pot/city_night.jpg'

time_s = time.time()
vid1_raw = cv2.VideoCapture(vid1_path)
img2_raw = plt.imread(img2)
print(f'loading elapsed={time.time() - time_s:.3f}')

time_s = time.time()
img2_tensor = img2tensor(img2_raw)
img2_sampled = sample_pixel(img2_tensor)

for _, img1_raw in vid1_raw:
    img1_tensor = img2tensor(img1_raw)
    img1_sampled = sample_pixel(img1_tensor)
    
    if method == 'ot':
        _, P = sampling_sinkhorn_divergence(img1_sampled, img2_sampled, eta=0.01, ret_plan=True)
    elif method == 'uot':
        P = sampling_sinkhorn_uot(img1_sampled, img2_sampled, eta=0.01, t1=10., t2=1., n_iter=100)

print(f'sinkhorn elapsed={time.time() - time_s:.3f}')

time_s = time.time()
# prediction between images (using out of sample prediction as in [6])
transp_Xs = transfer_with_map(img2_sampled, P, img1_tensor, img1_sampled, batch_size=100000)
print(f'transfer elapsed={time.time() - time_s:.3f}')

transp_Xs = transp_Xs.detach().cpu().numpy()
img1_transformed = minmax(mat2im(transp_Xs, img1_raw.shape))

# I1te = denormalize_rgb(I1te)
# I1te = np.array(equalize_adapthist(Image.fromarray(I1te, mode='RGB'))) / 255
# I1te = equalize_adapthist(I1te)

##############################################################################
# Plot new images
# ---------------

# plt.figure(figsize=(16, 8))

# plt.subplot(2, 3, 1)
# plt.imshow(I1)
# plt.axis('off')
# plt.title('Image 1')

# plt.subplot(2, 3, 2)
# plt.imshow(I2)
# plt.axis('off')
# plt.title('Image 2')

# plt.subplot(2, 3, 3)
# plt.imshow(I1te)
# plt.axis('off')
# plt.title('Image 1 Adapt (reg)')

# plt.tight_layout()

import os

def path2name(path_img):
    name = os.path.splitext(path_img)[0]
    return name.split('/')[1]

plt.savefig(f'color_transfer_{method}_source={path2name(img1)}_target={path2name(img2)}.png')
plt.savefig(f'color_transfer_{method}_source={path2name(img1)}_target={path2name(img2)}.png')

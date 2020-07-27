# -*- coding: utf-8 -*-
from codebase.OT import sampling_sinkhorn_divergence
from codebase.UOT import sampling_sinkhorn_uot
from color_transfer import color_transfer
from transfer import transfer_with_map

import os
import torch
import numpy as np
import matplotlib.pylab as plt
import time
import cv2
import colorsys
from PIL import Image
r = np.random.RandomState(1000) 

##############################################################################
# Generate data
# -------------
img1 = 'data_pot/imageA.jpg'
img2 = 'data_pot/imageB.jpg'

time_s = time.time()
# Loading images
img1_raw = cv2.imread(img1)
img2_raw = cv2.imread(img2)

print(type(img1_raw), img1_raw.dtype)

method = 'uot'
img1_transformed, time_total, time_mv2gpu = color_transfer(img1_raw, img2_raw, method=method, nb=10000)

print(f'time total = {time_total:.3f}, time moving data to gpu = {time_mv2gpu:.3f}')

##############################################################################
# Plot new images
# ---------------

plt.figure(figsize=(16, 8))

plt.subplot(2, 3, 1)
plt.imshow(img1_raw)
plt.axis('off')
plt.title('Image 1')

plt.subplot(2, 3, 2)
plt.imshow(img2_raw)
plt.axis('off')
plt.title('Image 2')

plt.subplot(2, 3, 3)
plt.imshow(img1_transformed)
plt.axis('off')
plt.title('Image 1 Adapt (reg)')

plt.tight_layout()

def path2name(path_img):
    name = os.path.splitext(path_img)[0]
    return name.split('/')[1]

plt.savefig(f'color_transfer_{method}_source={path2name(img1)}_target={path2name(img2)}.png')

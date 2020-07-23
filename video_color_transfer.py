# -*- coding: utf-8 -*-
import torch
import numpy as np
import matplotlib.pylab as plt
from codebase.OT import sampling_sinkhorn_divergence
from codebase.UOT import sampling_sinkhorn_uot
from transfer import transfer_with_map
import time
from skimage.exposure import equalize_adapthist
from color_transfer import color_transfer
from codebase.torchutils import check_nan_inf

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
    return np.clip(img, 0, 1) * 255

def img2tensor(img_raw):
    return torch.from_numpy(im2mat(img_raw.astype(np.float32) / 255))

def sample_pixel(img_tensor):
    idx = r.randint(img_tensor.shape[0], size=(nb,))
    return img_tensor[idx].to(0)

##############################################################################
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
vid1_path = 'oxford_short.mp4'
img2_path = 'data_pot/city_night.jpg'

vid1_raw = cv2.VideoCapture(vid1_path)
img2_raw = plt.imread(img2_path)

_, img_test = vid1_raw.read()
height, width, _ = img_test.shape
print(height, width)
video = cv2.VideoWriter('output.mp4', fourcc, 25, (width, height))

img2_tensor = img2tensor(img2_raw)
img2_sampled = sample_pixel(img2_tensor)
t = 0
while True:
    time_frame = time.time()
    time_mv2gpu = 0
    t += 1
    ret, img1_raw = vid1_raw.read()
    if ret is False: break
        
    img1_tensor = img2tensor(img1_raw)
    
    time_s = time.time()
    img1_sampled = sample_pixel(img1_tensor)
    time_mv2gpu += time.time() - time_s
    # SinkhornTransport

    if method == 'ot':
        _, P = sampling_sinkhorn_divergence(img1_sampled, img2_sampled, eta=0.01, ret_plan=True)
    elif method == 'uot':
        P = sampling_sinkhorn_uot(img1_sampled, img2_sampled, eta=0.01, t1=100., t2=1., n_iter=100)
        
    try:
        check_nan_inf(P, stop=True)
    except Exception:
        print(f'skipped frame {t}')
        continue
        
    # prediction between images (using out of sample prediction as in [6])
    transp_Xs, time_mv2gpu_transfer = transfer_with_map(img2_sampled, P, img1_tensor, img1_sampled, batch_size=100000)
    time_mv2gpu += time_mv2gpu_transfer
    
    transp_Xs = transp_Xs.detach().cpu().numpy()
    img1_transformed = minmax(mat2im(transp_Xs, img1_raw.shape))
    
    video.write(np.uint8(img1_transformed))
    
    print(f'frame {t} - time total = {time.time() - time_frame:.3f} s, time spent moving data to gpu = {time_mv2gpu:.3f} s')

cv2.destroyAllWindows()
video.release()

# def path2name(path_img):
#     name = os.path.splitext(path_img)[0]
#     return name.split('/')[1]

# plt.savefig(f'color_transfer_{method}_source={path2name(img1)}_target={path2name(img2)}.png')
# plt.savefig(f'color_transfer_{method}_source={path2name(img1)}_target={path2name(img2)}.png')


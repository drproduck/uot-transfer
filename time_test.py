import cv2
import torch
import time
from color_transfer import img2tensor

# img = cv2.imread('data_pot/imageA.jpg')
# print(img)
# 
# img = img2tensor(img)
for i in range(100):
    time_s = time.time()
    img = torch.rand(400, 600, 3)
    img = img.cuda()
    print(time.time() - time_s)

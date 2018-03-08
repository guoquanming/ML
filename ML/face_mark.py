##############读取已识别出的特征点，并将其叠加到图像上####################
from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")





# 读取csv文件中的特征点
landmarks_frame = pd.read_csv('C:/Users/Administrator/Desktop/faces/face_landmarks.csv')

# n表示读取文件中第几行，也就是第几张图片
n = 0

# 读取第n行第0列作为图片名字
img_name = landmarks_frame.ix[n, 0]

# 读取第一列及之后的数据作为标记点
landmarks = landmarks_frame.ix[n, 1:].as_matrix().astype('float')
landmarks = landmarks.reshape(-1, 2)

print('Image name: {}'.format(img_name))
print('Landmarks shape: {}'.format(landmarks.shape))
print('First 4 Landmarks: {}'.format(landmarks[:4]))



# plt.ion()   # interactive mode


def show_landmarks(image, landmarks):
    """Show image with landmarks"""
    plt.imshow(image)
    # 把特征点叠加到图像上
    plt.scatter(landmarks[:, 0], landmarks[:, 1], s=10, marker='1', c='r')
    plt.pause(0.001)  # 暂定一会儿，使数据更新

# 初始化
plt.figure()
# 获取目录下的img_name文件，并把读取到的标记点加载到图片上
show_landmarks(io.imread(os.path.join('C:/Users/Administrator/Desktop/faces/', img_name)),
               landmarks)
# 显示结果
plt.show()
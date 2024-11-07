#!/usr/bin/python3
# _*_ coding: utf-8 _*_
# @Time    : 2023/7/2 17:56
# @Author  : Linxuan Jiang
# @File    : Inference_camera.py
# @IDE     : PyCharm
# @Email   : 1195860834@qq.com
# Copyright MIT

import os
import shutil
import time

# utf-8
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from Face_preprocess import Face_detect_crop
# from models.mobilenet.model_1 import Generator

from cv2 import getTickCount
import cv2
import sys


transformer = transforms.Compose([
	transforms.ToTensor(),
	# transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def _totensor(array):
	img = torch.from_numpy(array)
	img = img.permute(2, 0, 1) / 255.0
	return img.contiguous()


def postprocess(x):
	"""[0,1] to uint8."""
	x = np.clip(255 * x, 0, 255)
	x = np.cast[np.uint8](x)
	return x


def tile(X, rows, cols):
	"""Tile images for display."""
	tiling = np.zeros((rows * X.shape[1], cols * X.shape[2], X.shape[3]), dtype=X.dtype)
	for i in range(rows):
		for j in range(cols):
			idx = i * cols + j
			if idx < X.shape[0]:
				img = X[idx, ...]
				tiling[i * X.shape[1]:(i + 1) * X.shape[1],
				j * X.shape[2]:(j + 1) * X.shape[2], :] = img
	return tiling


def contrast_brightness_demo(image, c, b):  # 定义方法， c @ contrast  对比度 ; b @ brightness 亮度
	h, w, ch = image.shape
	blank = np.zeros([h, w, ch], image.dtype)  # 定义一张空白图像
	dst = cv2.addWeighted(image, c, blank, 1 - c, b)  # 设定权重
	return dst


"""
https://zhuanlan.zhihu.com/p/570822430  tensor Rt的优势
https://blog.csdn.net/ltochange/article/details/120432092 几种框架下的速度对比


opencv 自带的SR： https://blog.csdn.net/jiazhen/article/details/115274863
模型
https://blog.csdn.net/jiazhen/article/details/115274863


跳帧处理高速视频流:
https://developer.aliyun.com/article/1099123


OpenCV读取视频时会自动丢掉重复帧，导致读取到的帧数和视频里的实际总帧数不一致
"""


class Inference:
	def __init__(self):

		self.detect_model = None
		self.face_fusion = None
		# """function flowchat"""
		self.init_face_detect_model()


	def log(self, msg):
		msg = " {}:     {}".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), msg)
		print(msg)

	def init_face_detect_model(self):
		self.log(" 加载人脸检测模型 ")
		self.detect_model = Face_detect_crop(model_weight_path='Weight/preprocess/Scrfd/scrfd_2.5g_kps.onnx')

		# self.detect_model = Face_detect_crop(model_weight_path='Weight/preprocess/Scrfd/scrfd_500m_kps.onnx')
  
	def inference(self, img_path, crop_size=256):
		farme = cv2.imread(img_path)

		# 人脸检测
		faces = self.detect_model.get(farme,crop_size)
		return faces


if __name__ == "__main__":
    img_path = f"none_face.jpg"
    infer = Inference()
    res = infer.inference(img_path)
    if res is not None:
        print(res.shape)
    else:
        print("未检测到人脸")

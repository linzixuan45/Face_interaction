#!/usr/bin/python3
# _*_ coding: utf-8 _*_
# @Time    : 2023/7/7 19:50
# @Author  : Linxuan Jiang
# @File    : Detector_align.py
# @IDE     : PyCharm
# @Email   : 1195860834@qq.com
# Copyright MIT


import numpy as np
from .Scrfd_model import SCRFD
import cv2
from skimage import transform as trans


class Face_detect_crop:
	def __init__(self, model_weight_path='Weight/preprocess/Scrfd/scrfd_10g_kps.onnx', rank=0):
		assert model_weight_path != '', print('please input model_weight_path')

		detect_model = SCRFD(model_file=model_weight_path)
		detect_model.prepare(ctx_id=rank)  # 使用的设备名称
		self.detect_model = detect_model

	def get(self, img, crop_size, swap_rb=False, get_crop=True):
		bboxes, kpss = self.detect_model.detect(img,
												input_size=(640, 640),
												swap_rb=swap_rb,
												det_thresh=0.6,
												metric='default')  # 得到box，关键点
		if bboxes.shape[0] == 0:
			return None

		det_score = bboxes[..., 4]  # 因为是只要一个人脸，所以只选了置信度最高的bbox作为人脸交换的目标
		# select the face with the hightest detection score
		best_index = np.argmax(det_score)
		kps = None
		if kpss is not None:
			kps = kpss[best_index]
		M, min_index = self.estimate_norm(kps, crop_size)
		if get_crop:
			align_img = cv2.warpAffine(img, M, (crop_size, crop_size), borderValue=0.0)
			return align_img, M
		else:
			return M


	def estimate_norm(self, lmk, image_size):
		"""
		facial alignment, taken from https://github.com/deepinsight/insightface
		return： M , min_index     [仿射变换矩阵,   最相似的关键点类别]
		"""

		'''[112,112]  mean 5 kps'''
		src1 = np.array([[51.642, 50.115], [57.617, 49.990], [35.740, 69.007],
						 [51.157, 89.050], [57.025, 89.702]],
						dtype=np.float32)
		# <--left
		src2 = np.array([[45.031, 50.118], [65.568, 50.872], [39.677, 68.111],
						 [45.177, 86.190], [64.246, 86.758]],
						dtype=np.float32)

		# ---frontal
		src3 = np.array([[39.730, 51.138], [72.270, 51.138], [56.000, 68.493],
						 [42.463, 87.010], [69.537, 87.010]],
						dtype=np.float32)

		# -->right
		src4 = np.array([[46.845, 50.872], [67.382, 50.118], [72.737, 68.111],
						 [48.167, 86.758], [67.236, 86.190]],
						dtype=np.float32)

		# -->right profile
		src5 = np.array([[54.796, 49.990], [60.771, 50.115], [76.673, 69.007],
						 [55.388, 89.702], [61.257, 89.050]],
						dtype=np.float32)

		src_arcface = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
								[41.5493, 92.3655], [70.7299, 92.2041]],
							   dtype=np.float32)

		src_ffhq = np.array([[192.98138, 239.94708], [318.90277, 240.1936], [256.63416, 314.01935],
							 [201.26117, 371.41043], [313.08905, 371.15118]], dtype=np.float32)

		src = np.array([src1, src2, src3, src4, src5])
		assert lmk.shape == (5, 2)

		# 相似变换：等距变换+均匀尺度缩放，所谓等距变换就是平移+旋转变换。
		lmk_tran = np.insert(lmk, 2, values=np.ones(5), axis=1)
		min_M = []
		min_index = []
		min_error = float('inf')

		src = src * image_size / 112

		tform = trans.SimilarityTransform()
		for i in np.arange(src.shape[0]):
			tform.estimate(lmk, src[i])
			M = tform.params[0:2, :]
			results = np.dot(M, lmk_tran.T)
			results = results.T
			error = np.sum(np.sqrt(np.sum((results - src[i]) ** 2, axis=1)))
			#         print(error)
			if error < min_error:
				min_error = error
				min_M = M
				min_index = i

		return min_M, min_index

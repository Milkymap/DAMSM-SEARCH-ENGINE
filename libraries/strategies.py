import cv2  
import textwrap

import pickle 
import numpy as np 
import operator as op 
import itertools as it, functools as ft 

import torch as th
import torchvision as tv 

from os import path 
from glob import glob 
from torchvision import transforms as T 

def read_image(image_path, by='cv'):
	if by == 'cv':
		return cv2.imread(image_path, cv2.IMREAD_COLOR)
	if by == 'th':
		return tv.io.read_image(image_path)
	raise ValueError(by)

def th2cv(tensor_3d):
	red, green, blue = tensor_3d.numpy()
	return cv2.merge((blue, green, red))

def cv2th(bgr_image):
	blue, green, red = cv2.split(bgr_image)
	return th.from_numpy(np.stack([red, green, blue]))

def to_grid(batch_images, nb_rows=8, padding=10, normalize=True):
	grid_images = tv.utils.make_grid(batch_images, nrow=nb_rows, padding=padding, normalize=normalize)
	return grid_images

def denormalize(tensor_data, mean, std):
	mean = th.tensor(mean)
	std = th.tensor(std)
	return tensor_data * std[:, None, None] + mean[:, None, None]

def caption2image(caption, shape=(256, 256), text_width=20, font_face=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.7, font_thickness=2, text_color=(255, 255, 255), broder_color=(0, 255, 244), margin=2):
	H, W = shape 
	image = np.zeros(shape).astype('uint8')
	image = cv2.merge((image, image, image))

	lines = textwrap.wrap(caption, width=text_width)
	accumulator = []
	y_position = 0
	for line in lines:
		(t_w, t_h), t_b = cv2.getTextSize(line, font_face, font_scale, font_thickness)
		accumulator.append((t_w, t_h, t_b))
		y_position = y_position + t_h + t_b 

	y_position = (H - y_position) // 2
	for line, (t_w, t_h, t_b) in zip(lines, accumulator):
		cv2.putText(image, line, ((W - t_w) // 2, y_position + t_h + t_b), font_face, font_scale, text_color, font_thickness)
		y_position = y_position + t_h + t_b
		
	cv2.rectangle(image, (margin, margin), (W-margin, H-margin), broder_color, margin)
	return image 


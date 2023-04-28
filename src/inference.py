import torch
import numpy as np

import cv2
import matplotlib.pyplot as plt
from matplotlib import patches
from PIL import Image
from google.colab.patches import cv2_imshow


import argparse
import glob as glob
import os

from utils import get_classes, get_device, get_num_of_classes
from model import create_model




def inference():

	abspath = os.path.abspath(__file__)
	dname = os.path.dirname(abspath)
	dname_list = dname.split('/')
	root = "/".join(dname_list[:(len(dname_list) - 1)])
	os.chdir(root)

	# load the model and the trained weights
	model = create_model(num_classes=len(CLASSES)).to(DEVICE)
	model.load_state_dict(torch.load(
		args.model, map_location=DEVICE
	))
	model.eval()
        

	# get the image file name for saving output later on
	image_name = image_path.split('/')[-1].split('.')[0]
	image = cv2.imread(image_path)
	orig_image = image.copy()
	# BGR to RGB
	image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB).astype(np.float32)
	# make the pixel range between 0 and 1
	image /= 255.0
	# bring color channels to front
	image = np.transpose(image, (2, 0, 1)).astype(np.float)
	# convert to tensor
	image = torch.tensor(image, dtype=torch.float).cuda()
	# add batch dimension
	image = torch.unsqueeze(image, 0)

	with torch.inference_mode():
		outputs = model(image)

	# load all detection to CPU for further operations
	outputs = [{k: v.to('cpu') for k, v in t.items()} for t in outputs]

	# carry further only if there are detected boxes
	if len(outputs[0]['boxes']) != 0:
		boxes = outputs[0]['boxes'].data.numpy()
		scores = outputs[0]['scores'].data.numpy()
		# filter out boxes according to `detection_threshold`
		boxes = boxes[scores >= detection_threshold].astype(np.int32)
		draw_boxes = boxes.copy()
		# get all the predicited class names
		pred_classes = [CLASSES[i] for i in outputs[0]['labels'].cpu().numpy()]

		# draw the bounding boxes and write the class name on top of it
		for j, box in enumerate(draw_boxes):
			cv2.rectangle(orig_image,
						(int(box[0]), int(box[1])),
						(int(box[2]), int(box[3])),
						(0, 0, 255), 2)
			cv2.putText(orig_image, pred_classes[j], 
						(int(box[0]), int(box[1]-5)),
						cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 
						2, lineType=cv2.LINE_AA)
		cv2_imshow(orig_image)
		cv2.imwrite(f"{image_name}.jpg", orig_image,)
		
		
	print('TEST PREDICTIONS COMPLETE')


if __name__ == '__main__':
        
	# arg parser initailizing  
	parser = argparse.ArgumentParser()
	parser.add_argument("--model", type=str, required= True, help="Inference model path")
	parser.add_argument("--threshold", type=float, required= False, help="Detection threshold", default=0.8)
	parser.add_argument("--image", type=str, required= False, help="Test Image Path", default=200)
	args = parser.parse_args()

	DEVICE = get_device()
	CLASSES = get_classes()

	inference()
	
    
import cPickle
import numpy as np
import cv2
import os
import sys

import xml.etree.ElementTree as ET

BLUE = (255,0,0)

sys.path.append('.')
sys.path.append('./lib')

from lib.dataset.labelme import labelme

imdb_root = "/home/vincent/LabelMe"
imdb_img_dir = os.path.join(imdb_root, "Images")
imdb_annot_dir = os.path.join(imdb_root, "Annotations")

image_set = "singulation_test_resized"
cache_path = "./data"
mask_size = 21
binary_thresh = 0.4


cls = 'envelope'

classes = ["box","envelope"]
imdb = labelme(image_set, imdb_root, cache_path, mask_size=mask_size, binary_thresh=binary_thresh, classes=classes)

results_folder = "./data/cache/results/labelme_singulation_test_resized"
det_file = os.path.join(results_folder, "%s_det.pkl"%(cls))
seg_file = os.path.join(results_folder, "%s_seg.pkl"%(cls))

# Get predict pickle file for this class
with open(det_file, 'rb') as f:
    print("Loading det predictions %s..."%(det_file))
    boxes_pkl = cPickle.load(f)
with open(seg_file, 'rb') as f:
    print("Loading seg predictions %s..."%(seg_file))
    masks_pkl = cPickle.load(f)

# box shape [-1, -1, 5]  image index, number of detections, [x1 y1 x2 y2 score]
# mask shape [-1, -1, 21, 21]  image index, number of detections, mask shape (21 21)

binary_thresh = 0.4

SAMPLE_IDX = 3

sample_box = boxes_pkl[SAMPLE_IDX][0]
sample_mask = masks_pkl[SAMPLE_IDX][0]
pred_box = np.round(sample_box[:4]).astype(int)
pred_mask = sample_mask
mask_w = pred_box[2] - pred_box[0] + 1
mask_h = pred_box[3] - pred_box[1] + 1

pred_mask = cv2.resize(pred_mask.astype(np.float32), (mask_w, mask_h))
pred_mask = pred_mask >= binary_thresh

m = pred_mask.astype(np.uint8)
m[m==1] *= 255

print(imdb.image_index[SAMPLE_IDX])

im_path = os.path.join(imdb_img_dir, image_set, imdb.image_index[SAMPLE_IDX])
img = cv2.imread(im_path)

cv2.rectangle(img, (pred_box[0],pred_box[1]),(pred_box[2],pred_box[3]), BLUE)
cv2.imshow('bbox', img)
cv2.imshow('mask', m)
cv2.waitKey(0)

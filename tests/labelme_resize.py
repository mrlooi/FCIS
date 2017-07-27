import xml.etree.ElementTree as ET
import numpy as np
import random
import cv2

# def get_data(input_file):
# 	et = ET.parse(input_file)

def view_seg_polygons(img, poly_pts,line_thickness=3):#, fill=False):
	# from matplotlib.collections import PatchCollection
	# from matplotlib.patches import Polygon

	output = img.copy()
	# overlay = img.copy()

	if type(poly_pts) != list:
		poly_pts = [poly_pts]

	polygons = []
	for p_ in poly_pts:
		# convert seg list to 2d array (each item being x,y)
		pts = p_.reshape((-1,1,2))
		color = tuple(random.randint(0,255) for i in range(3))

		pts = pts.astype(int)
		cv2.polylines(output,[pts],True,color,line_thickness)

		polygons.append(pts)
		# if fill:

		# 	cv2.fillConvexPoly(overlay,pts,color)
		# 	alpha = 0.3
		# 	cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)

	cv2.imshow('polygons', output)
	cv2.waitKey(0)

	return polygons

def get_seg_pt_mask(poly_pts, height, width):

	if type(poly_pts) != list:
		poly_pts = [poly_pts]

	masks = []

	for p_ in poly_pts:
		m_ = np.zeros((height,width, 1), dtype=np.float32)
		cv2.fillPoly(m_, [p_.reshape((-1,1,2))], (1))
		m_ = m_.squeeze()
		masks.append(m_)

	return masks

def view_seg_pt_mask(masks, height, width):
	mask_view = np.zeros((height, width, 1), dtype=np.uint8)
	for m_ in masks:
		mask_view[m_==1] = 255

	cv2.imshow('mask', mask_view)
	cv2.waitKey(0)

	return mask_view

def get_seg_pt_bbox(poly_pts):
	if type(poly_pts) != list:
		poly_pts = [poly_pts]
	
	bboxes = []

	for p_ in poly_pts:
		p_max = np.amax(p_, axis=0)
		p_min = np.amin(p_, axis=0)
		bboxes.append([tuple(p_min), tuple(p_max)])

	return bboxes

def view_seg_pt_bbox(img, bboxes):
	img_copy = img.copy()
	for b_ in bboxes:
		cv2.rectangle(img_copy, b_[0], b_[1], (0,0,255))

	cv2.imshow('bboxes', img_copy)
	cv2.waitKey(0)

def get_crop_interpolated_pts(poly_pts, h, w, crop_top = 0, crop_bottom = 0, crop_left = 0, crop_right = 0):
	total_y_crop = crop_top + crop_bottom
	total_x_crop = crop_left + crop_right
	assert(0 <= total_y_crop < h)
	assert(0 <= total_x_crop < w)

	max_y = h - crop_top - crop_bottom
	max_x = w - crop_left - crop_right

	if type(poly_pts) != list:
		poly_pts = [poly_pts]
	
	inter_pts = []

	for p_ in poly_pts:
		p_ = np.array(p_) if type(p_) != np.ndarray else p_.copy()
		p_[:,1] -= crop_top
		p_[:,0] -= crop_left

		p_[:,1]=np.clip(p_[:,1],0,max_y) 
		p_[:,0]=np.clip(p_[:,0],0,max_x)

		# if all points on an axis are the same, indicates a line -> no longer a polygon
		if (p_[:,0] == p_[0,0]).all() or (p_[:,1] == p_[0,1]).all():
			continue	
		inter_pts.append(p_)

	return inter_pts

def get_resized_interpolated_pts(poly_pts, h, w, th, tw):
	assert(h > 0 and w > 0 and th > 0 and tw > 0)

	w_tw_ratio = tw/float(w)
	h_th_ratio = th/float(h)

	if type(poly_pts) != list:
		poly_pts = [poly_pts]

	resize_inter_pts = []
	for p_ in poly_pts:
		dtype_ = p_.dtype
		p_ = p_.astype(np.float64)
		p_[:,0] *= w_tw_ratio
		p_[:,1] *= h_th_ratio
		p_ = np.around(p_).astype(dtype_)  # round instead of floor for more accuracy
		resize_inter_pts.append(p_)

	return resize_inter_pts

def updateETTree(ET_obj, output_file, poly_cls, poly_pts, h, w):
	element = ET_obj.getroot()

	# reset the height and width
	# element_img_sz = element.find('imagesize')
	element.find('imagesize').find('nrows').text = str(h)
	element.find('imagesize').find('ncols').text = str(w)

	# delete old the polygon points
	for i, e in enumerate(element.findall('object')):
		if i >= len(poly_pts):
			element.remove(e)
	# total_extra_e_objs = len(element_objects) - len(poly_pts)

	element_objects = element.findall('object')
	for i, e in enumerate(element_objects):  # clear old polygon pts
		e.find('deleted').text = '0'
		e.find('id').text = str(i)
		e_poly = e.find('polygon')
		for p in e_poly.findall('pt'):
			e_poly.remove(p)

	# add the new polygon points
	if type(poly_pts) != list:
		poly_pts = [poly_pts]
	for i, p_ in enumerate(poly_pts):
		e = element_objects[i]
		e_poly = e.find('polygon')
		e.find('name').text = poly_cls[i]
		for pt_ in p_:
			elem_pt_x = ET.Element("x")
			elem_pt_y = ET.Element("y")
			elem_pt_x.text = str(pt_[0])
			elem_pt_y.text = str(pt_[1])
			elem_pt = ET.Element("pt")
			elem_pt.append(elem_pt_x)
			elem_pt.append(elem_pt_y)
			e_poly.append(elem_pt)

	ET_obj.write(output_file)
	print("Wrote to %s. Total final element objects %d"%(output_file, len(element_objects)))

if __name__ == '__main__':
	import glob
	import os
	import os.path as osp 

	labelme_root = "/home/vincent/LabelMe"
	image_set = 'singulation_test'
	image_set_new = 'singulation_test_resized'

	
	annot_dir = osp.join(labelme_root, "Annotations", image_set)
	img_dir = osp.join(labelme_root, "Images", image_set)
	annot_dir_new = osp.join(labelme_root, "Annotations", image_set_new)
	img_dir_new = osp.join(labelme_root, "Images", image_set_new)


	TARGET_HEIGHT = 720
	TARGET_WIDTH = 1280

	assert osp.exists(img_dir) and osp.exists(annot_dir), "%s and/or %s does not exist!"%(img_dir, annot_dir)
	if not osp.exists(img_dir_new):
		os.makedirs(img_dir_new)
	if not osp.exists(annot_dir_new):
		os.makedirs(annot_dir_new)

	img_paths = os.listdir(img_dir)
	img_names = [f[:f.rfind('.')] for f in img_paths]
	for ix, im_path in enumerate(img_paths):

	 #    # load image
		im_path_full = osp.join(img_dir, im_path)
		img = cv2.imread(im_path_full)

		h,w,_ = img.shape

	 #    # annot file
		annot_file = osp.join(annot_dir, img_names[ix] + ".xml")
		# print(im_path_full, annot_file)

		et = ET.parse(annot_file)
		element = et.getroot()
		# element_file = element.find('filename')
		# element_img_sz = element.find('imagesize')
		
		element_objects = [e for e in element.findall('object') if int(e.find("deleted").text) != 1]

		e_pts_all = []
		e_cls = []
		for e in element_objects:
			# if int(e.find('deleted').text) == 1:
			# 	continue
			# print(e.find('name').text)
			e_poly = e.find('polygon')
			cls = e.find('name').text
			e_cls.append(cls)

			e_pts = [( float(p.find('x').text), float(p.find('y').text) ) for p in e_poly.findall('pt')]
			e_pts = np.array(e_pts).astype(np.int32)
			e_pts_all.append(e_pts)

		# resize
		img_resized = cv2.resize(img, (TARGET_WIDTH,TARGET_HEIGHT))
		resized_pts = get_resized_interpolated_pts(e_pts_all, h, w, TARGET_HEIGHT, TARGET_WIDTH)

		# save img
		im_file_new = osp.join(img_dir_new, im_path)
		cv2.imwrite(im_file_new, img_resized)

		# save annot
		annot_file_new = osp.join(annot_dir_new, img_names[ix] + ".xml")
		updateETTree(et, annot_file_new, e_cls, resized_pts, TARGET_HEIGHT, TARGET_WIDTH)

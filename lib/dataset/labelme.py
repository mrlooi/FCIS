# --------------------------------------------------------
# Fully Convolutional Instance-aware Semantic Segmentation
# Copyright (c) 2017 Microsoft
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Modified by Haozhi Qi, Guodong Zhang
# --------------------------------------------------------

"""
Pascal VOC database
This class loads ground truth notations from standard Pascal VOC XML data formats
and transform them into IMDB format. Selective search is used for proposals, see roidb
function. Results are written as the Pascal VOC format. Evaluation is based on mAP
criterion.
"""

import cPickle
import cv2
import os
import scipy.io as sio
import numpy as np
import glob
import hickle as hkl
import xml.etree.ElementTree as ET

from imdb import IMDB
# from pascal_voc_eval import voc_eval, voc_eval_sds
from ds_utils import unique_boxes, filter_small_boxes
from labelme_eval import labelme_eval, labelme_eval_sds

class labelme(IMDB):
    def __init__(self, image_set, data_path, cache_path, mask_size=21, binary_thresh=0.4, classes=[]):
        super(labelme, self).__init__('labelme', image_set, cache_path, data_path)  # set self.name

        self.data_images_path = os.path.join(data_path, "Images", image_set)
        self.data_annotation_path = os.path.join(data_path, "Annotations", image_set)

        assert os.path.exists(self.data_path), \
                'Path does not exist: {}'.format(self.data_path)
        assert os.path.exists(self.data_images_path), \
                'Path does not exist: {}'.format(self.data_images_path)
        assert os.path.exists(self.data_annotation_path), \
                'Path does not exist: {}'.format(self.data_annotation_path)
        self.image_set_index = self.load_image_set_index()
        self.image_index = self.image_set_index

        self.num_images = len(self.image_set_index)
        print('num_images: %d'%(self.num_images))

        # classes and indices
        bg_cls = '__background__'
        if type(classes) == list and len(classes) > 0:
            self.classes = [bg_cls] + sorted([c.lower() for c in classes if c.lower() != bg_cls])
        else:
            self.classes = [bg_cls] + sorted(self.get_classes())  # background always index 0
        self.num_classes = len(self.classes)
        self.class_to_ind = dict(zip(self.classes, xrange(self.num_classes)))
        print("classes: %s"%(self.classes))

        self.mask_size = mask_size
        self.binary_thresh = binary_thresh

        # DB specific config options
        self.config = {}

    def get_classes(self):
        classes_ = {}
        for img_name in self.image_index:
            filename = self.annotation_path_from_index(img_name)
            tree = ET.parse(filename)
            element = tree.getroot()
            objs = [e for e in element.findall('object') if int(e.find("deleted").text) != 1]

            for ix, e in enumerate(objs):
                cls = e.find('name').text.lower().strip()
                classes_[cls] = 1

        return classes_.keys()

    def annotation_path_at(self, i):
        """
        Return the absolute path to annotation i in the image sequence.
        """
        return self.annotation_path_from_index(self.image_index[i])

    def annotation_path_from_index(self, img_name):
        """
        Construct an annotation path from the image's "index" identifier.
        """
        base_name = img_name[:img_name.rfind('.')]
        annot_path = os.path.join(self.data_annotation_path, base_name + ".xml")
        assert os.path.exists(annot_path), \
                'Path does not exist: {}'.format(annot_path)
        return annot_path

    def load_image_set_index(self):
        """
        Load the indexes listed in this dataset's image set file.
        """
        img_set_path = self.data_images_path
        assert os.path.exists(img_set_path), \
                'Path does not exist: {}'.format(img_set_path)
        image_set = [f.split("/")[-1] for f in glob.glob(img_set_path + "/*")]# if f.split(".")[-1] in VALID_IMG_EXT]
        return image_set

    def image_path_from_index(self, img_name):
        """
        Construct an image path from the image's "index" identifier.
        """
        image_path = os.path.join(self.data_images_path, img_name)
        assert os.path.exists(image_path), \
                'Path does not exist: {}'.format(image_path)
        return image_path

    def gt_roidb(self, use_cache=True):
        """
        Return the database of ground-truth regions of interest.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        if use_cache:
            if os.path.exists(cache_file):
                print("[INFO] Loading cached gt roidb %s..."%(cache_file))
                with open(cache_file, 'rb') as fid:
                    roidb = cPickle.load(fid)
                print('[INFO] Loaded {} cache gt roidb loaded from {}'.format(self.name, cache_file))
                return roidb
            else:
                print("[WARN] %s cache file does not exist. Will reload from annotations"%(cache_file))

        gt_roidb = [self.load_annotation(index)
                    for index in self.image_set_index]
        with open(cache_file, 'wb') as fid:
            cPickle.dump(gt_roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote gt roidb to {}'.format(cache_file)

        return gt_roidb

    def gt_sdsdb(self, use_cache=True):
        return self.gt_roidb(use_cache=use_cache)

    def load_annotation(self, index):
        """
        for a given index, load image and bounding boxes info from XML file
        :param index: index of a specific image
        :return: record['boxes', 'gt_classes', 'gt_overlaps', 'flipped']
        """
        roi_rec = dict()
        roi_rec['image'] = self.image_path_from_index(index)

        filename = self.annotation_path_from_index(index)
        tree = ET.parse(filename)
        element = tree.getroot()

        img_size = element.find('imagesize')
        height = int(img_size.find('nrows').text)
        width = int(img_size.find('ncols').text)
        roi_rec['height'] = float(height)
        roi_rec['width'] = float(width)

        objs = [e for e in element.findall('object') if int(e.find("deleted").text) != 1 and e.find('name').text.lower().strip() in self.classes]
        # if not self.config['use_diff']:
        #     non_diff_objs = [obj for obj in objs if int(obj.find('difficult').text) == 0]
        #     objs = non_diff_objs
        num_objs = len(objs)

        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
        gt_masks = np.zeros((num_objs, height, width))

        # Load object bounding boxes into a data frame.
        for ix, e in enumerate(objs):
            cls_ = e.find('name').text.lower().strip()
            # if cls_ not in self.classes:
            #     continue
            cls_idx = self.class_to_ind[cls_]

            e_poly = e.find('polygon')
            e_pts = [( float(p.find('x').text), float(p.find('y').text) ) for p in e_poly.findall('pt')]
            e_pts = np.array(e_pts).astype(np.int32)

            # mask
            cur_gt_mask = np.zeros((height,width, 1), dtype=np.float32)
            cv2.fillPoly(cur_gt_mask, [e_pts.reshape((-1,1,2))], (1))
            cur_gt_mask = cur_gt_mask.squeeze().astype(np.bool)

            # bbox
            p_max = np.amax(e_pts, axis=0)
            p_min = np.amin(e_pts, axis=0)

            x1 = p_min[0]
            y1 = p_min[1]
            x2 = p_max[0]
            y2 = p_max[1]

            boxes[ix, :] = [x1, y1, x2, y2]
            gt_classes[ix] = cls_idx
            overlaps[ix, cls_idx] = 1.0
            gt_masks[ix, :, :] = cur_gt_mask

        roi_rec.update({'boxes': boxes,
                        'gt_classes': gt_classes,
                        'gt_overlaps': overlaps,
                        'max_classes': overlaps.argmax(axis=1),
                        'max_overlaps': overlaps.max(axis=1),
                        'cache_seg_inst': self.save_mask_path_from_index(index, gt_masks, verbose=True),
                        'flipped': False
                        })
        return roi_rec


    def save_mask_path_from_index(self, index, gt_mask, verbose=True):
        """
        given image index, cache high resolution mask and return full path of masks
        :param index: index of a specific image
        :return: full path of this mask
        """
        # if self.image_set == 'val':
        #     return []
        base_name = index[:index.rfind('.')]
        cache_folder = os.path.join(self.cache_path, self.name, 'Mask')
        if not os.path.exists(cache_folder):
            os.makedirs(cache_folder)
        # instance level segmentation
        gt_mask_file = os.path.join(cache_folder, base_name + '.hkl')
        if not os.path.exists(gt_mask_file):
            hkl.dump(gt_mask.astype('bool'), gt_mask_file, mode='w', compression='gzip')
        # cache flip gt_masks
        gt_mask_flip_file = os.path.join(cache_folder, base_name + '_flip.hkl')
        if not os.path.exists(gt_mask_flip_file):
            hkl.dump(gt_mask[:, :, ::-1].astype('bool'), gt_mask_flip_file, mode='w', compression='gzip')
        if verbose:
            print("Saved mask files in %s: %s, %s"%(cache_folder, gt_mask_file, gt_mask_flip_file))
        return gt_mask_file

    def get_result_dir(self):
        return os.path.join(self.result_path, 'results')

    def get_result_file_template(self):#, output_dir = None):
        """
        """
        # result_dir = output_dir if output_dir is not None else os.path.join(self.result_path, 'results')
        result_dir = self.get_result_dir()
        filedir = os.path.join(result_dir, self.image_set)
        if not os.path.exists(filedir):
            os.makedirs(filedir)
        filename = 'det_' + self.image_set + '_{:s}.txt'
        path = os.path.join(filedir, filename)
        return path

    '''DETECTION RESULTS ONLY'''
    def evaluate_detections(self, detections):
        """
        top level evaluations
        :param detections: result matrix, [bbox, confidence]
        :return: None
        """
        self._write_labelme_detection_results(detections)
        info = self.do_python_detection_eval()
        return info

    '''DETECTION RESULTS ONLY'''
    def _write_labelme_detection_results(self, all_boxes):
        for cls_ind, cls in enumerate(self.classes):
            if cls == '__background__':
                continue
            print 'Writing {} Labelme results file'.format(cls)
            filename = self.get_result_file_template().format(cls)
            with open(filename, 'wt') as f:
                for im_ind, index in enumerate(self.image_index):
                    dets = all_boxes[cls_ind][im_ind]
                    if dets == []:
                        continue
                    # the VOCdevkit expects 1-based indices
                    for k in xrange(dets.shape[0]):
                        f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                            format(index, dets[k, -1],              # filename(stem), score
                                   dets[k, 0] + 1, dets[k, 1] + 1,  # x1, y1, x2, y2
                                   dets[k, 2] + 1, dets[k, 3] + 1))

    '''DETECTION RESULTS ONLY'''
    def do_python_detection_eval(self):
        """
        python evaluation wrapper
        :return: info_str
        """
        info_str = ''
        annopath = os.path.join(self.data_annotation_path, '{:s}.xml')
        cachedir = os.path.join(self.cache_path, self.name)
        # The PASCAL VOC metric changed in 2010
        # use_07_metric = True if self.year == 'SDS' or int(self.year) < 2010 else False
        use_07_metric = False
        print 'VOC07 metric? ' + ('Y' if use_07_metric else 'No')
        info_str += 'VOC07 metric? ' + ('Y' if use_07_metric else 'No')
        info_str += '\n'

        # AP@0.5
        ovthresh = 0.5
        aps = []
        for cls_ind, cls in enumerate(self.classes):
            if cls == '__background__':
                continue
            filename = self.get_result_file_template().format(cls)
            rec, prec, ap = labelme_eval(
                                filename, annopath, self.image_index, cls, cachedir, ovthresh=ovthresh,
                                use_07_metric=use_07_metric)
            aps += [ap]
            print('AP for {} = {:.4f}'.format(cls, ap))
            info_str += 'AP for {} = {:.4f}\n'.format(cls, ap)
        print('Mean AP@{:.1f} = {:.4f}'.format(ovthresh, np.mean(aps)))
        info_str += 'Mean AP@{:.1f} = {:.4f}\n\n'.format(ovthresh, np.mean(aps))

        # AP@0.7
        ovthresh = 0.7
        aps = []
        for cls_ind, cls in enumerate(self.classes):
            if cls == '__background__':
                continue
            filename = self.get_result_file_template().format(cls)
            rec, prec, ap = labelme_eval(
                                filename, annopath, self.image_index, cls, cachedir, ovthresh=ovthresh,
                                use_07_metric=use_07_metric)
            aps += [ap]
            print('AP for {} = {:.4f}'.format(cls, ap))
            info_str += 'AP for {} = {:.4f}\n'.format(cls, ap)
        print('Mean AP@{:.1f} = {:.4f}'.format(ovthresh, np.mean(aps)))
        info_str += 'Mean AP@{:.1f} = {:.4f}\n\n'.format(ovthresh, np.mean(aps))

        return info_str


    '''SEGMENTATION RESULTS'''
    def evaluate_sds(self, all_boxes, all_masks):
        result_dir = os.path.join(self.get_result_dir(), self.name)
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

        det_results_file_format = os.path.join(result_dir, "%s_det.pkl") # %s is cls
        seg_results_file_format = os.path.join(result_dir, "%s_seg.pkl") # %s is cls

        self._write_seg_results_file(all_boxes, all_masks, det_results_file_format, seg_results_file_format)
        info = self._py_evaluate_segmentation(det_results_file_format, seg_results_file_format)
        return info

    '''SEGMENTATION RESULTS'''
    def _write_seg_results_file(self, all_boxes, all_masks, det_results_file_format, seg_results_file_format):
        """
        Write results as a pkl file, note this is different from
        detection task since it's difficult to write masks to txt
        """
        # Always reformat result in case of sometimes masks are not
        # binary or is in shape (n, sz*sz) instead of (n, sz, sz)

        all_boxes, all_masks = self._reformat_result(all_boxes, all_masks)
        for cls_inds, cls in enumerate(self.classes):
            if cls == '__background__':
                continue
            det_results_file = det_results_file_format%(cls)
            seg_results_file = seg_results_file_format%(cls)
            print('Writing %s results file: %s, %s'%(cls, det_results_file, seg_results_file))
            # print filename
            with open(det_results_file, 'wb') as f:
                cPickle.dump(all_boxes[cls_inds], f, cPickle.HIGHEST_PROTOCOL)
            with open(seg_results_file, 'wb') as f:
                cPickle.dump(all_masks[cls_inds], f, cPickle.HIGHEST_PROTOCOL)

    '''SEGMENTATION RESULTS'''
    def _reformat_result(self, boxes, masks):
        num_images = self.num_images
        num_class = len(self.classes)
        reformat_masks = [[[] for _ in xrange(num_images)]
                          for _ in xrange(num_class)]
        for cls_inds in xrange(1, num_class):  # ignore bg class
            for img_inds in xrange(num_images):
                if len(masks[cls_inds][img_inds]) == 0:
                    continue
                num_inst = masks[cls_inds][img_inds].shape[0]
                reformat_masks[cls_inds][img_inds] = masks[cls_inds][img_inds]\
                    .reshape(num_inst, self.mask_size, self.mask_size)
                # reformat_masks[cls_inds][img_inds] = reformat_masks[cls_inds][img_inds] >= 0.4
        all_masks = reformat_masks
        return boxes, all_masks

    '''SEGMENTATION RESULTS'''
    def _py_evaluate_segmentation(self, det_results_file_format, seg_results_file_format):
        info_str = ''

        annot_dir = self.data_annotation_path
        cache_dir = os.path.join(self.cache_path, self.name)
        output_dir = self.get_result_dir()
        # The PASCAL VOC metric changed in 2010
        # use_07_metric = True if self.year == 'SDS' or int(self.year) < 2010 else False

        # define this as true according to SDS's evaluation protocol
        use_07_metric = True
        print 'VOC07 metric? ' + ('Y' if use_07_metric else 'No')
        info_str += 'VOC07 metric? ' + ('Y' if use_07_metric else 'No')
        info_str += '\n'

        # AP@0.5
        ovthresh = 0.5
        aps = []
        print ('~~~~~~ Evaluation use min overlap = %.1f ~~~~~~'%(ovthresh))
        info_str += '~~~~~~ Evaluation use min overlap = %.1f ~~~~~~'%(ovthresh)
        info_str += '\n'

        # det, seg filenames

        for i, cls in enumerate(self.classes):
            if cls == '__background__':
                continue
            det_results_file = det_results_file_format%(cls) 
            seg_results_file = seg_results_file_format%(cls)
            ap = labelme_eval_sds(det_results_file, seg_results_file, annot_dir,
                                  self.image_index, cls, cache_dir, self.classes, self.mask_size, self.binary_thresh, ov_thresh=ovthresh)
            aps += [ap]
            print('AP for {} = {:.2f}'.format(cls, ap*100))
            info_str += 'AP for {} = {:.2f}\n'.format(cls, ap*100)
        print('Mean AP@{:.1f} = {:.2f}'.format(ovthresh, np.mean(aps)*100))
        info_str += 'Mean AP@{:.1f} = {:.2f}\n'.format(ovthresh, np.mean(aps)*100)

        # AP@0.7
        ovthresh = 0.7
        aps = []
        print ('~~~~~~ Evaluation use min overlap = %.1f ~~~~~~'%(ovthresh))
        info_str += '~~~~~~ Evaluation use min overlap = %.1f ~~~~~~'%(ovthresh)
        info_str += '\n'
        for i, cls in enumerate(self.classes):
            if cls == '__background__':
                continue
            det_results_file = det_results_file_format%(cls) 
            seg_results_file = seg_results_file_format%(cls)
            ap = labelme_eval_sds(det_results_file, seg_results_file, annot_dir,
                                  self.image_index, cls, cache_dir, self.classes, self.mask_size, self.binary_thresh, ov_thresh=ovthresh)
            aps += [ap]
            print('AP for {} = {:.2f}'.format(cls, ap*100))
            info_str += 'AP for {} = {:.2f}\n'.format(cls, ap*100)
        print('Mean AP@{:.1f} = {:.2f}'.format(ovthresh, np.mean(aps)*100))
        info_str += 'Mean AP@{:.1f} = {:.2f}\n'.format(ovthresh, np.mean(aps)*100)

        print("\n********SUMMARY********\n")
        print(info_str)
        return info_str

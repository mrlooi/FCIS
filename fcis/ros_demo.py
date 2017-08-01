# --------------------------------------------------------
# Fully Convolutional Instance-aware Semantic Segmentation
# Copyright (c) 2017 Microsoft
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Haochen Zhang, Yi Li, Haozhi Qi
# --------------------------------------------------------

import _init_paths

import argparse
import os
import os.path as osp
import glob
import sys
import logging
import pprint
import random
import cv2
import numpy as np
# get config
os.environ['PYTHONUNBUFFERED'] = '1'
os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
os.environ['MXNET_ENABLE_GPU_P2P'] = '0'

from config.config import config, update_config
cur_path = os.path.abspath(os.path.dirname(__file__))

sys.path.append(".")
# sys.path.insert(0, os.path.join(cur_path, '../external/mxnet', config.MXNET_VERSION))

import mxnet as mx
print("using mxnet at %s"%(mx.__file__))
from core.tester import im_detect, Predictor
from symbols import *
from utils.load_model import load_param, load_param_file
from utils.show_masks import show_masks
from utils.tictoc import tic, toc
from nms.nms import py_nms_wrapper
from mask.mask_transform import gpu_mask_voting, cpu_mask_voting

from utils.image import resize, transform
from demo import DataBatchWrapper, inference

# ros stuff
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

# custom ros stuff
from vision.srv import _Detection2
from vision.msg import DetectionData

RED = (0,0,255)

class RosFCISPredictor(object):
    def __init__(self, predictor, data_batch_wrapper, classes, min_score=0.85, publish=False, ctx_id=[0]):
        if type(ctx_id) != list:
            ctx_id = [ctx_id]
        self.ctx_id = ctx_id
        self.min_score = min_score
        self.publish = publish

        self.bridge = CvBridge()

        # class
        self.classes = classes
        self.classes_color = [(random.randint(0,255),random.randint(0,255),random.randint(0,255)) for c in self.classes]
        
        self.image_sub = rospy.Subscriber("/kinect2/hd/image_color",Image, self.callback)

        ros_node_extension = '_'.join([c for c in classes if c.lower() != "__background__"])
        if self.publish:
            self.image_pub = rospy.Publisher("/kinect2/hd/image_mask_%s"%(ros_node_extension),Image,queue_size=5)

        # service 
        self.image_service = rospy.Service('/kinect2/hd/mask_service_%s'%(ros_node_extension), _Detection2.Detection2, self.callback_service)
        self.current_cv_image = None

        # model
        self.predictor = predictor

        # data batch wrapper
        self.data_batch_wr = data_batch_wrapper

        # data 
        self.current_detection_items = []

        print("RosFCISPredictor is running. Target classes: %s. Using min score of %.3f..."%(self.classes, self.min_score))

    def callback(self, data):
        try:
            im = self.bridge.imgmsg_to_cv2(data, "bgr8")
            self.current_cv_image = im
        except CvBridgeError as e:
            print(e)


        if self.publish:
            data = self.inference(self.current_cv_image)
            self.current_detection_items = self._format_detection_response(data)

            # max_scores = 
            try:
                img_copy = self.current_cv_image.copy()
                for cls,cls_data in data.items():
                    cls_color = self.classes_color[self.classes.index(cls)]
                    for d in cls_data:
                        bbox = d['bbox']
                        if len(bbox) != 4:
                            continue
                        cnt = d['contours']
                        score = d['score']
                        # self.current_max_bbox = bbox
                        bbox_top_pt = (int(bbox[0]),int(bbox[1]))
                        cv2.rectangle(img_copy, bbox_top_pt, (int(bbox[2]),int(bbox[3])), cls_color, 3)
                        cv2.putText(img_copy, "Score: %.3f, %s"%(score, cls), bbox_top_pt, cv2.FONT_HERSHEY_SIMPLEX, 0.8, RED, 2)
                        cv2.drawContours(img_copy,[cnt],0,cls_color,2)
                self.image_pub.publish(self.bridge.cv2_to_imgmsg(img_copy, "bgr8"))
                cv2.imshow("bounding boxes", img_copy)
            except CvBridgeError as e:
                print("[ERROR] CvBridgeError: %s"%e)

        cv2.waitKey(1)
        
    def callback_service(self, req):
        detection_items = []
        if self.current_cv_image is None:
            print("Current image is empty")
        else:
            if self.publish:
                detection_items = self.current_detection_items
            else:
                data = self.inference(self.current_cv_image)

                detection_items = self._format_detection_response(data)

        return _Detection2.Detection2Response(detection_items)

    def inference(self, im):
        data_batch = self.data_batch_wr.get_data_batch(im)

        tic()
        dets, masks = inference(self.predictor, data_batch, self.data_batch_wr.data_names, len(self.classes), BINARY_THRESH=config.BINARY_THRESH, CONF_THRESH=self.min_score, gpu_id=self.ctx_id[0])
        print('inference time: {:.4f}s'.format(toc()))

        data = self.reformat_data(dets, masks)
        return data

    def reformat_data(self, dets, masks):
        data = {}

        for cls_ix, cls in enumerate([c for c in self.classes if c.lower() != "__background__"]): # ignore bg class
            cls_dets = dets[cls_ix]
            cls_masks = masks[cls_ix]
            data[cls] = []
            for ix, bbox in enumerate(cls_dets):
                pred_score = bbox[-1]
                pred_box = np.round(bbox[:4]).astype(np.int32)
                pred_mask = cls_masks[ix]

                mask_w = pred_box[2] - pred_box[0] + 1
                mask_h = pred_box[3] - pred_box[1] + 1

                # reshape mask 
                pred_mask = cv2.resize(pred_mask.astype(np.float32), (mask_w, mask_h))
                pred_mask = pred_mask >= config.BINARY_THRESH

                # find mask contours
                m = pred_mask.astype(np.uint8)
                m[m==1] *= 255
                cnt, _ = cv2.findContours(m,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
                cnt = cnt[0]
                cnt += pred_box[:2]

                data[cls].append({'score': pred_score, 'bbox': pred_box, 'mask': pred_mask, 'contours': cnt})

        return data

    def _format_detection_response(self, detection_data):
        # sorted_idx = np.argsort(scores[cls])  # already sorted!
        # print(list(detection_data['envelope'][0]['bbox']))
        # print([list(d['bbox']) for cls, cls_data in detection_data.items() for d in cls_data if len(d['bbox']) == 4])
        det_data = []
        for cls,cls_data in detection_data.items():
            for d in cls_data:
                bbox = d['bbox']
                if len(bbox) != 4:
                    continue     
                cnt = d['contours'].squeeze()
                # print(cnt.shape)
                # bbox = list(bbox)
                cnt = cnt.reshape(cnt.shape[0]*cnt.shape[1])
                ddd = DetectionData(type=cls,score=float(d['score']),bbox=bbox,contours=cnt)
                det_data.append(ddd)
        return det_data
        # return [DetectionData(type=cls,score=d['score'],bbox=[1204, 100, 1408, 328],contours=list(d['contours'])) for cls, cls_data in detection_data.items() for d in cls_data if len(d['bbox']) == 4]

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='FCIS ROS demo')
    parser.add_argument('--cfg', dest='cfg_file', help='required config file (YAML file)', 
                        required=True, type=str)
    parser.add_argument('--model', dest='model', help='path to trained model (.params file)',
                        required=True, type=str)
    parser.add_argument('--publish', dest='publish', 
                        help='Publish segmentation results for each frame ',
                        action='store_true')
    parser.add_argument('--min_score', dest='min_score', help='Minimum score for detections. Default 0.85',
                            default=0.85, type=float)

    args = parser.parse_args()

    return args

# cur_path + '/../experiments/fcis/cfgs/fcis_coco_demo.yaml'
def main():
    args = parse_args()

    # load config
    cfg_path = args.cfg_file
    assert osp.exists(cfg_path), ("Could not find config file %s"%(cfg_path))
    update_config(cfg_path)
    print("\nLoaded config %s\n"%(cfg_path))
    # pprint.pprint(config)

    # set up class names
    CLASSES = config.dataset.CLASSES
    num_classes = len(CLASSES)

    # load model
    model_path = args.model
    if '.params' not in model_path:
        model_path += ".params"
    assert osp.exists(model_path), ("Could not find model path %s"%(model_path))
    arg_params, aux_params = load_param_file(model_path, process=True)
    print("\nLoaded model %s\n"%(model_path))

    # get symbol
    ctx_id = [int(i) for i in config.gpus.split(',')]
    sym_instance = eval(config.symbol)()
    sym = sym_instance.get_symbol(config, is_train=False)


    # key parameters
    data_names = ['data', 'im_info']
    label_names = []
    max_data_shape = [[('data', (1, 3, max([v[0] for v in config.SCALES]), max([v[1] for v in config.SCALES])))]]

    target_size = config.SCALES[0][0]
    max_size = config.SCALES[0][1]

    data_batch_wr = DataBatchWrapper(target_size, max_size, image_stride=config.network.IMAGE_STRIDE, 
                          pixel_means=config.network.PIXEL_MEANS, data_names=data_names, label_names=label_names)

    im = np.zeros((target_size,max_size,3))
    data_tensor_info = data_batch_wr.get_data_tensor_info(im)

    # get predictor
    predictor = Predictor(sym, data_names, label_names,
                          context=[mx.gpu(ctx_id[0])], max_data_shapes=max_data_shape,
                          provide_data=[[(k, v.shape) for k, v in zip(data_names, data_tensor_info)]], provide_label=[None],
                          arg_params=arg_params, aux_params=aux_params)

    # # warm up predictor
    for i in xrange(2):
        data_batch = mx.io.DataBatch(data=[data_tensor_info], label=[], pad=0, index=0,
                                     provide_data=[[(k, v.shape) for k, v in zip(data_names, data_tensor_info)]],
                                     provide_label=[None])
        scales = [data_batch.data[i][1].asnumpy()[0, 2] for i in xrange(len(data_batch.data))]
        _, _, _, _ = im_detect(predictor, data_batch, data_names, scales, config)

    # init ros mask predictor
    ros_predictor = RosFCISPredictor(predictor, data_batch_wr, CLASSES, publish=args.publish)
    rospy.init_node('fcis_predictor', anonymous=True)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # python ./fcis/demo.py --cfg ./experiments/fcis/cfgs/fcis_coco_demo.yaml --model ./model/fcis_coco-0000.params --img_dir ./demo
    # python fcis/demo.py --cfg ./experiments/fcis/cfgs/resnet_v1_101_vocSDS_fcis_end2end.yaml --model ./output/fcis/voc/resnet_v1_101_vocSDS_fcis_end2end/SDS_train/e2e-0020.params --img_dir ./demo/ 
    main()

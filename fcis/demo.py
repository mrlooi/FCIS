import _init_paths

# import os
import os.path as osp
import glob
import sys
# import logging
# import pprint
import cv2
import numpy as np
import json
from natsort import natsorted as nts

# get config

from config.config import config, update_config
# cur_path = os.path.abspath(os.path.dirname(__file__))

# sys.path.append(".")
# sys.path.insert(0, os.path.join(cur_path, '../external/mxnet', config.MXNET_VERSION))

import mxnet as mx
print("using mxnet at %s"%(mx.__file__))
from core.tester import im_detect, Predictor
from symbols import resnet_v1_101_fcis
from utils.load_model import load_param, load_param_file
from utils.show_masks import show_masks
from utils.tictoc import tic, toc
from nms.nms import py_nms_wrapper
from bbox.bbox_transform import clip_boxes
from mask.mask_transform import gpu_mask_voting, cpu_mask_voting

from utils.image import resize, transform

(CV2_MAJOR, CV2_MINOR, _) = cv2.__version__.split(".")
CV2_MAJOR = int(CV2_MAJOR)
CV2_MINOR = int(CV2_MINOR)

class DataBatchWrapper(object):

    def __init__(self, target_size, max_size, image_stride, pixel_means, data_names = ['data', 'im_info'], label_names = []):
        self.target_size = target_size
        self.max_size = max_size
        self.image_stride = image_stride
        self.pixel_means = pixel_means

        self.data_names = data_names
        self.label_names = label_names
        pass

    def get_data_tensor_info(self, im):
        im_, im_scale = resize(im, self.target_size, self.max_size, stride=self.image_stride)
        im_tensor = transform(im_, self.pixel_means)
        im_info = np.array([[im_tensor.shape[2], im_tensor.shape[3], im_scale]], dtype=np.float32)
        data_tensor_info = [mx.nd.array(im_tensor), mx.nd.array(im_info)]
        
        return data_tensor_info

    def get_data_batch(self, im):
        data_tensor_info = self.get_data_tensor_info(im)
        data_batch = mx.io.DataBatch(data=[data_tensor_info], label=[], pad=0, 
                                             provide_data=[[(k, v.shape) for k, v in zip(self.data_names, data_tensor_info)]],
                                             provide_label=[None])
        return data_batch

class FCISNet(object):
    def __init__(self, cfg_path, model_path):

        # load config
        assert osp.exists(cfg_path), ("Could not find config file %s"%(cfg_path))
        update_config(cfg_path)

        self.cfg = config

        # load model
        if '.params' not in model_path:
            model_path += ".params"
        assert osp.exists(model_path), ("Could not find model path %s"%(model_path))
        self.model_path = model_path

        self.net = None
        self.data_batch_wr = None

        # config params
        self.classes = self.cfg.dataset.CLASSES
        self.num_classes = len(self.classes)
        self.ctx_id = [int(i) for i in self.cfg.gpus.split(',')]

        self.init_net()

    def init_net(self):
        config = self.cfg

        # get symbol
        sym_instance = resnet_v1_101_fcis()
        sym = sym_instance.get_symbol(config, is_train=False)

        # key parameters
        data_names = ['data', 'im_info']
        label_names = []
        max_data_shape = [[('data', (1, 3, max([v[0] for v in config.SCALES]), max([v[1] for v in config.SCALES])))]]

        target_size = config.SCALES[0][0]
        max_size = config.SCALES[0][1]

        self.data_batch_wr = DataBatchWrapper(target_size, max_size, image_stride=config.network.IMAGE_STRIDE, 
                              pixel_means=config.network.PIXEL_MEANS, data_names=data_names, label_names=label_names)

        im = np.zeros((target_size,max_size,3))
        data_tensor_info = self.data_batch_wr.get_data_tensor_info(im)

        # get predictor
        arg_params, aux_params = load_param_file(self.model_path, process=True)
        print("\nLoaded model %s\n"%(self.model_path))

        self.net = Predictor(sym, data_names, label_names,
                              context=[mx.gpu(self.ctx_id[0])], max_data_shapes=max_data_shape,
                              provide_data=[[(k, v.shape) for k, v in zip(data_names, data_tensor_info)]], provide_label=[None],
                              arg_params=arg_params, aux_params=aux_params)
        self.data_names = data_names

        # # warm up predictor
        for i in xrange(2):
            data_batch = self.data_batch_wr.get_data_batch(im)
            scales = [data_batch.data[i][1].asnumpy()[0, 2] for i in xrange(len(data_batch.data))]
            _, _, _, _ = im_detect(self.net, data_batch, data_names, scales, config)

    def forward(self, im, conf_thresh = 0.7):
        data_batch = self.data_batch_wr.get_data_batch(im)

        dets, masks = inference(self.net, data_batch, self.data_names, self.num_classes, self.cfg.BINARY_THRESH, conf_thresh, gpu_id=self.ctx_id[0])

        return dets, masks


def inference(predictor, data_batch, data_names, num_classes, BINARY_THRESH = 0.4, CONF_THRESH=0.7, gpu_id=0):
    scales = [data_batch.data[i][1].asnumpy()[0, 2] for i in xrange(len(data_batch.data))]
    im_shapes = [data_batch.data[i][0].shape[2:4] for i in xrange(len(data_batch.data))]

    scores, boxes, masks, data_dict = im_detect(predictor, data_batch, data_names, scales, config)
    if not config.TEST.USE_MASK_MERGE:
        all_boxes = [[] for _ in xrange(num_classes)]
        all_masks = [[] for _ in xrange(num_classes)]
        nms = py_nms_wrapper(config.TEST.NMS)
        for j in range(1, num_classes):
            indexes = np.where(scores[0][:, j] > CONF_THRESH)[0]
            cls_scores = scores[0][indexes, j, np.newaxis]
            cls_masks = masks[0][indexes, 1, :, :]
            # try:
            #     if config.CLASS_AGNOSTIC:
            #         cls_boxes = boxes[0][indexes, :]
            #     else:
            #         raise Exception()
            # except:
            if config.CLASS_AGNOSTIC:
                cls_boxes = boxes[0][indexes, :]
            else:
                cls_boxes = boxes[0][indexes, j * 4:(j + 1) * 4]

            cls_dets = np.hstack((cls_boxes, cls_scores))
            keep = nms(cls_dets)
            all_boxes[j] = cls_dets[keep, :]
            all_masks[j] = cls_masks[keep, :]
        dets = [all_boxes[j] for j in range(1, num_classes)]
        masks = [all_masks[j] for j in range(1, num_classes)]
    else:
        masks = masks[0][:, 1:, :, :]
        im_height = np.round(im_shapes[0][0] / scales[0]).astype('int')
        im_width = np.round(im_shapes[0][1] / scales[0]).astype('int')
        # print (im_height, im_width)
        boxes = clip_boxes(boxes[0], (im_height, im_width))
        result_masks, result_dets = gpu_mask_voting(masks, boxes, scores[0], num_classes,
                                                    100, im_width, im_height,
                                                    config.TEST.NMS, config.TEST.MASK_MERGE_THRESH,
                                                    BINARY_THRESH, gpu_id)

        dets = [result_dets[j] for j in range(1, num_classes)]
        masks = [result_masks[j][:, 0, :, :] for j in range(1, num_classes)]

    for i in xrange(len(dets)):
        keep = np.where(dets[i][:,-1] > CONF_THRESH)
        dets[i] = dets[i][keep]
        masks[i] = masks[i][keep]

    return dets, masks

def reformat_data(dets, masks, classes):
    data = {}

    for cls_ix, cls in enumerate([c for c in classes if c.lower() != "__background__"]): # ignore bg class
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

            cnt = cv2.findContours(m,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
            cnt = cnt[0] if CV2_MAJOR != 3 else cnt[1]
            cnt = cnt[0]
            cnt += pred_box[:2]

            data[cls].append({'score': pred_score, 'bbox': pred_box, 'mask': pred_mask, 'contours': cnt})

    return data

def write_reformat_data_json(reformatted_data, json_path):
    '''must have data structure returned from reformat_data function'''
    data = []
    for cls,cls_data in reformatted_data.items():
        for d in cls_data:
            bbox = d['bbox']
            if len(bbox) != 4:
                continue     
            cnt = d['contours'].squeeze()
            # print(cnt.shape)
            # bbox = list(bbox)
            cnt = cnt.reshape(cnt.shape[0]*cnt.shape[1])
            ddd = {'type': cls,'score': float(d['score']),'bbox':bbox.tolist(),'contours':cnt.tolist()}
            data.append(ddd)
    with open(json_path, 'w') as f:
        json.dump(data, f)
    print("Saved to %s"%(json_path))


def main():
    import argparse

    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='FCIS demo')
    parser.add_argument('--cfg', dest='cfg_file', help='required config file (YAML file)', 
                        required=True, type=str)
    parser.add_argument('--model', dest='model', help='path to trained model (.params file)',
                        required=True, type=str)
    parser.add_argument('--img_dir', dest='img_dir', help='path to directory of images for demo',
                        required=True, type=str)
    parser.add_argument('--min_score', dest='min_score', help='Minimum score. Default 0.85',
                            default=0.85, type=float)
    parser.add_argument('--save', dest='save', help='Saves inference data per image as JSON files (stored in img_dir directory)',
                         action='store_true')
    parser.add_argument('--novis', dest='novis', help='Turn off visualization of inference',
                         action='store_true')
    parser.add_argument('--wait', dest='wait', help='Set the wait time in between frames in opencv waitKey() (default 0 i.e. pause until button pressed)',
                         default=0, type=int)

    args = parser.parse_args()

    # load image demo directory
    img_dir = args.img_dir
    assert osp.exists(img_dir), ("Could not find image directory %s"%(img_dir))

    image_names = nts(glob.glob(osp.join(img_dir,"*")))

    if len(image_names) == 0:
        print("No files in %s"%(img_dir))
        return

    cfg_path = args.cfg_file
    model_path = args.model

    # load net
    fcis_net = FCISNet(cfg_path, model_path)
    CLASSES = fcis_net.classes

    # test: run predictions
    CONF_THRESH = args.min_score
    print("Using min score of %.3f...\n"%(CONF_THRESH))

    for idx, im_name in enumerate(image_names):
        im = cv2.imread(im_name, cv2.IMREAD_COLOR)# | cv2.IMREAD_IGNORE_ORIENTATION)
        if im is None:
            print("Could not read %s"%(im_name))
            continue
        # im_copy = im.copy()
        
        tic()
        dets, masks = fcis_net.forward(im, conf_thresh=CONF_THRESH)
        print('inference time %s: %.4fs'%(im_name, toc()))

        if args.save:
            im_name_basename = im_name[:im_name.rfind('.')]
            json_file = osp.join(img_dir, im_name_basename + ".json")
            reformatted_data = reformat_data(dets, masks, CLASSES)
            write_reformat_data_json(reformatted_data, json_file)

        # vis
        if not args.novis:
            plt_show = args.wait == 0 
            im_seg = show_masks(im, dets, masks, CLASSES, config.BINARY_THRESH, show=plt_show)
            if not plt_show:
                cv2.imshow("seg", im_seg)
                cv2.waitKey(args.wait) 

    print('\nDONE\n')

if __name__ == '__main__':
    # python ./fcis/demo.py --cfg ./experiments/fcis/cfgs/fcis_coco_demo.yaml --model ./model/fcis_coco-0000.params --img_dir ./demo
    # python fcis/demo.py --cfg ./experiments/fcis/cfgs/resnet_v1_101_vocSDS_fcis_end2end.yaml --model ./output/fcis/voc/resnet_v1_101_vocSDS_fcis_end2end/SDS_train/e2e-0020.params --img_dir ./demo/ 
    main()

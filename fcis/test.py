# --------------------------------------------------------
# Fully Convolutional Instance-aware Semantic Segmentation
# Copyright (c) 2016 by Contributors
# Copyright (c) 2017 Microsoft
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Modified by Guodong Zhang, Haozhi Qi
# --------------------------------------------------------

import _init_paths

import argparse
import os
import os.path as osp
import sys
import cv2
import pprint

import logging

import mxnet as mx

# from function.test_fcis import test_fcis
from dataset import *

from symbols import resnet_v1_101_fcis
from utils.load_model import load_param, load_param_file

from utils.load_data import load_gt_sdsdb
from utils.create_logger import create_logger

from core.loader import TestLoader
from core.tester import Predictor, pred_eval

from config.config import config, update_config

os.environ['PYTHONUNBUFFERED'] = '1'
os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
os.environ['MXNET_ENABLE_GPU_P2P'] = '0'

curr_path = os.path.abspath(os.path.dirname(__file__))
sys.path.append('.')
# sys.path.insert(0, os.path.join(curr_path, '../external/mxnet', config.MXNET_VERSION))

    
def parse_args():
    parser = argparse.ArgumentParser(description='Test a Faster R-CNN network')
    # general
    # configuration file is required
    parser.add_argument('--cfg', help='experiment configure file name', required=True, type=str)

    # model file is required
    parser.add_argument('--model', dest='model', help='path to trained model (.params file)',
                        required=True, type=str)
    # rcnn
    parser.add_argument('--vis', help='turn on visualization', action='store_true')
    parser.add_argument('--ignore_cache', help='ignore cached results boxes', action='store_true')
    parser.add_argument('--thresh', help='valid detection threshold', default=1e-3, type=float)
    parser.add_argument('--shuffle', help='shuffle data on visualization', action='store_true')
    args = parser.parse_args()
    return args


def test_net(args):
    # init config
    cfg_path = args.cfg
    update_config(cfg_path)

    # test parameters
    has_rpn = config.TEST.HAS_RPN
    if not has_rpn:
        raise NotImplementedError, "Network without RPN is not implemented"

    # load model
    model_path = args.model
    if '.params' not in model_path:
        model_path += ".params"
    assert osp.exists(model_path), ("Could not find model path %s"%(model_path))
    arg_params, aux_params = load_param_file(model_path, process=True)
    print("\nLoaded model %s\n"%(model_path))

    # gpu stuff
    ctx = [mx.gpu(int(i)) for i in config.gpus.split(',')]

    # load test dataset
    cfg_ds = config.dataset
    ds_name = cfg_ds.dataset
    ds_path = cfg_ds.dataset_path
    test_image_set = cfg_ds.test_image_set


    # logger
    logger, output_path = create_logger(config.output_path, args.cfg, config.dataset.test_image_set)
    logger.info('testing config:{}\n'.format(pprint.pformat(config)))


    if ds_name.lower() == "labelme":
        # from utils.load_data import load_labelme_gt_sdsdb
        imdb = labelme(test_image_set, ds_path, cfg_ds.root_path, mask_size=config.MASK_SIZE, 
                                binary_thresh=config.BINARY_THRESH, classes=cfg_ds.CLASSES)
    else:
        imdb = eval(ds_name)(test_image_set, cfg_ds.root_path, ds_path, result_path=output_path, 
                             binary_thresh=config.BINARY_THRESH, mask_size=config.MASK_SIZE)
    sdsdb = imdb.gt_sdsdb()

    # load network
    network = resnet_v1_101_fcis()
    sym = network.get_symbol(config, is_train=False)

    # get test data iter
    test_data = TestLoader(sdsdb, config, batch_size=len(ctx), shuffle=args.shuffle, has_rpn=has_rpn)

    # infer shape
    data_shape_dict = dict(test_data.provide_data_single)
    network.infer_shape(data_shape_dict)

    network.check_parameter_shapes(arg_params, aux_params, data_shape_dict, is_train=False)

    # decide maximum shape
    data_names = [k[0] for k in test_data.provide_data_single]
    label_names = []
    max_data_shape = [[('data', (1, 3, max([v[0] for v in config.SCALES]), max([v[1] for v in config.SCALES])))]]

    # # create predictor
    predictor = Predictor(sym, data_names, label_names,
                          context=ctx, max_data_shapes=max_data_shape,
                          provide_data=test_data.provide_data, provide_label=test_data.provide_label,
                          arg_params=arg_params, aux_params=aux_params)

    # print(test_data.provide_data_single[0][1])
    # print(test_data.provide_label)

    # start detection
    pred_eval(predictor, test_data, imdb, config, vis=args.vis, ignore_cache=args.ignore_cache, thresh=args.thresh, logger=logger)


def main():
    args = parse_args()
    print('Called with argument:', args)

    test_net(args)    

    # test_fcis(config, config.dataset.dataset, config.dataset.test_image_set, config.dataset.root_path, config.dataset.dataset_path,
    #           ctx, os.path.join(final_output_path, '..', '_'.join([iset for iset in config.dataset.image_set.split('+')]), config.TRAIN.model_prefix), config.TEST.test_epoch,
    #           args.vis, args.ignore_cache, args.shuffle, config.TEST.HAS_RPN, config.dataset.proposal, args.thresh, logger=logger, output_path=final_output_path)

if __name__ == '__main__':
    main()

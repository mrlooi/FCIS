import _init_paths

import os.path as osp
# import sys
# import logging
import cv2
import numpy as np

# get config
from config.config import config, update_config

from utils.tictoc import tic, toc

from demo import FCISNet, FCISBase

# ros stuff
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

# custom ros stuff
from vision.srv import _Detection2
from vision.msg import DetectionData

KINECT_TOPIC = "/kinect2/hd/image_color"

# def normalize_luminance(img):
#     # convert to YUV colorspace to normalize luminance by performing hist equalization on Y channel
#     image_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
#     image_yuv[:, :, 0] = cv2.equalizeHist(image_yuv[:, :, 0]) 

#     # convert YUV back to RGB
#     image = cv2.cvtColor(image_yuv, cv2.COLOR_YUV2BGR)

#     return image
              
class RosFCISPredictor(FCISBase):
    def __init__(self, net, min_score=0.85, publish=False):
        super(RosFCISPredictor, self).__init__(net, min_score=min_score)
        self.publish = publish

        print("RosFCISPredictor is running. Target classes: %s. Using min score of %.3f..."%(self.classes, self.min_score))

    def init_ros_nodes(self):
        # data 
        self.current_cv_image = None
        self.current_detection_items = []

        self.bridge = CvBridge()

        self.image_sub = rospy.Subscriber(KINECT_TOPIC, Image, self.callback)

        ros_node_extension = '_'.join([c for c in self.classes if c.lower() != "__background__"])
        if self.publish:
            self.image_pub = rospy.Publisher("/kinect2/hd/image_mask_%s"%(ros_node_extension),Image,queue_size=5)

        # service 
        self.image_service = rospy.Service('/kinect2/hd/mask_service_%s'%(ros_node_extension), _Detection2.Detection2, self.callback_service)

    def callback(self, data):
        try:
            im = self.bridge.imgmsg_to_cv2(data, "bgr8")
            self.current_cv_image = im
        except CvBridgeError as e:
            print(e)

        if self.publish:
            inf_data = self.inference(self.current_cv_image)
            self.current_detection_items = self._format_detection_response(inf_data)

            # max_scores = 
            try:
                img_copy = self.draw_segmentation(self.current_cv_image, inf_data)
                self.image_pub.publish(self.bridge.cv2_to_imgmsg(img_copy, "bgr8"))
                cv2.imshow("bounding boxes", img_copy)
                cv2.waitKey(1)
            except CvBridgeError as e:
                print("[ERROR] CvBridgeError: %s"%e)

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
                if len(cnt.shape) < 2:
                    continue
                # print(cnt.shape)
                # bbox = list(bbox)
                cnt = cnt.reshape(cnt.shape[0]*cnt.shape[1])
                ddd = DetectionData(type=cls,score=float(d['score']),bbox=bbox,contours=cnt)
                det_data.append(ddd)
        return det_data
        # return [DetectionData(type=cls,score=d['score'],bbox=[1204, 100, 1408, 328],contours=list(d['contours'])) for cls, cls_data in detection_data.items() for d in cls_data if len(d['bbox']) == 4]


# cur_path + '/../experiments/fcis/cfgs/fcis_coco_demo.yaml'
def main():
    import argparse

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

    cfg_path = args.cfg_file
    model_path = args.model

    # load net
    fcis_net = FCISNet(cfg_path, model_path)

    # init ros mask predictor
    ros_predictor = RosFCISPredictor(fcis_net, publish=args.publish)
    ros_predictor.init_ros_nodes()
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

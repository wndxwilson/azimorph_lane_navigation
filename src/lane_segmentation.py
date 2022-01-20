#!/usr/bin/env python3

import rospy
from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation
from sensor_msgs.msg import Image
import numpy as np
import torch
import ros_numpy
import cv2
import time

class LaneSegmentation():

    def __init__(self):
        # Initialise node
        rospy.init_node("lane_segmentation_node", anonymous=True)
        rospy.loginfo("Running lane segmentation node")

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('Using device:', device)
        #  Load params
        self.image_topic = rospy.get_param('image_topic',"/image/image_raw" )
        # self.palette = torch.tensor([[255, 255, 255], [255, 255, 255], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], 
        #          [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]])
        self.palette = torch.tensor([[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156], [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0], [107, 142, 35], 
            [152, 251, 152], [70, 130, 180], [220, 20, 60], [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100], [0, 80, 100], [0, 0, 230], [119, 11, 32]])
        # Load model
        self.feature_extractor = SegformerFeatureExtractor.from_pretrained("nvidia/segformer-b0-finetuned-cityscapes-1024-1024")
        self.model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-cityscapes-1024-1024")
        rospy.loginfo('Model loaded')

        # Subscribe to image feed
        rospy.Subscriber(self.image_topic,Image,self.handle_image_cb)

        # Publisher
        self.pub = rospy.Publisher('image/image_seg',Image,queue_size=10)


        rospy.spin()

    def handle_image_cb(self,img_msg):
        if(img_msg != None):
            #   Convert Sensor Img -> numpy array
            numpy_img = np.frombuffer(img_msg.data, dtype=np.uint8).reshape(img_msg.height, img_msg.width, -1)
            start_time = time.time()

            inputs = self.feature_extractor(images=numpy_img, return_tensors="pt")
            outputs = self.model.forward(**inputs)
            logits = outputs.logits
            seg = logits.softmax(1).argmax(1).to(int)
            seg_map = self.palette[seg].squeeze().to(torch.uint8)       
            rospy.loginfo(time.time() - start_time)
            msg = ros_numpy.msgify(Image, seg_map.numpy(), encoding='rgb8')

            self.pub.publish(msg)
            cv2.waitKey(1)

            


if __name__ == '__main__':
    try:
        LaneSegmentation()

    except rospy.ROSInterruptException:
        pass
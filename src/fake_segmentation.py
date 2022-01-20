#!/usr/bin/env python3
import rospy
import ros_numpy
from sensor_msgs.msg import  Image
import cv2
import time

class FakeSegmentation():

    def __init__(self):
        # Initialise node
        rospy.init_node("fake_segmentation_node", anonymous=True)
        rospy.loginfo("Running fake segmentation node")

        pub = rospy.Publisher('image/image_raw',Image,queue_size=1)
        video_path = "/home/wilson/catkin_ws/src/lane_navigation/src/vid.mp4"
     
        cap = cv2.VideoCapture(video_path)

        rate = rospy.Rate(1) # 10hz

        current_time = time.time()

        while not rospy.is_shutdown():

            if(time.time() - current_time >= 0.5):
                success, img = cap.read()

                img = cv2.resize(img,(480,240))
                msg = ros_numpy.msgify(Image, img, encoding='rgb8')
                pub.publish(msg)
                current_time = time.time()
            

if __name__ == '__main__':
    try:
        FakeSegmentation()

    except rospy.ROSInterruptException:
        pass
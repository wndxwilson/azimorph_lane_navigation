#!/usr/bin/env python3

import rospy
from std_msgs.msg import Float32
from sensor_msgs.msg import Image
import numpy as np
import utils
import cv2

class LaneNavigation():

    def __init__(self):
        # Initialise node
        rospy.init_node("lane_detection_node", anonymous=True)
        rospy.loginfo("Running lane detection node")

        #  Load params
        self.view_points = rospy.get_param('points',[102, 80, 20, 214 ] )
        self.image_topic = rospy.get_param('image_topic',"/image/image_seg" )
        self.avg = rospy.get_param('avg',10 )
        self.normalise = rospy.get_param('avg',50 )
        self.debug = rospy.get_param('debug',True )

        # Subscribe to image feed
        rospy.Subscriber(self.image_topic,Image,self.handle_image_cb)

        # Publisher curve data
        self.curve_pub = rospy.Publisher("/curvepoint",Float32,queue_size=10)

        # init
        self.curveList = []
        rospy.spin()

    def handle_image_cb(self,img_msg):
        if(img_msg != None):
            #   Convert Sensor Img -> numpy array
            numpy_img = np.frombuffer(img_msg.data, dtype=np.uint8).reshape(img_msg.height, img_msg.width, -1)
            img = cv2.resize(numpy_img,(480,240))

            curve = self.getLaneCurve(img)
            self.curve_pub.publish(curve)
            cv2.waitKey(1)

            print(curve)

    def getLaneCurve(self,img):
        if(self.debug):
            if len(self.curveList)==0:
                intialTrackBarVals = self.view_points
                utils.initializeTrackbars(intialTrackBarVals)
            points = utils.valTrackbars()

        else:
            points = utils.valPoints(self.view_points)

        imgCopy = img.copy()
        imgResult = img.copy()

        #### STEP 1
        imgThres = utils.thresholding(img)
        # imgThres = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        #### STEP 2
        hT, wT, c = img.shape
        imgWarp = utils.warpImg(imgThres,points,wT,hT)
        imgWarpPoints = utils.drawPoints(imgCopy,points)
    
        #### STEP 3
        middlePoint,imgHist = utils.getHistogram(imgWarp,display=True,minPer=0.5,region=4)
        curveAveragePoint, imgHist = utils.getHistogram(imgWarp, display=True, minPer=0.9)
        curveRaw = curveAveragePoint - middlePoint
    
        #### SETP 4
        self.curveList.append(curveRaw)
        if len(self.curveList)>self.avg :
            self.curveList.pop(0)
        curve = int(sum(self.curveList)/len(self.curveList))
        
        
        #### STEP 5
        if self.debug:
            imgInvWarp = utils.warpImg(imgWarp, points, wT, hT, inv=True)
            imgInvWarp = cv2.cvtColor(imgInvWarp, cv2.COLOR_GRAY2BGR)
            imgInvWarp[0:hT // 3, 0:wT] = 0, 0, 0
            imgLaneColor = np.zeros_like(img)
            imgLaneColor[:] = 0, 255, 0
            imgLaneColor = cv2.bitwise_and(imgInvWarp, imgLaneColor)
            imgResult = cv2.addWeighted(imgResult, 1, imgLaneColor, 1, 0)
            midY = 450
            cv2.putText(imgResult, str(curve), (wT // 2 - 80, 85), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 0, 255), 3)
            cv2.line(imgResult, (wT // 2, midY), (wT // 2 + (curve * 3), midY), (255, 0, 255), 5)
            cv2.line(imgResult, ((wT // 2 + (curve * 3)), midY - 25), (wT // 2 + (curve * 3), midY + 25), (0, 255, 0), 5)
            for x in range(-30, 30):
                w = wT // 20
                cv2.line(imgResult, (w * x + int(curve // 50), midY - 10),
                        (w * x + int(curve // 50), midY + 10), (0, 0, 255), 2)
            imgStacked = utils.stackImages(0.7, ([img, imgWarpPoints, imgWarp],
                                                [imgHist, imgLaneColor, imgResult]))
            cv2.imshow('ImageStack', imgStacked)

    
        
        #### NORMALIZATION
        curve = curve/self.normalise
        if curve>1: curve =1
        if curve<= -1:curve = -1
        
        return curve

if __name__ == '__main__':
    try:
        LaneNavigation()

    except rospy.ROSInterruptException:
        pass
#!/usr/bin/env python

import rospy
from std_msgs.msg import Float32
from geometry_msgs.msg import Twist
import time

class LaneFollowing():

    def __init__(self):
        # Initialise node
        rospy.init_node("lane_following_node", anonymous=True)
        rospy.loginfo("Running lane following node")

        # Subscribe to curve value
        rospy.Subscriber("/curvepoint",Float32,self.handle_cb)

        # Publisher
        self.velocity_publisher = rospy.Publisher('/cmd_vel', Twist, queue_size=10)

        # pid params
        self.Kp = 1
        self.Ki = 0.01
        self.Kd = 0
        
        self.last_time = time.time()

        self.cont_e = 0.0
        self.windup_guard = 20.0
        self.last_error = 0.0

        # motor control params
        self.target_speed = 0.5
        self.max_angular_speed = 2.0

        rospy.spin()

    def handle_cb(self,msg):
        
        value = self.pid(0,msg.data)
        cmdVel_msg = self.cmd_vel_control(value)
        self.velocity_publisher.publish(cmdVel_msg)


    def pid(self,set_point, current_point):
        error = set_point - current_point
        current_time = time.time()
        delta_time = current_time - self.last_time
        delta_error = error - self.last_error

        # P
        p = self.Kp * error

        # I
        self.cont_e += error * delta_time
        if self.cont_e > self.windup_guard:
            self.cont_e = self.windup_guard
        elif self.cont_e < -self.windup_guard:
            self.cont_e = -self.windup_guard
        i = self.Ki * self.cont_e

        # D
        d = self.Kd * (delta_error/delta_time)

        # Store last time and error
        self.last_time = current_time
        self.last_error = error

        output = p + i + d
        
        return output

    def cmd_vel_control(self,value):
        vel_msg = Twist()

        vel_msg.linear.x = self.target_speed
        vel_msg.linear.y = 0
        vel_msg.linear.z = 0

        vel_msg.angular.x = 0
        vel_msg.angular.y = 0
        vel_msg.angular.z = value*self.max_angular_speed

        return vel_msg
if __name__ == '__main__':
    try:
        LaneFollowing()

    except rospy.ROSInterruptException:
        pass
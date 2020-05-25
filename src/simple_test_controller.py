#!/usr/bin/env python
import rospy
import numpy as np
from sensor_msgs.msg import Range
from geometry_msgs.msg import Twist


class ThymioController:

    def __init__(self, name):
        self.name = name
        self.hz = 10.0

        rospy.init_node('basic_thymio_controller')

        self.velocity_publisher = rospy.Publisher(self.name + '/cmd_vel',
                                                  Twist, queue_size=10)


        self.rate = rospy.Rate(self.hz)

    def run(self):
        vel_msg = Twist()
        while not rospy.is_shutdown():
            vel_msg.angular.z = 0.4  # rad/s
            self.velocity_publisher.publish(vel_msg)
            self.rate.sleep()


if __name__ == '__main__':
    controller = ThymioController('thymio10')
    try:
        controller.run()
    except rospy.ROSInterruptException as e:
        pass

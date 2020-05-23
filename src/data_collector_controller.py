#!/usr/bin/env python
import rospy
import numpy as np
from sensor_msgs.msg import Range
from geometry_msgs.msg import Twist

wall_threshold = 0.07
max_sensor_dist = 0.12
min_sensor_dist = 0.01


class ThymioController:
    WALKING = 0
    TURNING = 1
    POINTED = 2

    def __init__(self, name):
        self.name = name
        self.state = ThymioController.WALKING
        self.hz = 10.0
        self.dt = 1.0 / self.hz

        rospy.init_node('basic_thymio_controller', disable_signals=True)

        self.velocity_publisher = rospy.Publisher(self.name + '/cmd_vel',
                                                  Twist, queue_size=10)

        rospy.Subscriber(self.name + '/proximity/center', Range, self.update_state)
        rospy.Subscriber(self.name + '/proximity/center_left', Range, self.update_state)
        rospy.Subscriber(self.name + '/proximity/left', Range, self.update_state)
        rospy.Subscriber(self.name + '/proximity/center_right', Range, self.update_state)
        rospy.Subscriber(self.name + '/proximity/right', Range, self.update_state)
        rospy.on_shutdown(self.stop)

        self.rate = rospy.Rate(self.hz)



    def update_state(self, data):
        if data.range < wall_threshold and self.state == ThymioController.WALKING:
            self.state = ThymioController.TURNING

    def move_near_obstacle(self):
        scale = 3.5
        plus = 8

        vel_msg = Twist()
        for ang in np.array([2, -1, -1, -1, -1]) / scale:
            for _ in range(12):
                vel_msg.linear.x = 0
                vel_msg.angular.z = ang
                self.velocity_publisher.publish(vel_msg)
                self.rate.sleep()

            for _ in range(20):
                vel_msg.linear.x = -0.1
                vel_msg.angular.z = 0
                self.velocity_publisher.publish(vel_msg)
                self.rate.sleep()

            for _ in range(3):
                vel_msg.linear.x = 0
                vel_msg.angular.z = 0
                self.velocity_publisher.publish(vel_msg)
                self.rate.sleep()

            for _ in range(20 + plus):
                vel_msg.linear.x = 0.1
                vel_msg.angular.z = 0
                self.velocity_publisher.publish(vel_msg)
                self.rate.sleep()

            for _ in range(8):
                vel_msg.linear.x = -0.1
                vel_msg.angular.z = 0
                self.velocity_publisher.publish(vel_msg)
                self.rate.sleep()

        for _ in range(12):
            vel_msg.linear.x = 0
            vel_msg.angular.z = 2 / scale
            self.velocity_publisher.publish(vel_msg)
            self.rate.sleep()

        vel_msg.linear.x = 0
        vel_msg.angular.z = 0

        self.velocity_publisher.publish(vel_msg)

    def random_turn(self):
        dur = np.random.randint(40, 55)
        ang = np.random.choice([-0.5, 0.5])

        vel_msg = Twist()
        vel_msg.linear.y = 0
        vel_msg.linear.z = 0
        vel_msg.angular.x = 0
        vel_msg.angular.y = 0

        for _ in range(dur):
            vel_msg.linear.x = 0
            vel_msg.angular.z = ang
            self.velocity_publisher.publish(vel_msg)
            self.rate.sleep()

        vel_msg.linear.x = 0
        vel_msg.angular.z = 0

        self.velocity_publisher.publish(vel_msg)

    def stop(self):
        self.velocity_publisher.publish(Twist())
        self.rate.sleep()

    def run(self):
        vel_msg = Twist()
        while not rospy.is_shutdown():
            if self.state == ThymioController.WALKING:
                rospy.loginfo('moving forward')
                vel_msg.linear.x = 0.1  # m/s
                vel_msg.angular.z = 0  # rad/s
                self.velocity_publisher.publish(vel_msg)
                self.rate.sleep()

            if self.state == ThymioController.TURNING:
                rospy.loginfo('moving near the obstacle')
                vel_msg.linear.x = 0
                vel_msg.angular.z = 0
                self.velocity_publisher.publish(vel_msg)
                self.move_near_obstacle()
                self.state = ThymioController.POINTED
                
            if self.state == ThymioController.POINTED:
                rospy.loginfo('turning around')
                self.random_turn()
                self.state = ThymioController.WALKING

if __name__ == '__main__':
    controller = ThymioController('thymio10')
    try:
        controller.run()
    except rospy.ROSInterruptException as e:
        pass
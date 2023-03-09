#!/usr/bin/env python3
from __future__ import division, print_function
import time

import numpy as np
import rospy
import tf_conversions
import tf2_ros
import rosbag
import rospkg

# msgs
from turtlebot3_msgs.msg import SensorState
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Pose, Twist, TransformStamped, Transform, Quaternion
from std_msgs.msg import Empty
from tf.transformations import euler_from_quaternion, quaternion_from_euler

from utils import convert_pose_to_tf, euler_from_ros_quat, ros_quat_from_euler


ENC_TICKS = 4096
RAD_PER_TICK = 0.001533981
WHEEL_RADIUS = 0.032104807719179417 # 0.0345 # .066 / 2 # replace with calibrated values
BASELINE = 0.14317312485378525 # .287 / 2 # replace with calibrated values
INT32_MAX = 2**31

R_MATRIX = np.array(
    [
        [WHEEL_RADIUS/2.0, WHEEL_RADIUS/2.0],
        [WHEEL_RADIUS/(2.0 * BASELINE), -WHEEL_RADIUS/(2.0 * BASELINE)]
    ]
)

def get_rotation(quat: Quaternion):
    orientation_list = [quat.x, quat.y, quat.z, quat.w]
    (_, _, yaw) = euler_from_quaternion(orientation_list)
    return yaw

def safe_del_phi(a, b):
    #Need to check if the encoder storage variable has overflowed
    diff = np.int64(b) - np.int64(a)
    if diff < -np.int64(INT32_MAX): #Overflowed
        delPhi = (INT32_MAX - 1 - a) + (INT32_MAX + b) + 1
    elif diff > np.int64(INT32_MAX) - 1: #Underflowed
        delPhi = (INT32_MAX + a) + (INT32_MAX - 1 - b) + 1
    else:
        delPhi = b - a  
    return delPhi

class WheelOdom:
    def __init__(self):
        # publishers, subscribers, tf broadcaster
        self.sensor_state_sub = rospy.Subscriber('/sensor_state', SensorState, self.sensor_state_cb, queue_size=1)
        self.odom_sub = rospy.Subscriber('/odom', Odometry, self.odom_cb, queue_size=1)
        self.wheel_odom_pub = rospy.Publisher('/odom_est', Odometry, queue_size=1)
        self.tf_br = tf2_ros.TransformBroadcaster()

        # attributes
        self.odom = Odometry()
        self.odom.pose.pose.position.x = 1e10
        self.wheel_odom = Odometry()
        self.wheel_odom.header.frame_id = 'odom'
        self.wheel_odom.child_frame_id = 'wo_base_link'
        self.wheel_odom_tf = TransformStamped()
        self.wheel_odom_tf.header.frame_id = 'odom'
        self.wheel_odom_tf.child_frame_id = 'wo_base_link'
        self.pose = Pose()
        self.pose.orientation.w = 1.0
        self.twist = Twist()
        self.last_enc_l = None
        self.last_enc_r = None
        self.last_time = None

        # rosbag
        rospack = rospkg.RosPack()
        path = rospack.get_path("rob521_lab3")
        self.bag = rosbag.Bag(path+"/motion_estimate.bag", 'w')

        # reset current odometry to allow comparison with this node
        reset_pub = rospy.Publisher('/reset', Empty, queue_size=1, latch=True)
        reset_pub.publish(Empty())
        while not rospy.is_shutdown() and (self.odom.pose.pose.position.x >= 1e-3 or self.odom.pose.pose.position.y >= 1e-3 or
               self.odom.pose.pose.orientation.z >= 1e-2):
            time.sleep(0.2)  # allow reset_pub to be ready to publish
        print('Robot odometry reset.')

        rospy.spin()
        self.bag.close()
        print("saving bag")

    def sensor_state_cb(self, sensor_state_msg):
        # Callback for whenever a new encoder message is published
        # set initial encoder pose
        if self.last_enc_l is None:
            self.last_enc_l = sensor_state_msg.left_encoder
            self.last_enc_r = sensor_state_msg.right_encoder
            self.last_time = sensor_state_msg.header.stamp
        else:
            # update calculated pose and twist with new data
            le = sensor_state_msg.left_encoder
            re = sensor_state_msg.right_encoder

            # # YOUR CODE HERE!!!
            # Update your odom estimates with the latest encoder measurements and populate the relevant area
            # of self.pose and self.twist with estimated position, heading and velocity

            theta_bar = get_rotation(self.pose.orientation)

            print(self.last_enc_r, re, re - self.last_enc_r)
            dphi_r = 2 * np.pi * safe_del_phi(self.last_enc_r, re)  / ENC_TICKS
            dphi_l = 2 * np.pi * safe_del_phi(self.last_enc_l, le)  / ENC_TICKS
            self.last_enc_r = re
            self.last_enc_l = le

            dphi = np.array([[dphi_r], [dphi_l]])
            transformation_matrix = np.array([
                    [np.cos(theta_bar), 0],
                    [np.sin(theta_bar), 0],
                    [0, 1]
                ]
            )
            mu_dot = transformation_matrix @ R_MATRIX @ dphi



            self.pose.position.x = self.pose.position.x + mu_dot[0].item()
            self.pose.position.y = self.pose.position.y + mu_dot[1].item()
            new_theta = theta_bar + mu_dot[2].item()
            quat_arr = quaternion_from_euler(0, 0, new_theta)
            quat = Quaternion(x = quat_arr[0], y = quat_arr[1], z = quat_arr[2], w = quat_arr[3])
            self.pose.orientation = quat

            # do these need to be divided by delta t?
            dt = (rospy.Time.now() - self.last_time).to_sec()
            self.twist.linear.x = mu_dot[0].item() / dt
            self.twist.linear.y = mu_dot[1].item() / dt
            self.twist.angular.z = mu_dot[2].item() / dt

            # publish the updates as a topic and in the tf tree
            current_time = rospy.Time.now()
            self.wheel_odom_tf.header.stamp = current_time
            self.wheel_odom_tf.transform = convert_pose_to_tf(self.pose)
            self.tf_br.sendTransform(self.wheel_odom_tf)

            self.wheel_odom.header.stamp = current_time
            self.wheel_odom.pose.pose = self.pose
            self.wheel_odom.twist.twist = self.twist
            self.wheel_odom_pub.publish(self.wheel_odom)

            self.bag.write('odom_est', self.wheel_odom)
            self.bag.write('odom_onboard', self.odom)

            # for testing against actual odom
            print("Wheel Odom: x: %2.3f, y: %2.3f, t: %2.3f" % (
                self.pose.position.x, self.pose.position.y, new_theta
            ))
            print("Turtlebot3 Odom: x: %2.3f, y: %2.3f, t: %2.3f" % (
                self.odom.pose.pose.position.x, self.odom.pose.pose.position.y,
                euler_from_ros_quat(self.odom.pose.pose.orientation)[2]
            ))

    def odom_cb(self, odom_msg):
        # get odom from turtlebot3 packages
        self.odom = odom_msg

    def plot(self, bag):
        data = {"odom_est":{"time":[], "data":[]}, 
                "odom_onboard":{'time':[], "data":[]}}
        for topic, msg, t in bag.read_messages(topics=['odom_est', 'odom_onboard']):
            print(msg)


if __name__ == '__main__':
    try:
        rospy.init_node('wheel_odometry')
        wheel_odom = WheelOdom()
    except rospy.ROSInterruptException:
        pass

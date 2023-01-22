#!/usr/bin/env python3
import rospy
import math
from geometry_msgs.msg import Twist
from std_msgs.msg import String

def publisher_node():
    """TODO: initialize the publisher node here, \
            and publish wheel command to the cmd_vel topic')"""
    pub = rospy.Publisher("cmd_vel", Twist, queue_size = 1)
    rate = rospy.Rate(10)

    msg = Twist() # stop 
    msg.linear.x = 0
    pub.publish(msg)
    rate.sleep()

    curr_time = rospy.get_rostime()
    lin_vel = 0.1
    ang_vel = -1

    max_dur = rospy.Duration.from_sec(1.0/abs(lin_vel))
    while rospy.get_rostime() - curr_time < max_dur:
        msg = Twist()
        msg.linear.x = lin_vel
        pub.publish(msg)
        rate.sleep()

    msg = Twist() # stop 
    msg.linear.x = 0
    pub.publish(msg)
    rate.sleep()

    curr_time = rospy.get_rostime()
    max_dur = rospy.Duration.from_sec(2 * math.pi/abs(ang_vel))
    while rospy.get_rostime() - curr_time < max_dur:
        msg = Twist()
        msg.angular.z = ang_vel
        pub.publish(msg)
        rate.sleep()

    msg = Twist() # stop 
    msg.angular.z = 0
    pub.publish(msg)
    rate.sleep()

def main():
    try:
        rospy.init_node('motor')
        publisher_node()
    except rospy.ROSInterruptException:
        pass


if __name__ == "__main__":
    main()

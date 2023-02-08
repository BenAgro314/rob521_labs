#r!/usr/bin/env python3
from __future__ import division, print_function
import os

import numpy as np
from scipy.linalg import block_diag
from scipy.spatial.distance import cityblock
import rospy
import tf2_ros
from nodes.l2_planning import PathPlanner

# msgs
from geometry_msgs.msg import TransformStamped, Twist, PoseStamped
from nav_msgs.msg import Path, Odometry, OccupancyGrid
from visualization_msgs.msg import Marker

# ros and se2 conversion utils
import utils


TRANS_GOAL_TOL = .1  # m, tolerance to consider a goal complete
ROT_GOAL_TOL = .3  # rad, tolerance to consider a goal complete
TRANS_VEL_OPTS = np.linspace(-0.26, 0.26, 11) # m/s, max of real robot is .26
ROT_VEL_OPTS = np.linspace(-1.82, 1.82, 11)  # rad/s, max of real robot is 1.82
CONTROL_RATE = 5  # Hz, how frequently control signals are sent
CONTROL_HORIZON = 5.0  # seconds. if this is set too high and INTEGRATION_DT is too low, code will take a long time to run!
INTEGRATION_DT = .025  # s, delta t to propagate trajectories forward by
COLLISION_RADIUS = 0.225  # m, radius from base_link to use for collisions, min of 0.2077 based on dimensions of .281 x .306
ROT_DIST_MULT = .1  # multiplier to change effect of rotational distance in choosing correct control
OBS_DIST_MULT = .1  # multiplier to change the effect of low distance to obstacles on a path
MIN_TRANS_DIST_TO_USE_ROT = TRANS_GOAL_TOL  # m, robot has to be within this distance to use rot distance in cost
#PATH_NAME = 'path_complete.npy'  # saved path from l2_planning.py, should be in the same directory as this file
PATH_NAME = 'shortest_path_rrt_star.npy'  # saved path from l2_planning.py, should be in the same directory as this file

# here are some hardcoded paths to use if you want to develop l2_planning and this file in parallel
#TEMP_HARDCODE_PATH = [[2, 0, 0], [2.75, -1, -np.pi/2], [2.75, -4, -np.pi/2], [2, -4.4, np.pi]]  # almost collision-free
TEMP_HARDCODE_PATH = [[2, -.5, 0], [2.4, -1, -np.pi/2], [2.45, -3.5, -np.pi/2], [1.5, -4.4, np.pi]]  # some possible collisions


class PathFollower():
    def __init__(self):
        # time full path
        self.path_follow_start_time = rospy.Time.now()

        # use tf2 buffer to access transforms between existing frames in tf tree
        self.tf_buffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tf_buffer)
        rospy.sleep(1.0)  # time to get buffer running

        # constant transforms
        self.map_odom_tf = self.tf_buffer.lookup_transform('map', 'odom', rospy.Time(0), rospy.Duration(2.0)).transform

        # subscribers and publishers
        self.cmd_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        self.global_path_pub = rospy.Publisher('~global_path', Path, queue_size=1, latch=True)
        self.local_path_pub = rospy.Publisher('~local_path', Path, queue_size=1)
        #self.local_path_candidates_pub = rospy.Publisher('~local_path_candidates', Path, queue_size=1)
        self.collision_marker_pub = rospy.Publisher('~collision_marker', Marker, queue_size=1)

        # map
        map = rospy.wait_for_message('/map', OccupancyGrid)
        self.map_np = np.array(map.data).reshape(map.info.height, map.info.width)
        self.map_resolution = round(map.info.resolution, 5)
        self.map_origin = -utils.se2_pose_from_pose(map.info.origin)  # negative because of weird way origin is stored
        self.map_nonzero_idxes = np.argwhere(self.map_np)

        # collisions
        self.collision_radius_pix = COLLISION_RADIUS / self.map_resolution
        self.collision_marker = Marker()
        self.collision_marker.header.frame_id = '/map'
        self.collision_marker.ns = '/collision_radius'
        self.collision_marker.id = 0
        self.collision_marker.type = Marker.CYLINDER
        self.collision_marker.action = Marker.ADD
        self.collision_marker.scale.x = COLLISION_RADIUS * 2
        self.collision_marker.scale.y = COLLISION_RADIUS * 2
        self.collision_marker.scale.z = 1.0
        self.collision_marker.color.g = 1.0
        self.collision_marker.color.a = 0.5

        # transforms
        self.map_baselink_tf = self.tf_buffer.lookup_transform('map', 'base_link', rospy.Time(0), rospy.Duration(2.0))
        self.pose_in_map_np = np.zeros(3)
        self.pos_in_map_pix = np.zeros(2)
        self.update_pose()

        # path variables
        cur_dir = os.path.dirname(os.path.realpath(__file__))

        # to use the temp hardcoded paths above, switch the comment on the following two lines
        self.path_tuples = np.load(os.path.join(cur_dir, PATH_NAME)).T # (N, 3)
        # self.path_tuples = np.array(TEMP_HARDCODE_PATH)

        self.path = utils.se2_pose_list_to_path(self.path_tuples, 'map')
        self.global_path_pub.publish(self.path)

        # goal
        self.cur_path_index = 1
        self.cur_goal = np.array(self.path_tuples[self.cur_path_index])

        # trajectory rollout tools
        # self.all_opts is a Nx2 array with all N possible combinations of the t and v vels, scaled by integration dt
        self.all_opts = np.array(np.meshgrid(TRANS_VEL_OPTS, ROT_VEL_OPTS)).T.reshape(-1, 2)

        # if there is a [0, 0] option, remove it
        all_zeros_index = (np.abs(self.all_opts) < [0.001, 0.001]).all(axis=1).nonzero()[0]
        if all_zeros_index.size > 0:
            self.all_opts = np.delete(self.all_opts, all_zeros_index, axis=0)
        self.all_opts_scaled = self.all_opts * INTEGRATION_DT

        self.num_opts = self.all_opts_scaled.shape[0]
        self.horizon_timesteps = int(np.ceil(CONTROL_HORIZON / INTEGRATION_DT))

        self.rate = rospy.Rate(CONTROL_RATE)

        rospy.on_shutdown(self.stop_robot_on_shutdown)

        map_filename = "willowgarageworld_05res.png"
        map_settings_filename = "willowgarageworld_05res.yaml"
        self.planner = PathPlanner(map_filename, map_settings_filename, self.path_tuples[-1, None].T[:2], 0.5)

        self.prev_control = None
        self.prev_path = None

        self.follow_path()

    def follow_path(self):
        while not rospy.is_shutdown():
            # timing for debugging...loop time should be less than 1/CONTROL_RATE
            tic = rospy.Time.now()

            self.update_pose()
            self.check_and_update_goal()

            #print(f"Curr path index: {self.cur_path_index}")

            #if np.linalg.norm(self.pose_in_map_np - self.cur_goal, axis = -1) < 0.1:
            #    self.cur_path_index += 1
            #    if self.cur_path_index > self.path_tuples.shape[0]:
            #        break # at goal
            #    self.cur_goal = np.array(self.path_tuples[self.cur_path_index])

            dist_from_goal = np.linalg.norm(self.pose_in_map_np[:2] - self.cur_goal[:2])

            control_horizon = np.clip(dist_from_goal / 0.26, a_min = 1, a_max = 5)

            # start trajectory rollout algorithm
            local_paths = self.planner.trajectory_rollout(
                self.all_opts[:, 0:1],
                self.all_opts[:, 1:2],
                self.pose_in_map_np[:, None],
                fix_col = False,
                timestep = control_horizon,
                num_substeps = np.ceil(CONTROL_HORIZON / INTEGRATION_DT).astype(np.int)
            ) # (num, T, 3)
            #local_paths = np.zeros([self.horizon_timesteps + 1, self.num_opts, 3])
            #local_paths[0] = np.atleast_2d(self.pose_in_map_np).repeat(self.num_opts, axis=0)
            col_traj_mask = np.any(np.any(np.isnan(local_paths), axis = -1), axis = -1) # (N)
            valid_opts = np.array(range(self.num_opts))
            local_paths = local_paths[~col_traj_mask] # remove all coliding trajectories
            valid_opts = valid_opts[~col_traj_mask]

            #for i in range(local_paths.shape[0]):
                #self.local_path_candidates_pub.publish(utils.se2_pose_list_to_path(local_paths[i], 'map'))

            #print("TO DO: Propogate the trajectory forward, storing the resulting points in local_paths!")
            #for t in range(1, self.horizon_timesteps + 1):
            #    # propogate trajectory forward, assuming perfect control of velocity and no dynamic effects
            #    pass

            # check all trajectory points for collisions
            # first find the closest collision point in the map to each local path point

            #local_paths_pixels = (self.map_origin[:2] + local_paths[:, :, :2]) / self.map_resolution
            #local_paths_lowest_collision_dist = np.ones(self.num_opts) * 50

            #print("TO DO: Check the points in local_path_pixels for collisions")
            #for opt in range(local_paths_pixels.shape[1]):
            #    for timestep in range(local_paths_pixels.shape[0]):
            #        pass

            ## remove trajectories that were deemed to have collisions
            #print("TO DO: Remove trajectories with collisions!")

            # calculate final cost and choose best option
            #print("TO DO: Calculate the final cost and choose the best control option!")
            #final_cost = np.zeros(local_paths.shape[0])
            #if final_cost.size == 0:  # hardcoded recovery if all options have collision
            #    control = [-.1, 0]
            #else:
            #    best_opt = valid_opts[final_cost.argmin()]
            #    control = self.all_opts[best_opt]
            #    self.local_path_pub.publish(utils.se2_pose_list_to_path(local_paths[:, best_opt], 'map'))


            #cost_endpoint_trans_dist = np.min(np.linalg.norm(local_paths[:, :, :2] - self.cur_goal[:2], axis = -1), axis = -1)
            cost_endpoint_trans_dist = np.linalg.norm(local_paths[:, -1, :2] - self.cur_goal[:2], axis = -1)
            #dangle = np.min(local_paths[:, :, 2] - self.cur_goal[2], axis = -1)
            dangle = local_paths[:, -1, 2] - self.cur_goal[2]
            cost_endpoint_rot_dist = np.abs(np.arctan2(np.sin(dangle), np.cos(dangle)))
            cost_endpoint_rot_dist = np.where(cost_endpoint_rot_dist > np.pi, cost_endpoint_rot_dist - 2*np.pi,  cost_endpoint_rot_dist)
            #print(local_paths[:, -1, 2], self.cur_goal[2])
            #print(cost_endpoint_rot_dist)


            rot_mult = ROT_DIST_MULT if np.linalg.norm(self.pose_in_map_np[:2] - self.cur_goal[:2]) < MIN_TRANS_DIST_TO_USE_ROT else 0

            sim_cost = np.zeros_like(cost_endpoint_trans_dist)
            #if self.prev_control is not None:
            #    controls = self.all_opts[valid_opts] 
            #    sim_cost = np.linalg.norm(controls - self.prev_control, ord = 1, axis = -1)

            #if self.prev_path is not None:
                #sim_cost = np.sum(np.linalg.norm(self.prev_path[1:, :2] - local_paths[:, :-1, :2], axis = -1), axis = -1)

            #print(f"Trans cost: {cost_endpoint_trans_dist}")
            #print(f"Rot cost: {rot_mult * cost_endpoint_rot_dist}")
            #print(f"Sim cost: {sim_cost / local_paths.shape[1]}")
            
            final_cost = cost_endpoint_trans_dist + rot_mult * cost_endpoint_rot_dist #+  (sim_cost / local_paths.shape[1])

            if final_cost.size == 0: # hardcoded recovery if all options have collision
                control = [-.1, 0]
            else:
                best_opt = valid_opts[final_cost.argmin()]
                control = self.all_opts[best_opt]
                traj = local_paths[final_cost.argmin()]
                self.local_path_pub.publish(utils.se2_pose_list_to_path(traj, 'map'))
                self.prev_path = traj

            self.prev_control = control

            # send command to robot
            self.cmd_pub.publish(utils.unicyle_vel_to_twist(control))

            # uncomment out for debugging if necessary
            #print("Selected control: {control}, Loop time: {time}, Max time: {max_time}".format(
            #    control=control, time=(rospy.Time.now() - tic).to_sec(), max_time=1/CONTROL_RATE))

            self.rate.sleep()

    def update_pose(self):
        # Update numpy poses with current pose using the tf_buffer
        self.map_baselink_tf = self.tf_buffer.lookup_transform('map', 'base_link', rospy.Time(0)).transform
        self.pose_in_map_np[:] = [self.map_baselink_tf.translation.x, self.map_baselink_tf.translation.y,
                                  utils.euler_from_ros_quat(self.map_baselink_tf.rotation)[2]]
        self.pos_in_map_pix = (self.map_origin[:2] + self.pose_in_map_np[:2]) / self.map_resolution
        self.collision_marker.header.stamp = rospy.Time.now()
        self.collision_marker.pose = utils.pose_from_se2_pose(self.pose_in_map_np)
        self.collision_marker_pub.publish(self.collision_marker)

    def check_and_update_goal(self):
        # iterate the goal if necessary
        dist_from_goal = np.linalg.norm(self.pose_in_map_np[:2] - self.cur_goal[:2])
        #abs_angle_diff = np.abs(self.pose_in_map_np[2] - self.cur_goal[2])
        dangle = self.pose_in_map_np[2] - self.cur_goal[2]
        rot_dist_from_goal = np.abs(np.arctan2(np.sin(dangle), np.cos(dangle)))
        #rot_dist_from_goal = min(np.pi * 2 - abs_angle_diff, abs_angle_diff)
        if dist_from_goal < TRANS_GOAL_TOL and rot_dist_from_goal < ROT_GOAL_TOL:
            rospy.loginfo("Goal {goal} at {pose} complete.".format(
                    goal=self.cur_path_index, pose=self.cur_goal))
            if self.cur_path_index == len(self.path_tuples) - 1:
                rospy.loginfo("Full path complete in {time}s! Path Follower node shutting down.".format(
                    time=(rospy.Time.now() - self.path_follow_start_time).to_sec()))
                rospy.signal_shutdown("Full path complete! Path Follower node shutting down.")
            else:
                self.cur_path_index += 1
                self.cur_goal = np.array(self.path_tuples[self.cur_path_index])
        else:
            rospy.logdebug("Goal {goal} at {pose}, trans error: {t_err}, rot error: {r_err}.".format(
                goal=self.cur_path_index, pose=self.cur_goal, t_err=dist_from_goal, r_err=rot_dist_from_goal
            ))

    def stop_robot_on_shutdown(self):
        self.cmd_pub.publish(Twist())
        rospy.loginfo("Published zero vel on shutdown.")


if __name__ == '__main__':
    try:
        rospy.init_node('path_follower', log_level=rospy.DEBUG)
        pf = PathFollower()
    except rospy.ROSInterruptException:
        pass
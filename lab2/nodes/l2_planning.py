#!/usr/bin/env python3
#Standard Libraries
import numpy as np
import yaml
import os
import pygame
import time
from nodes.utils import get_maps_dir
import pygame_utils
import matplotlib.image as mpimg
import math
from skimage.draw import disk
from functools import cached_property
from scipy.linalg import block_diag
from tf import transformations

#Map Handling Functions
def load_map(filename):
    im = mpimg.imread(os.path.join(get_maps_dir(), filename))
    im_np = np.array(im)  #Whitespace is true, black is false
    #im_np = np.logical_not(im_np)    
    return im_np

def load_map_yaml(filename):
    with open(os.path.join(get_maps_dir(), filename), "r") as stream:
            map_settings_dict = yaml.safe_load(stream)
    return map_settings_dict

#Node for building a graph
class Node:
    def __init__(self, point, parent_id, cost):
        self.point = point # A 3 by 1 vector [x, y, theta]
        self.parent_id = parent_id # The parent node id that leads to this node (There should only every be one parent in RRT)
        self.cost = cost # The cost to come to this node
        self.children_ids = [] # The children node ids of this node
        return


#Path Planner 
class PathPlanner:
    #A path planner capable of perfomring RRT and RRT*
    def __init__(self, map_filename, map_setings_filename, goal_point, stopping_dist):
        #Get map information
        self.occupancy_map = load_map(map_filename)
        self.sampled_points = np.zeros_like(self.occupancy_map).astype(bool)
        self.node_points = np.zeros_like(self.occupancy_map).astype(bool)
        self.map_shape = self.occupancy_map.shape
        self.map_settings_dict = load_map_yaml(map_setings_filename)

        #Get the metric bounds of the map
        self.bounds = np.zeros([2,2]) #m
        self.bounds[0, 0] = self.map_settings_dict["origin"][0] # -21
        self.bounds[1, 0] = self.map_settings_dict["origin"][1] # -49.25
        self.bounds[0, 1] = self.map_settings_dict["origin"][0] + self.map_shape[1] * self.map_settings_dict["resolution"] # 59
        self.bounds[1, 1] = self.map_settings_dict["origin"][1] + self.map_shape[0] * self.map_settings_dict["resolution"] # 30.75

        #Robot information
        self.robot_radius = 0.22 #m
        self.vel_max = 0.5 #m/s (Feel free to change!)
        self.rot_vel_max = 0.2 #rad/s (Feel free to change!)

        #Goal Parameters
        self.goal_point = goal_point #m
        self.stopping_dist = stopping_dist #m

        #Trajectory Simulation Parameters
        self.timestep = 1.0 #s
        self.num_substeps = 10

        #Planning storage
        self.start_point = np.array([[0], [0], [0]])
        self.nodes = [Node(self.start_point, -1, 0)] # assuming robot starts at origin
        inds = self.point_to_cell(self.nodes[0].point[:-1])[0]
        self.sampled_points[inds[1], inds[0]] = 1

        #RRT* Specific Parameters
        self.lebesgue_free = np.sum(self.occupancy_map) * self.map_settings_dict["resolution"] **2
        self.zeta_d = np.pi
        self.gamma_RRT_star = 2 * (1 + 1/2) ** (1/2) * (self.lebesgue_free / self.zeta_d) ** (1/2)
        self.gamma_RRT = self.gamma_RRT_star + .1
        self.epsilon = 2.5
        
        #Pygame window for visualization
        self.window = pygame_utils.PygameWindow(
            "Path Planner", (1000, 1000), self.occupancy_map.shape, self.map_settings_dict, self.goal_point, self.stopping_dist)
        return


    @cached_property
    def t_wm(self):
        # Returns t_wm --- transformation matrix from coordinates in map frame to coordinates in world frame
        x, y, theta = self.map_settings_dict["origin"]
        t_wm = transformations.euler_matrix(0, 0, theta)
        t_wm[0, -1] = x
        t_wm[1, -1] = y
        return t_wm

    @cached_property
    def t_mw(self):
        return np.linalg.inv(self.t_wm)

    #Functions required for RRT
    def sample_map_space(self):
        #Return an [x,y] coordinate to drive the robot towards
        #print("TO DO: Sample point to drive towards")

        # sample point around current points

        sample_goal = np.random.rand() < 0.1
        # sample points from un-occupied space
        theta = np.random.rand() * 2 * np.pi  - np.pi # [pi, -pi]
        if not sample_goal:

            #x = np.random.rand() * (self.bounds[0, 1] - self.bounds[0, 0])  + self.bounds[0, 0] 
            #y = np.random.rand() * (self.bounds[1, 1] - self.bounds[1, 0])  + self.bounds[1, 0] 
            #theta = np.random.rand() * 2 * np.pi  - np.pi # [pi, -pi]
            #point = np.array([[x], [y], [theta]])
            

            #row, col = np.nonzero(self.sampled_points)
            #res = self.map_settings_dict["resolution"]
            #min_row = np.min(row) - int(self.vel_max / res)
            #max_row = np.max(row) + int(self.vel_max / res)
            #min_col = np.min(col) - int(self.vel_max / res)
            #max_col = np.max(col) + int(self.vel_max / res)

            #inds = np.nonzero(self.occupancy_map) #np.logical_and(self.occupancy_map, (~self.sampled_points)))
            #inds = np.stack(inds, axis = -1) # num, 2
            #valid_inds = inds[np.logical_and(np.logical_and(inds[:, 0]>=min_row, inds[:, 0]<=max_row), np.logical_and(inds[:, 1]>=min_col, inds[:, 1]<=max_col))]
            #ind = np.random.choice(valid_inds.shape[0]) # (2, )
            #ind = valid_inds[ind]
            #self.sampled_points[ind[0], ind[1]] = 1
            #xy = self.cell_to_point(ind[:, None])[0]
            #y = (res * ind[0]) + self.map_settings_dict["origin"][1]
            #x = (res * ind[1]) + self.map_settings_dict["origin"][0]
            #point = np.array([[xy[0]], [xy[1]], [theta]])
            r = np.abs(np.linalg.norm(self.goal_point - self.start_point[:-1]) * np.random.randn())
            dphi = np.random.rand() * 2 * np.pi  - np.pi # [pi, -pi]
            dx = r * np.cos(dphi)
            dy = r * np.sin(dphi)
            point = np.array([[self.goal_point[0, 0] +dx], [self.goal_point[1, 0] + dy], [theta]])
        else:
            theta = np.random.rand() * 2 * np.pi  - np.pi # [pi, -pi]
            dx = self.stopping_dist * np.random.randn()
            dy = self.stopping_dist * np.random.randn()
            point = np.array([[self.goal_point[0, 0] +dx], [self.goal_point[1, 0] + dy], [theta]])
        return point 
    
    def check_if_duplicate(self, ind):
        #Check if point is a duplicate of an already existing node
        # TODO
        print("TO DO: Check that nodes are not duplicates")
        # point: (3, 1) world point
        if self.sampled_points[ind[1], ind[0]] == 1:
            return True
        return False
    
    def closest_node(self, point):
        #Returns the index of the closest node
        # TODO: replace with nearest neighbours

        best_ind = None
        min_dist = np.inf

        for i in range(len(self.nodes)):
            pt = self.nodes[i].point
            d = np.linalg.norm(pt[:-1] - point[:-1])
            if d < min_dist:
                best_ind = i
                min_dist = d

        return best_ind
    
    def simulate_trajectory(self, point_i, point_s):
        #Simulates the non-holonomic motion of the robot.
        #This function drives the robot from node_i towards point_s. This function does has many solutions!
        #node_i is a 3 by 1 vector [x;y;theta] this can be used to construct the SE(2) matrix T_{OI} in course notation
        #point_s is the sampled point vector [x; y]
        vel, rot_vel = self.robot_controller(point_i, point_s)
        vel = vel[None, :]
        rot_vel = rot_vel[None, :]
        robot_traj = self.trajectory_rollout(vel, rot_vel, point_i)[0]
        return robot_traj # (num_substeps, 3)
    
    def robot_controller(self, point_i, point_s):
        #This controller determines the velocities that will nominally move the robot from node i to node s
        #Max velocities should be enforced
        # point_i = [x_i; y_i; theta_i], shape (3, 1)
        # point_s = [x_s; y_s; theta_s], shape (3, 1)

        # grid search for best over possible
        # TODO: make this better

        vel_range = np.linspace(-self.vel_max, self.vel_max, 5)
        omega_range = np.linspace(-self.rot_vel_max, self.rot_vel_max, 5)
        vo = np.dstack(np.meshgrid(vel_range, omega_range)).reshape(-1, 2)
        pts = self.trajectory_rollout(vo[:, 0:1], vo[:, 1:2], point_i)[:, -1:, :] # (N, 1, 3)
        point_s = point_s[None, :, 0]
        pts = pts[:, 0, :]
        #dtheta = np.arctan2(np.sin(point_s[:, -1]),np.cos(pts[:, -1]))
        best_input_ind = np.argmin(np.linalg.norm(point_s[:, :-1] - pts[:, :-1], axis = -1, ord = 1), axis = 0)
        v_omega = vo[best_input_ind]
        return v_omega[0:1], v_omega[1:2] # (1, ), (1,)

        #dtheta = np.arctan2(np.sin(point_s[-1]), np.cos(point_i[-1]))
        #dx = point_s[0] - point_i[0]
        #omega = np.clip(dtheta / self.timestep, -self.rot_vel_max, self.rot_vel_max)

        #if omega == 0:
        #    v = dx / (self.timestep * np.cos(point_i[-1]))
        #else:
        #    ds = np.sin(point_s[-1]) - np.sin(point_i[-1])
        #    dc = np.cos(point_s[-1]) - np.cos(point_i[-1])
        #    if ds > dc:
        #        v = dx * omega / ds
        #    else:
        #        v = dx * omega / dc

        #return np.clip(v, -self.vel_max, self.vel_max), omega # (1,), (1,) numpy arrays
    
    def trajectory_rollout(self, vel, rot_vel, x_y_theta):
        # Given your chosen velocities determine the trajectory of the robot for your given timestep
        # The returned trajectory should be a series of points to check for collisions

        # x_y_theta: (x, y, theta), (3, 1)
        # vel.shape: (N, 1)
        # rot_vel: (N, 1)

        t = np.linspace(0, self.timestep, self.num_substeps)[None, :] # (1, num_substeps)
        x0 = x_y_theta[0:1] # (N, 1)
        y0 = x_y_theta[1:2] # (N, 1)
        theta0 = x_y_theta[2:3] # (N, 1)

        x = np.zeros((rot_vel.shape[0], t.shape[1])) # (N, num_substeps)
        y = np.zeros((rot_vel.shape[0], t.shape[1]))

        x = np.where(rot_vel == 0, vel * t * np.cos(theta0) + x0, (vel / rot_vel)  * (np.sin(rot_vel * t + theta0) - np.sin(theta0)) + x0)
        y = np.where(rot_vel == 0, vel * t * np.sin(theta0) + y0, -(vel / rot_vel)  * (np.cos(rot_vel * t + theta0) - np.cos(theta0)) + y0)
        theta = np.where(rot_vel == 0, theta0 * np.ones_like(rot_vel * t), rot_vel * t + theta0)

        # to check collisions, check if any of the x,y pairs result in collision.
        # take the maximum timestep that has no collision and copy that to all timesteps
        pts = np.stack((x, y), axis = -1) # (N, self.num_substeps, 2)
        p_w = pts.reshape((-1, 2)).T # (2, N * self.num_substeps)
        collisions = self.is_colliding(p_w).reshape((vel.shape[0], self.num_substeps)) # (N*self.num_substeps)
        np.maximum.accumulate(collisions, axis = 1, out = collisions)

        res = np.stack((x, y, theta), axis = -1) 
        mask = (collisions == 1)
        idx = np.where(~mask, np.arange(mask.shape[1]), 0)
        np.maximum.accumulate(idx, axis = 1, out = idx)
        res = res[np.arange(idx.shape[0])[:, None], idx]


        #arrived_to_pt = res[0][-1][:, None]  # (3, 1)
        #print(arrived_to_pt)
        ##self.window.add_point(arrived_to_pt[:-1, 0])
        #if self.is_colliding(arrived_to_pt[:-1]):
        #    print("err")
        #    pass

        return res # (N, self.num_substeps, 3)

    def is_colliding(self, p_w):
        # p_w.shape = (2, N)
        footprints = self.points_to_robot_circle(p_w) # (N, pts_per_circle, 2)
        # TODO: batch over footprints. Problem: some footprints have different number of points
        #res = np.zeros((p_w.shape[1], )) # (N, )
        is_col = np.any(self.occupancy_map[footprints[..., 1], footprints[..., 0]] == 0, axis = -1)# (N, pts_per_circle)
        return is_col

    def cell_to_point(self, ind):
        # ind is a 2 x N matrix of indices
        p_m = np.zeros_like(ind)
        res = self.map_settings_dict["resolution"]
        p_m[0, :] = ind[1] * res
        h = self.map_shape[1] * res
        p_m[1, :] = h - ind[0] * res
        p_m = p_m.T
        p_m = np.concatenate((p_m, np.zeros_like(p_m[:, -1:]), np.ones_like(p_m[:, -1:])), axis = -1)[:, :, None] # (N, 4, 1)
        p_w = self.t_wm @ p_m
        return p_w[:, :2, 0] # (N, 2)

    def point_to_cell(self, p_w):
        #Convert a series of [x,y] points in the map to the indices for the corresponding cell in the occupancy map
        #p_w is a 2 by N matrix of points of interest of points in the world coords
        num_pts = p_w.shape[-1] 
        res = self.map_settings_dict["resolution"]
        h = self.map_shape[1] * res
        assert p_w.shape == (2, num_pts)
        p_w = p_w.T
        p_w = np.concatenate((p_w, np.zeros_like(p_w[:, -1:]), np.ones_like(p_w[:, -1:])), axis = -1)[:, :, None] # (N, 4, 1)
        p_m = self.t_mw @ p_w
        x_idx = (p_m[:, 0] / res).astype(int)
        y_idx = ((h - p_m[:, 1]) / res).astype(int)
        return np.concatenate((x_idx, y_idx), axis = -1) # (N, 2)

    def points_to_robot_circle(self, points):
        #Convert a series of [x,y] points to robot map footprints for collision detection
        #Hint: The disk function is included to help you with this function
        #print("TO DO: Implement a method to get the pixel locations of the robot path")
        inds = self.point_to_cell(points) # px locations (N, 2)
        res = self.map_settings_dict["resolution"]
        radius = self.robot_radius / res # radius in px
        footprints = [] # one for each point
        # TODO: how to batch?
        for ind in inds:
            rr, cc = disk((ind[0], ind[1]), radius) #, shape = self.map_shape)
            rr[rr >= self.occupancy_map.shape[0]] = self.occupancy_map.shape[0] - 1
            cc[cc >= self.occupancy_map.shape[1]] = self.occupancy_map.shape[1] - 1
            rr[rr < 0] = 0
            cc[cc < 0] = 0
            footprints.append(np.stack((rr,cc), axis = -1))
        return np.stack(footprints, axis = 0) # (num_pts, num_pts_per_circle, 2)
    #Note: If you have correctly completed all previous functions, then you should be able to create a working RRT function

    #RRT* specific functions
    def ball_radius(self):
        #Close neighbor distance
        card_V = len(self.nodes)
        return min(self.gamma_RRT * (np.log(card_V) / card_V ) ** (1.0/2.0), self.epsilon)
    
    def connect_node_to_point(self, node_i, point_f):
        #Given two nodes find the non-holonomic path that connects them
        #Settings
        #node is a 3 by 1 node
        #point is a 2 by 1 point
        print("TO DO: Implement a way to connect two already existing nodes (for rewiring).")
        return np.zeros((3, self.num_substeps))
    
    def cost_to_come(self, trajectory_o):
        #The cost to get to a node from lavalle 
        print("TO DO: Implement a cost to come metric")
        return 0
    
    def update_children(self, node_id):
        #Given a node_id with a changed cost, update all connected nodes with the new cost
        print("TO DO: Update the costs of connected nodes after rewiring.")
        return

    #Planner Functions
    def rrt_planning(self):
        #This function performs RRT on the given map and robot
        #You do not need to demonstrate this function to the TAs, but it is left in for you to check your work
        #for i in range(1): #Most likely need more iterations than this to complete the map!
        i = 0
        while True:
            #Sample map space
            #print(i)
            point = self.sample_map_space()
            self.window.add_point(point[:-1, 0].copy(), color = (255, 0, 0))

            #Get the closest point
            closest_node_id = self.closest_node(point)
            #print(closest_node_id)
            #print(f"Sampled point {point}")
            #print(f"Closest point {self.nodes[closest_node_id].point}")

            #Simulate driving the robot towards the closest point
            trajectory_o = self.simulate_trajectory(self.nodes[closest_node_id].point, point)

            arrived_to_pt = trajectory_o[-1][:, None]  # (3, 1)
            #print(f"Arrived to point: {arrived_to_pt}")
            self.window.add_point(arrived_to_pt[:-1, 0].copy())
            assert not self.is_colliding(arrived_to_pt[:-1])

            self.nodes.append(Node(point = arrived_to_pt, parent_id = closest_node_id, cost = 0)) # no cost for RRT

            if np.linalg.norm(arrived_to_pt[:-1] - self.goal_point) < self.stopping_dist: # reached goal
                break

            i += 1

        return self.nodes
    
    def rrt_star_planning(self):
        #This function performs RRT* for the given map and robot        
        for i in range(1): #Most likely need more iterations than this to complete the map!
            #Sample
            point = self.sample_map_space()

            #Closest Node
            closest_node_id = self.closest_node(point)

            #Simulate trajectory
            trajectory_o = self.simulate_trajectory(self.nodes[closest_node_id].point, point)

            #Check for Collision
            print("TO DO: Check for collision.")

            #Last node rewire
            print("TO DO: Last node rewiring")

            #Close node rewire
            print("TO DO: Near point rewiring")

            #Check for early end
            print("TO DO: Check for early end")
        return self.nodes
    
    def recover_path(self, node_id = -1):
        path = [self.nodes[node_id].point]
        current_node_id = self.nodes[node_id].parent_id
        while current_node_id > -1:
            path.append(self.nodes[current_node_id].point)
            current_node_id = self.nodes[current_node_id].parent_id
        path.reverse()
        return path

def main():
    #Set map information
    map_filename = "willowgarageworld_05res.png"
    map_setings_filename = "willowgarageworld_05res.yaml"

    #robot information
    goal_point = np.array([[10], [10]]) #m
    stopping_dist = 0.5 #m

    #RRT precursor
    path_planner = PathPlanner(map_filename, map_setings_filename, goal_point, stopping_dist)
    #nodes = path_planner.rrt_star_planning()
    nodes = path_planner.rrt_planning()
    path = path_planner.recover_path()
    node_path_metric = np.hstack(path)

    last_node = path[0]
    for node in path[1:]:
        path_planner.window.add_line(last_node[:2, 0].copy(), node[:2, 0].copy(), width = 3, color = (0, 0, 255))
        last_node = node
    input()

    #Leftover test functions
    np.save("shortest_path.npy", node_path_metric)


if __name__ == '__main__':
    main()

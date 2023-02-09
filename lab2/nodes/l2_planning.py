#!/usr/bin/env python3
#Standard Libraries
import numpy as np
import scipy
from copy import copy
import yaml
import os
import pygame
import time
#from nodes.utils import get_maps_dir //Ben's Config
from utils import get_maps_dir  #Jerry's Config
import pygame_utils
import matplotlib.image as mpimg
import math
from skimage.draw import disk
from functools import cached_property
from scipy.linalg import block_diag
from tf import transformations
import scipy

np.random.seed(11) #11 Works well for rtt*

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
        self.traj_to_children  = {}

def define_circle(p1, p2, p3):
    """
    Returns the center and radius of the circle passing the given 3 points.
    In case the 3 points form a line, returns (None, infinity).
    """
    temp = p2[0] * p2[0] + p2[1] * p2[1]
    bc = (p1[0] * p1[0] + p1[1] * p1[1] - temp) / 2
    cd = (temp - p3[0] * p3[0] - p3[1] * p3[1]) / 2
    det = (p1[0] - p2[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p2[1])
    
    if abs(det) < 1.0e-6:
        return (None, np.inf)
    
    # Center of circle
    cx = (bc*(p2[1] - p3[1]) - cd*(p1[1] - p2[1])) / det
    cy = ((p1[0] - p2[0]) * cd - (p2[0] - p3[0]) * bc) / det
    
    radius = np.sqrt((cx - p1[0])**2 + (cy - p1[1])**2)
    return np.array([cx, cy]), radius

#Path Planner 
class PathPlanner:
    #A path planner capable of perfomring RRT and RRT*
    def __init__(self, map_filename, map_setings_filename, goal_point, stopping_dist):
        #Get map information
        self.occupancy_map = load_map(map_filename)
        if len(self.occupancy_map.shape) == 3:
            self.occupancy_map = self.occupancy_map[:, :, 0]
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
        self.vel_max = 0.2 # 0.5 #m/s (Feel free to change!)
        self.rot_vel_max = 1.82 #0.2 #rad/s (Feel free to change!)

        #Goal Parameters
        self.goal_point = goal_point #m
        self.stopping_dist = stopping_dist #m

        #Trajectory Simulation Parameters
        self.timestep = 5.0 #5.0 #s
        self.num_substeps = 10

        #Planning storage
        self.start_point = np.array([[0], [0], [0]])
        self.nodes = [Node(self.start_point, -1, 0)] # assuming robot starts at origin
        self.node_pts = self.start_point[:2][None] # (N, 2, 1)
        self.min_pt = self.start_point
        self.max_pt = self.start_point

        #RRT* Specific Parameters
        self.lebesgue_free = np.sum(self.occupancy_map) * self.map_settings_dict["resolution"] **2
        self.zeta_d = np.pi
        self.gamma_RRT_star = 2 * (1 + 1/2) ** (1/2) * (self.lebesgue_free / self.zeta_d) ** (1/2)
        self.gamma_RRT = self.gamma_RRT_star + .1
        self.epsilon = 2.5

        self.saved_trajs = {}
        
        #Pygame window for visualization
        if map_filename == "willowgarageworld_05res.png":
            self.bounds = np.array([[-3.5, 43.5],[-49.25, 10.5]])
            sh = self.occupancy_map.shape
        else:
            sh = (self.occupancy_map.shape[1] * 5, self.occupancy_map.shape[0] * 5)
        self.window = pygame_utils.PygameWindow(
            "Path Planner", sh, self.occupancy_map.T.shape, self.map_settings_dict, self.goal_point, self.stopping_dist, map_filename)

        self.goal_nodes = {}
        self.best_goal_node_id = None

        self.min_cost = np.linalg.norm(self.goal_point[:2] - self.start_point[:2])

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

    def get_max_cost(self):
        if self.best_goal_node_id is None:
            return np.inf
            #return np.linalg.norm(self.bounds[0] - self.bounds[1], ord = 1)
        else:
            return self.goal_nodes[self.best_goal_node_id].cost

    def sample_map_space(self):
        sample_goal = np.random.rand() < 0.05

        if not sample_goal:
            #x = np.random.rand() * (self.bounds[0, 1] - self.bounds[0, 0])  + self.bounds[0, 0] 
            #y = np.random.rand() * (self.bounds[1, 1] - self.bounds[1, 0])  + self.bounds[1, 0] 
            #theta = np.random.rand() * 2 * np.pi  - np.pi # [pi, -pi]
            #point = np.array([[x], [y], [theta]])

            #other_pt = point + np.random.randn(3, 1)
            #other_pt[-1] = theta

            #col1 = self.is_colliding(point[:2])
            #col2 = self.is_colliding(other_pt[:2])

            #if not col1 and col2:
            #    return point
            #elif not col2 and col1:
            #    return other_pt
            #while True:
            x = np.random.rand() * (self.bounds[0, 1] - self.bounds[0, 0])  + self.bounds[0, 0] 
            y = np.random.rand() * (self.bounds[1, 1] - self.bounds[1, 0])  + self.bounds[1, 0] 
            theta = np.random.rand() * 2 * np.pi  - np.pi # [pi, -pi]
            point = np.array([[x], [y], [theta]])
            #cost = np.linalg.norm(self.start_point[:2] - point[:2]) + np.linalg.norm(point[:2] - self.goal_point[:2])
            #    if cost < self.get_max_cost():
            #        break
        else:
            theta = np.random.rand() * 2 * np.pi  - np.pi # [pi, -pi]
            dx = 4 * self.stopping_dist * np.random.randn()
            dy = 4 * self.stopping_dist * np.random.randn()
            point = np.array([[self.goal_point[0, 0] +dx], [self.goal_point[1, 0] + dy], [theta]])


        return point

    def check_if_duplicate(self, point):
        #Check if point is a duplicate of an already existing node
        # TODO
        # point: (3, 1) world point
        i = self.closest_node(point)[0]
        if np.all(self.nodes[i].point == point):
            return True
        return False
    
    def closest_node(self, point, k = 1):
        #Returns the index of the closest node
        # TODO: replace with nearest neighbours

        #best_ind = None
        #min_dist = np.inf

        #for i in range(len(self.nodes)):
        #    pt = self.nodes[i].point
        #    d = np.linalg.norm(pt[:-1] - point[:-1])
        #    if d < min_dist:
        #        best_ind = i
        #        min_dist = d
        kdtree = scipy.spatial.cKDTree(self.node_pts[:, :, 0])
        d, i = kdtree.query(point[:2, 0][None], k = k)

        return i
    
    def simulate_trajectory(self, point_i, point_s, fix_col = True, type  = 1, vel_max = None, rot_vel_max = None):
        #Simulates the non-holonomic motion of the robot.
        #This function drives the robot from node_i towards point_s. This function does has many solutions!
        #node_i is a 3 by 1 vector [x;y;theta] this can be used to construct the SE(2) matrix T_{OI} in course notation
        #point_s is the sampled point vector [x; y]
        vel, rot_vel = self.robot_controller(point_i, point_s, type = type, vel_max = vel_max, rot_vel_max = rot_vel_max)
        vel = vel[None, :]
        rot_vel = rot_vel[None, :]
        robot_traj = self.trajectory_rollout(vel, rot_vel, point_i, fix_col = fix_col)[0]
        return robot_traj # (num_substeps, 3)
    
    def robot_controller(self, point_i, point_s, type = 1, vel_max = None, rot_vel_max = None, timestep = None):
        #This controller determines the velocities that will nominally move the robot from node i to node s
        #Max velocities should be enforced
        # point_i = [x_i; y_i; theta_i], shape (3, 1)
        # point_s = [x_s; y_s; theta_s], shape (3, 1)

        if vel_max is None:
            vel_max = self.vel_max
        if rot_vel_max is None:
            rot_vel_max = self.rot_vel_max
        if timestep is None:
            timestep = self.timestep

        # -------------------------------- grid search ----------------------------------------

        if type == 1:
            vel_range = np.linspace(-vel_max, vel_max, 5)
            omega_range = np.linspace(-rot_vel_max, rot_vel_max, 5)
            vo = np.dstack(np.meshgrid(vel_range, omega_range)).reshape(-1, 2)
            pts = self.trajectory_rollout(vo[:, 0:1], vo[:, 1:2], point_i)[:, -1:, :] # (N, 1, 3)
            point_s = point_s[None, :, 0]
            pts = pts[:, 0, :]
            dtheta = 0 #np.arctan2(np.sin(point_s[:, -1] - pts[:, -1]),np.cos(point_s[:, -1] - pts[:, -1]))
            best_input_ind = np.argmin(np.linalg.norm(point_s[:, :-1] - pts[:, :-1], axis = -1, ord = 1) + np.abs(dtheta), axis = 0)
            v_omega = vo[best_input_ind]
            return v_omega[0:1], v_omega[1:2] # (1, ), (1,)

        # ------------------------------- analytic solution with theta ----------------------------
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

        # ------------------------------- analytic solution without theta ----------------------------

        elif type == 2:
            raise NotImplementedError("removed this implementation")
            return v, omega
   

    
    def trajectory_rollout(self, vel, rot_vel, x_y_theta, fix_col = True, num_substeps = None, timestep = None):
        # Given your chosen velocities determine the trajectory of the robot for your given timestep
        # The returned trajectory should be a series of points to check for collisions

        # x_y_theta: (x, y, theta), (3, 1)
        # vel.shape: (N, 1)
        # rot_vel: (N, 1)

        if timestep is None:
            timestep = self.timestep
        if num_substeps is None:
            num_substeps = self.num_substeps

        t = np.linspace(0, timestep, num_substeps)[None, :] # (1, num_substeps)
        x0 = x_y_theta[0:1] # (N, 1)
        y0 = x_y_theta[1:2] # (N, 1)
        theta0 = x_y_theta[2:3] # (N, 1)

        x = np.zeros((rot_vel.shape[0], t.shape[1])) # (N, num_substeps)
        y = np.zeros((rot_vel.shape[0], t.shape[1]))

        x = np.where(np.isclose(rot_vel, 0), vel * t * np.cos(theta0) + x0, (vel / rot_vel)  * (np.sin(rot_vel * t + theta0) - np.sin(theta0)) + x0)
        y = np.where(np.isclose(rot_vel, 0), vel * t * np.sin(theta0) + y0, -(vel / rot_vel)  * (np.cos(rot_vel * t + theta0) - np.cos(theta0)) + y0)
        theta = np.where(np.isclose(rot_vel, 0), theta0 * np.ones_like(rot_vel * t), rot_vel * t + theta0)

        # to check collisions, check if any of the x,y pairs result in collision.
        # take the maximum timestep that has no collision and copy that to all timesteps
        pts = np.stack((x, y), axis = -1) # (N, self.num_substeps, 2)
        p_w = pts.reshape((-1, 2)).T # (2, N * self.num_substeps)
        collisions = self.is_colliding(p_w).reshape((vel.shape[0], num_substeps)) # (N*self.num_substeps)

        res = np.stack((x, y, theta), axis = -1) 
        if fix_col:
            np.maximum.accumulate(collisions, axis = 1, out = collisions)

            mask = (collisions == 1)
            idx = np.where(~mask, np.arange(mask.shape[1]), 0)
            np.maximum.accumulate(idx, axis = 1, out = idx)
            res = res[np.arange(idx.shape[0])[:, None], idx]
        else:
            res[collisions == 1] = np.nan

        return res # (N, self.num_substeps, 3)

    def is_colliding(self, p_w):
        # p_w.shape = (2, N)
        footprints = self.points_to_robot_circle(p_w) # (N, pts_per_circle, 2)
        # TODO: batch over footprints. Problem: some footprints have different number of points
        #res = np.zeros((p_w.shape[1], )) # (N, )
        rows = footprints[..., 1]
        #rows[rows >= self.occupancy_map.shape[0]] = self.occupancy_map.shape[0] - 1
        #rows[rows < 0] = 0
        cols = footprints[..., 0]
        #cols[cols >= self.occupancy_map.shape[1]] = self.occupancy_map.shape[1] - 1
        #cols[cols < 0] = 0
        is_col = np.any(self.occupancy_map[rows, cols] == 0, axis = -1)# (N, pts_per_circle)
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
        assert p_w.shape == (2, num_pts)
        p_w = p_w.T
        p_w = np.concatenate((p_w, np.zeros_like(p_w[:, -1:]), np.ones_like(p_w[:, -1:])), axis = -1)[:, :, None] # (N, 4, 1)
        p_m = self.t_mw @ p_w
        p_m /= res
        x_idx = p_m[:, 0].astype(int)
        y_idx = (self.map_shape[0] - p_m[:, 1]).astype(int) # Note that image coordinate y-axis points down
        return np.concatenate((x_idx, y_idx), axis = -1) # (N, 2)

    def points_to_robot_circle(self, points):
        #Convert a series of [x,y] points to robot map footprints for collision detection
        #Hint: The disk function is included to help you with this function
        #print("TO DO: Implement a method to get the pixel locations of the robot path")
        inds = self.point_to_cell(points) # px locations (N, 2)
        res = self.map_settings_dict["resolution"]
        radius = self.robot_radius / res # radius in px
        footprints = [] # one for each point

        N = inds.shape[0]
        indexings = np.arange(np.floor(-radius), np.ceil(radius),dtype=np.int64)

        # Creates a addition map that adds onto center point to produce points in circle
        x_indexings, y_indexings= np.meshgrid(indexings,indexings)
        x_indexings = np.tile(x_indexings.ravel(order='F'),(N,1)) # (N,len_indexings^2)
        y_indexings = np.tile(y_indexings.ravel(order='F'),(N,1)) # (N,len_indexings^2)
        distances = np.sqrt(x_indexings**2 + y_indexings**2) # (N,len_indexings^2)

        x_circlePts = np.tile(inds[:,0],(x_indexings.shape[1],1)).T + x_indexings # (N,len_indexings^2)
        y_circlePts = np.tile(inds[:,1],(y_indexings.shape[1],1)).T  + y_indexings # (N,len_indexings^2)

        x_circlePts = x_circlePts[distances < radius].reshape((N,-1))
        x_circlePts[x_circlePts >= self.occupancy_map.shape[1]] = self.occupancy_map.shape[1] - 1
        x_circlePts[x_circlePts<0]=0

        y_circlePts = y_circlePts[distances < radius].reshape((N,-1))
        y_circlePts[y_circlePts >= self.occupancy_map.shape[0]] = self.occupancy_map.shape[0] - 1
        y_circlePts[y_circlePts<0]=0
        
        circlePts = np.stack((x_circlePts,y_circlePts),axis=-1) # (num_pts, num_pts_per_circle, 2)

        """
        # Unbatched
        for ind in inds:
            rr, cc = disk((ind[0], ind[1]), radius) #, shape = self.map_shape)
            rr[rr >= self.occupancy_map.shape[0]] = self.occupancy_map.shape[0] - 1
            cc[cc >= self.occupancy_map.shape[1]] = self.occupancy_map.shape[1] - 1
            rr[rr < 0] = 0
            cc[cc < 0] = 0
            footprints.append(np.stack((rr,cc), axis = -1))

        circle=np.stack(footprints, axis = 0)
        print(np.sum(circlePts-circle))
        exit()
        """

        return circlePts # (num_pts, num_pts_per_circle, 2)
    #Note: If you have correctly completed all previous functions, then you should be able to create a working RRT function

    #RRT* specific functions
    def ball_radius(self):
        #Close neighbor distance
        card_V = len(self.nodes)
        return min(self.gamma_RRT * (np.log(card_V) / card_V ) ** (1.0/2.0), self.epsilon)
    
    def connect_node_to_point(self, point_i, point_s):
        #Given two nodes find the non-holonomic path that connects them
        #Settings
        #point_i is a 3 by 1 point
        #point_s is a 3 by 1 poinst
        #traj = self.simulate_trajectory(point_i, point_s, fix_col=False, type = 2, vel_max = 10, rot_vel_max = 10)
        # traj will have nan's if there is a collision

        t_wv = transformations.euler_matrix(0, 0, point_i[2])
        t_wv[0, -1] = point_i[0]
        t_wv[1, -1] = point_i[1]

        t_vw = np.linalg.inv(t_wv)
        ps_w = np.concatenate((point_s[:2], np.zeros_like(point_s[-1:]), np.ones_like(point_s[-1:])), axis = 0) # (4, 1)
        ps_v = (t_vw @ ps_w)
        pmirror_v = ps_v.copy() 
        pmirror_v[0] *= -1
        pmirror_w = t_wv @ pmirror_v

        center_v, radius = define_circle(np.zeros_like(point_i[:2]), ps_v[:2], pmirror_v[:2]) 
        # determine arc length


        d = np.linalg.norm(point_i[:-1] - point_s[:-1])
        if d == 0:
            substeps = 3
            traj = np.linspace(point_i, point_s, num = substeps) # (10, 3, 1)
        elif not np.isfinite(radius): # straight ahead or behind
            # check at sufficient resolution
            substeps = np.ceil(d / self.robot_radius).astype(np.int) + 1
            traj = np.linspace(point_i, point_s, num = substeps) # (10, 3, 1)
        else:
            c_phi = 1 - ((d**2) / (2 * radius**2))
            dphi = np.arctan2(np.sqrt(1 - c_phi**2), c_phi)
            s = dphi * radius
            substeps = np.ceil(s / self.robot_radius).astype(np.int).item()

            xc_v = center_v[0]
            yc_v = center_v[1]

            cur_phi = 0
            traj = []
            for i in range(substeps + 1):
                xy_v = np.array(
                    [
                        [xc_v + radius * np.sin(cur_phi) * np.sign(ps_v[0])],
                        [yc_v - radius * np.cos(cur_phi) * np.sign(yc_v)],
                        [0],
                        [1],
                    ]
                )
                xy_w = t_wv @ xy_v
                traj.append(
                    np.array(
                        [
                            xy_w[0, 0],
                            xy_w[1, 0],
                            cur_phi * np.sign(yc_v) + point_i[-1, 0],
                        ]    
                    )
                )
                cur_phi += (dphi / substeps) 
            traj = np.stack(traj, axis = 0)

        col_mask = self.is_colliding(traj[:, :-1, 0].T)
        traj[col_mask == 1] = np.nan
        traj = traj[:, :, 0]
        assert len(traj.shape) == 2
        assert traj.shape[1] == 3

        return traj
    
    def cost_to_come(self, trajectory_o):
        dist = np.linalg.norm(trajectory_o[1:, :2] - trajectory_o[:-1, :2], axis = -1).sum()
        return dist
    
    def update_children(self, node_id):
        #Given a node_id with a changed cost, update all connected nodes with the new cost
        node = self.nodes[node_id]
        children = node.children_ids
        for child in children:
            #traj = self.connect_node_to_point(node.point, self.nodes[child].point)
            traj = node.traj_to_children[child]
            assert not np.any(np.isnan(traj))
            self.nodes[child].cost = node.cost + self.cost_to_come(traj)
            self.update_children(child)
        return

    #Planner Functions
    def rrt_planning(self):
        #This function performs RRT on the given map and robot
        #You do not need to demonstrate this function to the TAs, but it is left in for you to check your work
        #for i in range(1): #Most likely need more iterations than this to complete the map!
        i = 0
        while True:
            #Sample map space
            print(i)
            point = self.sample_map_space()
            self.window.add_point(point[:-1, 0].copy(), color = (255, 0, 0))
            self.window.check_for_close()

            #Get the closest point
            closest_node_id = self.closest_node(point)[0]
            #print(closest_node_id)
            #print(f"Sampled point {point}")
            #print(f"Closest point {self.nodes[closest_node_id].point}")

            #Simulate driving the robot towards the closest point
            trajectory_o = self.simulate_trajectory(self.nodes[closest_node_id].point, point, fix_col = True)
            if np.any(np.isnan(trajectory_o)):
                continue

            arrived_to_pt = trajectory_o[-1][:, None]  # (3, 1)

            if self.check_if_duplicate(arrived_to_pt):
                continue
            #print(f"Arrived to point: {arrived_to_pt}")
            self.window.add_point(arrived_to_pt[:-1, 0].copy())
            assert not self.is_colliding(arrived_to_pt[:-1])


            self.nodes.append(Node(point = arrived_to_pt, parent_id = closest_node_id, cost = 0)) # no cost for RRT
            self.node_pts = np.concatenate((self.node_pts, arrived_to_pt[:2][None]), axis = 0)
            self.nodes[closest_node_id].children_ids.append(len(self.nodes) - 1)
            self.nodes[closest_node_id].traj_to_children[len(self.nodes) - 1] = trajectory_o

            if np.linalg.norm(arrived_to_pt[:-1] - self.goal_point) < self.stopping_dist: # reached goal
                self.goal_nodes[len(self.nodes) - 1] = self.nodes[-1]
                self.best_goal_node_id = len(self.nodes) - 1
                break

            i += 1

        return self.nodes
    
    def rrt_star_planning(self, max_iters = 4000):
        #This function performs RRT* for the given map and robot        
        i = 0
        while len(self.goal_nodes) == 0 or i < max_iters:
            print(i)

            #Sample
            point = self.sample_map_space()
            self.window.add_point(point[:-1, 0].copy(), color = (255, 0, 0))
            self.window.check_for_close()

            #Closest Node
            closest_node_id = self.closest_node(point)[0]

            #Simulate trajectory
            trajectory_o = self.simulate_trajectory(self.nodes[closest_node_id].point, point, fix_col = True)
            if np.any(np.isnan(trajectory_o)):
                continue

            arrived_to_pt = trajectory_o[-1][:, None]  # (3, 1)

            if self.check_if_duplicate(arrived_to_pt):
                #print("found duplicate")
                continue


            #print(f"Arrived to point: {arrived_to_pt}")
            self.window.add_point(arrived_to_pt[:-1, 0].copy())
            assert not self.is_colliding(arrived_to_pt[:-1])
            cost = self.cost_to_come(trajectory_o) + self.nodes[closest_node_id].cost
            self.nodes.append(Node(arrived_to_pt, parent_id = closest_node_id, cost = cost))
            self.node_pts = np.concatenate((self.node_pts, arrived_to_pt[:2][None]), axis = 0)
            self.nodes[closest_node_id].children_ids.append(len(self.nodes) - 1)
            self.nodes[closest_node_id].traj_to_children[len(self.nodes) - 1] = trajectory_o

            i += 1

            self.min_pt = np.minimum(self.min_pt, arrived_to_pt)
            self.max_pt = np.maximum(self.max_pt, arrived_to_pt)

            #Last node rewire
            #x = lambda node : np.linalg.norm(arrived_to_pt[:-1] - node.point[:-1])
            #dist = np.array(list(map(x, self.nodes[:-1])))
            #close_idx = np.where(dist < self.ball_radius())[0]
            kdtree = scipy.spatial.cKDTree(self.node_pts[:-1, :, 0])
            close_idx = kdtree.query_ball_point(arrived_to_pt[:2,0], r = self.ball_radius())
            best_ctc = np.inf
            best_id = closest_node_id
            best_traj = None
            for id in close_idx:
                new_traj = self.connect_node_to_point(self.nodes[id].point, arrived_to_pt)
                if np.any(np.isnan(new_traj)):
                    continue
                assert np.allclose(new_traj[0, :2], self.nodes[id].point[:2, 0]), f"{new_traj[0]}\n{self.nodes[id].point}"
                assert np.allclose(new_traj[-1, :2], arrived_to_pt[:2, 0]), f"{new_traj[-1, :2]}\n{arrived_to_pt[:2, 0]}"
                curr_ctc = self.cost_to_come(new_traj) + self.nodes[id].cost
                if curr_ctc < best_ctc:
                    best_ctc = curr_ctc
                    best_id = id
                    best_traj = new_traj

            if best_id != closest_node_id:
                self.nodes[-1].parent_id = best_id
                self.nodes[-1].cost = best_ctc
                self.nodes[best_id].children_ids.append(len(self.nodes) - 1)
                self.nodes[best_id].traj_to_children[len(self.nodes) - 1] = best_traj
                self.nodes[closest_node_id].children_ids.remove(len(self.nodes) - 1)
                del self.nodes[closest_node_id].traj_to_children[len(self.nodes) - 1]


            #Close node rewire
            #print("TO DO: Near point rewiring")
            for id in close_idx:
                new_traj = self.connect_node_to_point(arrived_to_pt, self.nodes[id].point)
                if np.isnan(new_traj).any():
                    continue
                assert np.allclose(new_traj[0, :2], arrived_to_pt[:2, 0])
                assert np.allclose(new_traj[-1, :-1], self.nodes[id].point[:-1, 0])
                new_ctc = self.cost_to_come(new_traj) + self.nodes[-1].cost
                if new_ctc < self.nodes[id].cost:
                    self.nodes[self.nodes[id].parent_id].children_ids.remove(id)
                    self.nodes[id].parent_id = len(self.nodes) - 1
                    self.nodes[id].cost = new_ctc
                    self.nodes[-1].children_ids.append(id)
                    self.nodes[-1].traj_to_children[id] = new_traj
                    self.update_children(id)

            if np.linalg.norm(arrived_to_pt[:-1] - self.goal_point) < self.stopping_dist: # reached goal
                self.goal_nodes[len(self.nodes) - 1] = self.nodes[-1]
                if self.best_goal_node_id is None or self.goal_nodes[self.best_goal_node_id].cost > self.nodes[-1].cost:
                    self.best_goal_node_id = len(self.nodes) - 1

        return self.nodes

    def recover_path(self):
        path = [(self.nodes[self.best_goal_node_id], self.best_goal_node_id)]
        current_node_id = self.nodes[self.best_goal_node_id].parent_id
        while current_node_id > -1:
            path.append((self.nodes[current_node_id], current_node_id))
            current_node_id = self.nodes[current_node_id].parent_id
        path.reverse()
        return path

def main():
    #Set map information for willow
    #map_filename = "willowgarageworld_05res.png"
    #map_settings_filename = "willowgarageworld_05res.yaml"
    #goal_point = np.array([[42], [-44]]) #m
    #goal_point = np.array([[20], [0]]) #m

    # set map info for myhal
    map_filename = "myhal.png"
    map_settings_filename = "myhal.yaml"
    goal_point = np.array([[7], [0]]) #m

    #robot information
    stopping_dist = 0.5 #m

    #RRT precursor
    path_planner = PathPlanner(map_filename, map_settings_filename, goal_point, stopping_dist)
    method = "rrt_star"
    if method == "rrt_star":
        nodes = path_planner.rrt_star_planning(max_iters = 400)
    else:
        nodes = path_planner.rrt_planning()
    path = path_planner.recover_path()

    #path = np.load("/home/agrobenj/catkin_ws/src/rob521_labs/lab2/nodes/path_complete.npy").T[:, :, None]
    #print(path.shape)
    node_path_metric = np.hstack(np.array([n[0].point for n in path]))

    plot_full = True

    if plot_full:
        for (n1, n1_id), (n2, n2_id) in zip(path[:-1], path[1:]):
            trajectory = n1.traj_to_children[n2_id]
            for pt1, pt2 in zip(trajectory[:-1], trajectory[1:]):
                path_planner.window.add_line(pt1[:2].copy(), pt2[:2].copy(), width = 3, color = (0, 0, 255))
                #last_node = node
    else:
        for (n1, n1_id), (n2, n2_id) in zip(path[:-1], path[1:]):
            path_planner.window.add_line(n1.point[:2, 0].copy(), n2.point[:2, 0].copy(), width = 3, color = (0, 0, 255))
        
    input("Press Enter to Save")

    np.save(f"shortest_path_{method}_{os.path.splitext(map_filename)[0]}.npy", node_path_metric)


if __name__ == '__main__':
    main()

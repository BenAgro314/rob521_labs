import numpy as np
from nodes.l2_planning import PathPlanner
from nodes.l2_planning import Node
from utils import get_maps_dir
import imageio

def main():
    #Set map information

    map_filename = "willowgarageworld_05res.png"

    #map_setings_filename = "willowgarageworld_05res.yaml"

    #map_filename = "simple_map.png"

    map_setings_filename = "willowgarageworld_05res.yaml"

    #robot information
    goal_point = np.array([[10], [10]]) #m
    stopping_dist = 0.5 #m

    # point to cell
    path_planner = PathPlanner(map_filename, map_setings_filename, goal_point, stopping_dist)

    #point_i = np.array([[0], [0], [0]])
    #point_s = np.array([[0], [0.5], [0]])
    #trajectory = path_planner.simulate_trajectory(point_i, point_s)
    #print(trajectory)
    #xy = trajectory[:, :-1].T # (2, 10)
    #covered_inds = path_planner.points_to_robot_circle(xy)
    #occ_map = path_planner.occupancy_map.copy()
    #for i, footprint in enumerate(covered_inds):
    #    occ_map[footprint[:, 1], footprint[:, 0]] = i / 10

    #imageio.imsave("out_test.png", occ_map)

    nodes = path_planner.rrt_planning()
    pass


if __name__ == "__main__":
    main()
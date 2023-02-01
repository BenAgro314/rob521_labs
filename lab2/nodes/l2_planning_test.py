import numpy as np
from nodes.l2_planning import PathPlanner
from utils import get_maps_dir
import imageio

def main():
    #Set map information

    map_filename = "willowgarageworld_05res.png"

    map_setings_filename = "willowgarageworld_05res.yaml"

    #robot information
    goal_point = np.array([[10], [10]]) #m
    stopping_dist = 0.5 #m

    # point to cell
    path_planner = PathPlanner(map_filename, map_setings_filename, goal_point, stopping_dist)
    p_w=np.array(
        [
            [i for i in range(0, 20, 2)],
            [i for i in range(0, 10, 1)]
        ]
    )
    print(p_w)
    inds = path_planner.point_to_cell(p_w)
    print(inds)
    covered_inds = path_planner.points_to_robot_circle(p_w)
    occ_map = path_planner.occupancy_map.copy()
    for i, footprint in enumerate(covered_inds):
        occ_map[footprint[:, 1], footprint[:, 0]] = i / 10

    #imageio.imsave("out_test.png", occ_map)

    x_y_theta = np.array([[5], [10], [np.pi/2]])
    vel = 10
    rot_vel = -np.pi
    pts = path_planner.trajectory_rollout(vel, rot_vel, x_y_theta) # (10, 3)
    xy = pts[..., :-1].T # (2, 10)
    covered_inds = path_planner.points_to_robot_circle(xy)
    occ_map = path_planner.occupancy_map.copy()
    for i, footprint in enumerate(covered_inds):
        occ_map[footprint[:, 1], footprint[:, 0]] = i / 10

    imageio.imsave("out_test.png", occ_map)
    pass


if __name__ == "__main__":
    main()
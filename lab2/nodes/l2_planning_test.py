import numpy as np
from nodes.l2_planning import PathPlanner
from nodes.l2_planning import Node
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

    #x_y_theta = np.array([[5], [10], [np.pi/2]])
    """
    x_y_theta = np.array([[0], [0], [0]])
    pt2 = np.array([[0], [1], [-np.pi]])
    vel, rot_vel = path_planner.robot_controller(x_y_theta, pt2)
    vel = np.array([vel])[None, :]
    rot_vel = np.array([rot_vel])[None, :]
    print(vel, rot_vel)
    #vel = 10
    #rot_vel = -np.pi
    pts = path_planner.trajectory_rollout(vel, rot_vel, x_y_theta) # (1, 10, 3)
    print(pts[0, -1])
    xy = pts[0, :, :-1].T # (2, 10)
    covered_inds = path_planner.points_to_robot_circle(xy)
    occ_map = path_planner.occupancy_map.copy()
    for i, footprint in enumerate(covered_inds):
        occ_map[footprint[:, 1], footprint[:, 0]] = i / 10

    imageio.imsave("out_test.png", occ_map)
    """

    node_i = Node(point = np.array([[0], [0], [0]]), parent_id=0, cost = 0)
    point_s = np.array([[0.2], [0.0], [0.0]])
    trajectory = path_planner.simulate_trajectory(node_i, point_s)
    xy = trajectory[0, :, :-1].T # (2, 10)
    covered_inds = path_planner.points_to_robot_circle(xy)
    occ_map = path_planner.occupancy_map.copy()
    for i, footprint in enumerate(covered_inds):
        occ_map[footprint[:, 1], footprint[:, 0]] = i / 10

    imageio.imsave("out_test.png", occ_map)


    pass


if __name__ == "__main__":
    main()
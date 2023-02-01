import numpy as np
from nodes.l2_planning import PathPlanner
from utils import get_maps_dir

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
            [0, 1, 0],
            [0, 0, 1],
        ]
    )
    inds = path_planner.point_to_cell(p_w)
    print(inds)

    covered_inds = path_planner.points_to_robot_circle(p_w)


if __name__ == "__main__":
    main()
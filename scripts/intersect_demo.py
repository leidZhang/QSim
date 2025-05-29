import sys
sys.path.insert(0, sys.path[0] + "/..")
from core.roadmap import ACCRoadMap
from environment.utils import is_waypoint_intersected

if __name__ == '__main__':
    roadmap: ACCRoadMap = ACCRoadMap()
    route_1, route_0 = [1, 13, 19, 17, 15], [12, 8, 23, 21]
    traj_1, traj_0 = roadmap.generate_path(route_1), roadmap.generate_path(route_0)
    print(is_waypoint_intersected(traj_1, traj_0))